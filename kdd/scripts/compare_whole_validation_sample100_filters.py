#!/usr/bin/env python3
"""Compare filtering rules on the whole-validation hist>=5 manual sample100."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Callable


DEFAULT_SAMPLE = "outputs/cross_domain_eval_selection/whole_validation_hist_ge5_sample100/full_fos_sample100_audit.jsonl"
DEFAULT_ANNOTATIONS = "outputs/cross_domain_eval_selection/whole_validation_hist_ge5_sample100/sample100_role_based_collaboration_annotations.tsv"
DEFAULT_JSD = "outputs/cross_domain_eval_selection/whole_validation_hist_ge5_sample100/sample100_author_jsd_all_levels.tsv"
DEFAULT_OUT = "outputs/cross_domain_eval_selection/whole_validation_hist_ge5_sample100/filter_method_comparison.tsv"
DEFAULT_TOP = "outputs/cross_domain_eval_selection/whole_validation_hist_ge5_sample100/filter_method_comparison_top.tsv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-jsonl", default=DEFAULT_SAMPLE)
    parser.add_argument("--annotations-tsv", default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--jsd-tsv", default=DEFAULT_JSD)
    parser.add_argument("--out-tsv", default=DEFAULT_OUT)
    parser.add_argument("--top-tsv", default=DEFAULT_TOP)
    parser.add_argument("--top-k", type=int, default=25)
    return parser.parse_args()


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def read_tsv_by_id(path: Path) -> dict[str, dict[str, str]]:
    return {row["paper_id"]: row for row in read_tsv(path)}


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def parse_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def metric_row(name: str, preds: list[bool], gold: list[bool], target: str, family: str) -> dict[str, object]:
    tp = sum(p and y for p, y in zip(preds, gold))
    fp = sum(p and not y for p, y in zip(preds, gold))
    tn = sum((not p) and (not y) for p, y in zip(preds, gold))
    fn = sum((not p) and y for p, y in zip(preds, gold))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "target": target,
        "family": family,
        "method": name,
        "selected": tp + fp,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "accuracy": (tp + tn) / len(gold) if gold else 0.0,
        "balanced_accuracy": (recall + specificity) / 2.0,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
    }


def paper_fos_counts(obj: dict, min_weight: float = 0.5) -> dict[str, int]:
    counts = {f"direct_l{level}_count": 0 for level in range(6)}
    direct_any = 0
    for row in obj.get("paper_fos") or []:
        if parse_float(row.get("weight")) < min_weight:
            continue
        level = row.get("level")
        if level is None:
            continue
        try:
            level_int = int(level)
        except (TypeError, ValueError):
            continue
        direct_any += 1
        key = f"direct_l{level_int}_count"
        counts[key] = counts.get(key, 0) + 1
    counts["direct_any_count"] = direct_any
    return counts


def build_rows(sample_path: Path, annotations_path: Path, jsd_path: Path) -> list[dict[str, object]]:
    annotations = read_tsv_by_id(annotations_path)
    jsd = read_tsv_by_id(jsd_path)
    rows = []
    for obj in iter_jsonl(sample_path):
        paper_id = str(obj.get("paper_id") or obj.get("id"))
        ann = annotations[paper_id]
        row: dict[str, object] = {
            "paper_id": paper_id,
            "title": obj.get("title", ""),
            "author_count": len(obj.get("authors") or []),
            "manual_label": ann["manual_label"],
            "direct_l2_count_sample": obj.get("direct_l2_count", 0),
        }
        row.update(paper_fos_counts(obj, min_weight=0.5))
        for key, value in jsd.get(paper_id, {}).items():
            if key.startswith("author_jsd_"):
                row[key] = parse_float(value)
        rows.append(row)
    return rows


def fixed_rules() -> list[tuple[str, str, Callable[[dict[str, object]], bool]]]:
    return [
        ("baseline", "all_negative", lambda row: False),
        ("baseline", "all_positive", lambda row: True),
        ("author_count", "author_count>=2", lambda row: parse_float(row.get("author_count")) >= 2),
        ("author_count", "author_count>=3", lambda row: parse_float(row.get("author_count")) >= 3),
        ("author_count", "author_count>=4", lambda row: parse_float(row.get("author_count")) >= 4),
        ("direct_fos", "direct_l2_count>=1", lambda row: parse_float(row.get("direct_l2_count_sample")) >= 1),
        ("direct_fos", "direct_l2_count>=2", lambda row: parse_float(row.get("direct_l2_count_sample")) >= 2),
        ("direct_fos", "direct_l2_count>=3", lambda row: parse_float(row.get("direct_l2_count_sample")) >= 3),
        ("direct_fos", "direct_l2_count>=4", lambda row: parse_float(row.get("direct_l2_count_sample")) >= 4),
        ("direct_fos", "direct_any_count>=5", lambda row: parse_float(row.get("direct_any_count")) >= 5),
        ("direct_fos", "direct_any_count>=8", lambda row: parse_float(row.get("direct_any_count")) >= 8),
        ("jsd_fixed", "author_jsd_l0_mean>=0.019", lambda row: parse_float(row.get("author_jsd_l0_mean")) >= 0.019),
        ("jsd_fixed", "author_jsd_l0_mean>=0.0524481", lambda row: parse_float(row.get("author_jsd_l0_mean")) >= 0.0524481),
        ("jsd_fixed", "author_jsd_l0_mean>=0.0603261", lambda row: parse_float(row.get("author_jsd_l0_mean")) >= 0.0603261),
        ("jsd_fixed", "author_jsd_l1_mean>=0.161422", lambda row: parse_float(row.get("author_jsd_l1_mean")) >= 0.161422),
        ("jsd_fixed", "author_jsd_l2_mean>=0.388193", lambda row: parse_float(row.get("author_jsd_l2_mean")) >= 0.388193),
        (
            "composition_fixed",
            "author_count>=3 AND author_jsd_l0_mean>=0.019",
            lambda row: parse_float(row.get("author_count")) >= 3 and parse_float(row.get("author_jsd_l0_mean")) >= 0.019,
        ),
        (
            "composition_fixed",
            "author_count>=3 AND author_jsd_l0_mean>=0.0524481",
            lambda row: parse_float(row.get("author_count")) >= 3 and parse_float(row.get("author_jsd_l0_mean")) >= 0.0524481,
        ),
        (
            "composition_fixed",
            "author_count>=3 AND author_jsd_l0_mean>=0.0603261",
            lambda row: parse_float(row.get("author_count")) >= 3 and parse_float(row.get("author_jsd_l0_mean")) >= 0.0603261,
        ),
        (
            "composition_fixed",
            "direct_l2_count>=3 AND author_count>=3",
            lambda row: parse_float(row.get("direct_l2_count_sample")) >= 3 and parse_float(row.get("author_count")) >= 3,
        ),
        (
            "composition_fixed",
            "direct_l2_count>=3 AND author_jsd_l0_mean>=0.019 AND author_count>=3",
            lambda row: parse_float(row.get("direct_l2_count_sample")) >= 3
            and parse_float(row.get("author_count")) >= 3
            and parse_float(row.get("author_jsd_l0_mean")) >= 0.019,
        ),
        (
            "composition_fixed",
            "direct_l2_count>=3 AND author_jsd_l0_mean>=0.0603261 AND author_count>=3",
            lambda row: parse_float(row.get("direct_l2_count_sample")) >= 3
            and parse_float(row.get("author_count")) >= 3
            and parse_float(row.get("author_jsd_l0_mean")) >= 0.0603261,
        ),
    ]


def numeric_columns(rows: list[dict[str, object]]) -> list[str]:
    skip = {"paper_id", "title", "manual_label"}
    columns = []
    for key in rows[0]:
        if key in skip:
            continue
        values = [parse_float(row.get(key)) for row in rows]
        if any(not math.isnan(value) for value in values):
            columns.append(key)
    return columns


def threshold_rules(rows: list[dict[str, object]]) -> list[tuple[str, str, list[bool]]]:
    rules: list[tuple[str, str, list[bool]]] = []
    for column in numeric_columns(rows):
        values = [parse_float(row.get(column)) for row in rows]
        thresholds = sorted({value for value in values if not math.isnan(value)})
        for threshold in thresholds:
            rules.append(("threshold_grid", f"{column}>={threshold:g}", [value >= threshold for value in values]))
            rules.append(("threshold_grid", f"{column}<={threshold:g}", [value <= threshold for value in values]))
    return rules


def and_grid(rows: list[dict[str, object]]) -> list[tuple[str, str, list[bool]]]:
    clauses = {
        "author_count>=2": [parse_float(row.get("author_count")) >= 2 for row in rows],
        "author_count>=3": [parse_float(row.get("author_count")) >= 3 for row in rows],
        "direct_l2_count>=2": [parse_float(row.get("direct_l2_count_sample")) >= 2 for row in rows],
        "direct_l2_count>=3": [parse_float(row.get("direct_l2_count_sample")) >= 3 for row in rows],
    }
    for column in (
        "author_jsd_direct_mean",
        "author_jsd_l0_mean",
        "author_jsd_l1_mean",
        "author_jsd_l2_mean",
        "author_jsd_l0_min",
        "author_jsd_l1_min",
    ):
        values = [parse_float(row.get(column)) for row in rows]
        for threshold in sorted({value for value in values if not math.isnan(value)}):
            clauses[f"{column}>={threshold:g}"] = [value >= threshold for value in values]

    names = list(clauses)
    rules = []
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            preds = [a and b for a, b in zip(clauses[left], clauses[right])]
            rules.append(("and_grid", f"{left} AND {right}", preds))
    return rules


def main() -> None:
    args = parse_args()
    rows = build_rows(Path(args.sample_jsonl), Path(args.annotations_tsv), Path(args.jsd_tsv))
    all_results: list[dict[str, object]] = []
    targets = [
        ("strict_role_based", lambda label: label == "role_based"),
        ("role_or_borderline", lambda label: label in {"role_based", "borderline"}),
    ]

    threshold_preds = threshold_rules(rows)
    and_preds = and_grid(rows)
    for target, positive_fn in targets:
        gold = [positive_fn(str(row["manual_label"])) for row in rows]
        for family, name, fn in fixed_rules():
            all_results.append(metric_row(name, [fn(row) for row in rows], gold, target, family))
        for family, name, preds in threshold_preds:
            all_results.append(metric_row(name, preds, gold, target, family))
        for family, name, preds in and_preds:
            all_results.append(metric_row(name, preds, gold, target, family))

    fieldnames = [
        "target",
        "family",
        "method",
        "selected",
        "tp",
        "fp",
        "tn",
        "fn",
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "specificity",
        "f1",
    ]
    out_path = Path(args.out_tsv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(all_results)

    top_rows = []
    for target, _positive_fn in targets:
        target_rows = [row for row in all_results if row["target"] == target and int(row["selected"]) > 0]
        for metric in ("f1", "balanced_accuracy", "accuracy", "precision"):
            ranked = sorted(
                target_rows,
                key=lambda row: (
                    float(row[metric]),
                    float(row["f1"]),
                    float(row["balanced_accuracy"]),
                    float(row["precision"]),
                    -int(row["selected"]),
                ),
                reverse=True,
            )[: args.top_k]
            for rank, row in enumerate(ranked, 1):
                top_row = dict(row)
                top_row["rank_metric"] = metric
                top_row["rank"] = rank
                top_rows.append(top_row)

    top_fieldnames = ["rank_metric", "rank"] + fieldnames
    top_path = Path(args.top_tsv)
    with top_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=top_fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(top_rows)

    print(f"rows={len(rows)}")
    print(f"wrote={out_path}")
    print(f"wrote={top_path}")


if __name__ == "__main__":
    main()
