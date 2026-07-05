#!/usr/bin/env python3
"""Grid author_count>=3 AND one JSD condition AND one taxonomy-node condition."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path


DEFAULT_SAMPLE = "outputs/cross_domain_eval_selection/whole_validation_hist_ge5_sample100/full_fos_sample100_audit.jsonl"
DEFAULT_ANNOTATIONS = "outputs/cross_domain_eval_selection/whole_validation_hist_ge5_sample100/sample100_role_based_collaboration_annotations.tsv"
DEFAULT_JSD = "outputs/cross_domain_eval_selection/whole_validation_hist_ge5_sample100/sample100_author_jsd_all_levels.tsv"
DEFAULT_OUT = "outputs/cross_domain_eval_selection/whole_validation_hist_ge5_sample100/filter_author_count3_jsd_taxnode_grid.tsv"
DEFAULT_TOP = "outputs/cross_domain_eval_selection/whole_validation_hist_ge5_sample100/filter_author_count3_jsd_taxnode_top.tsv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-jsonl", default=DEFAULT_SAMPLE)
    parser.add_argument("--annotations-tsv", default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--jsd-tsv", default=DEFAULT_JSD)
    parser.add_argument("--out-tsv", default=DEFAULT_OUT)
    parser.add_argument("--top-tsv", default=DEFAULT_TOP)
    parser.add_argument("--min-fos-weight", type=float, default=0.5)
    parser.add_argument("--top-k", type=int, default=30)
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


def entropy(values: list[float]) -> float:
    total = sum(values)
    if total <= 0:
        return 0.0
    out = 0.0
    for value in values:
        if value <= 0:
            continue
        p = value / total
        out -= p * math.log(p)
    return out


def tax_metrics(obj: dict, min_weight: float) -> dict[str, float]:
    level_counts = Counter()
    level_weights = Counter()
    kept_weights = []
    for item in obj.get("paper_fos") or []:
        weight = parse_float(item.get("weight"))
        if math.isnan(weight) or weight < min_weight:
            continue
        level = item.get("level")
        if level is None:
            continue
        try:
            level = int(level)
        except (TypeError, ValueError):
            continue
        level_counts[level] += 1
        level_weights[level] += weight
        kept_weights.append(weight)

    metrics: dict[str, float] = {
        "tax_direct_any_count": float(sum(level_counts.values())),
        "tax_level_count": float(len(level_counts)),
        "tax_level_entropy": entropy(list(level_weights.values())),
        "tax_weight_entropy": entropy(kept_weights),
    }
    for level in range(6):
        metrics[f"tax_direct_l{level}_count"] = float(level_counts[level])
        metrics[f"tax_direct_l{level}_weight"] = float(level_weights[level])
    # Useful aliases for comparison with earlier conversations.
    metrics["direct_l2_count"] = metrics["tax_direct_l2_count"]
    metrics["direct_l3_count"] = metrics["tax_direct_l3_count"]
    metrics["direct_l4_count"] = metrics["tax_direct_l4_count"]
    return metrics


def build_rows(args: argparse.Namespace) -> list[dict[str, object]]:
    annotations = read_tsv_by_id(Path(args.annotations_tsv))
    jsd = read_tsv_by_id(Path(args.jsd_tsv))
    rows = []
    for obj in iter_jsonl(Path(args.sample_jsonl)):
        paper_id = str(obj.get("paper_id") or obj.get("id"))
        row: dict[str, object] = {
            "paper_id": paper_id,
            "title": str(obj.get("title") or ""),
            "author_count": float(len(obj.get("authors") or [])),
            "manual_label": annotations[paper_id]["manual_label"],
        }
        row.update(tax_metrics(obj, args.min_fos_weight))
        for key, value in jsd.get(paper_id, {}).items():
            if key.startswith("author_jsd_"):
                row[key] = parse_float(value)
        rows.append(row)
    return rows


def metric_row(name: str, preds: list[bool], gold: list[bool], target: str, jsd_rule: str, tax_rule: str) -> dict[str, object]:
    tp = sum(p and y for p, y in zip(preds, gold))
    fp = sum(p and not y for p, y in zip(preds, gold))
    tn = sum((not p) and (not y) for p, y in zip(preds, gold))
    fn = sum((not p) and y for p, y in zip(preds, gold))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "target": target,
        "method": name,
        "jsd_rule": jsd_rule,
        "tax_rule": tax_rule,
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


def threshold_rules(rows: list[dict[str, object]], columns: list[str], prefix: str) -> list[tuple[str, list[bool]]]:
    rules = []
    for column in columns:
        values = [parse_float(row.get(column)) for row in rows]
        for threshold in sorted({value for value in values if not math.isnan(value)}):
            rules.append((f"{column}>={threshold:g}", [value >= threshold for value in values]))
    return rules


def main() -> None:
    args = parse_args()
    rows = build_rows(args)
    jsd_columns = [
        "author_jsd_direct_mean",
        "author_jsd_l0_mean",
        "author_jsd_l1_mean",
        "author_jsd_l2_mean",
        "author_jsd_direct_min",
        "author_jsd_l0_min",
        "author_jsd_l1_min",
        "author_jsd_l2_min",
    ]
    tax_columns = [
        "tax_direct_any_count",
        "tax_direct_l1_count",
        "tax_direct_l2_count",
        "tax_direct_l3_count",
        "tax_direct_l4_count",
        "tax_level_count",
        "tax_level_entropy",
        "tax_weight_entropy",
    ]
    jsd_rules = threshold_rules(rows, jsd_columns, "jsd")
    tax_rules = threshold_rules(rows, tax_columns, "tax")
    author_count_preds = [parse_float(row.get("author_count")) >= 3 for row in rows]

    targets = [
        ("strict_role_based", lambda label: label == "role_based"),
        ("role_or_borderline", lambda label: label in {"role_based", "borderline"}),
    ]
    results = []
    for target, positive_fn in targets:
        gold = [positive_fn(str(row["manual_label"])) for row in rows]
        for jsd_name, jsd_preds in jsd_rules:
            for tax_name, tax_preds in tax_rules:
                preds = [a and j and t for a, j, t in zip(author_count_preds, jsd_preds, tax_preds)]
                name = f"author_count>=3 AND {jsd_name} AND {tax_name}"
                results.append(metric_row(name, preds, gold, target, jsd_name, tax_name))

    fieldnames = [
        "target",
        "method",
        "jsd_rule",
        "tax_rule",
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
        writer.writerows(results)

    top_rows = []
    for target, _ in targets:
        target_rows = [row for row in results if row["target"] == target and int(row["selected"]) > 0]
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
                out = dict(row)
                out["rank_metric"] = metric
                out["rank"] = rank
                top_rows.append(out)

    top_path = Path(args.top_tsv)
    with top_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["rank_metric", "rank"] + fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(top_rows)

    print(f"rows={len(rows)}")
    print(f"jsd_rules={len(jsd_rules)} tax_rules={len(tax_rules)}")
    print(f"wrote={out_path}")
    print(f"wrote={top_path}")


if __name__ == "__main__":
    main()
