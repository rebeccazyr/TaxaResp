#!/usr/bin/env python3
"""Compare smoke-set cross-domain classifier heuristics against manual labels."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Callable


DEFAULT_FEATURES_TSV = "outputs/cross_domain_eval_selection_smoke_200/smoke_200_cross_domain_all_features.tsv"
DEFAULT_MANUAL_TSV = "outputs/cross_domain_eval_selection_smoke_200/smoke_200_manual_cross_domain_annotations.tsv"
DEFAULT_OUT_DIR = "outputs/cross_domain_eval_selection_smoke_200"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-tsv", default=DEFAULT_FEATURES_TSV)
    parser.add_argument("--manual-tsv", default=DEFAULT_MANUAL_TSV)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def metric_row(name: str, preds: list[bool], gold: list[bool], target: str) -> dict[str, object]:
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
        "selected": tp + fp,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "accuracy": (tp + tn) / len(gold) if gold else 0.0,
        "balanced_accuracy": (recall + specificity) / 2,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def parse_float(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def is_true(row: dict[str, str], column: str) -> bool:
    return row.get(column, "") == "1"


def numeric_columns(rows: list[dict[str, str]]) -> list[str]:
    skip = {
        "paper_id",
        "title",
        "manual_label",
        "final_label",
        "label_source",
        "reason",
        "task_labels",
        "covered_assignments",
        "level0_projected_labels",
        "level1_projected_labels",
        "level2_projected_labels",
    }
    columns: list[str] = []
    for column in rows[0].keys():
        if column in skip:
            continue
        parsed = [parse_float(row.get(column, "")) for row in rows]
        if any(not math.isnan(value) for value in parsed):
            columns.append(column)
    return columns


def best_threshold(
    rows: list[dict[str, str]],
    gold: list[bool],
    target: str,
    column: str,
    optimize: str,
) -> dict[str, object]:
    values = [parse_float(row.get(column, "")) for row in rows]
    thresholds = sorted({value for value in values if not math.isnan(value)})
    candidates: list[dict[str, object]] = []
    for threshold in thresholds:
        candidates.append(
            metric_row(
                f"{column}>={threshold:g}",
                [value >= threshold for value in values],
                gold,
                target,
            )
        )
        candidates.append(
            metric_row(
                f"{column}<={threshold:g}",
                [value <= threshold for value in values],
                gold,
                target,
            )
        )
    if not candidates:
        return metric_row(f"{column}:no_threshold", [False] * len(gold), gold, target)
    return max(
        candidates,
        key=lambda row: (
            float(row[optimize]),
            float(row["f1"]),
            float(row["balanced_accuracy"]),
            float(row["precision"]),
        ),
    )


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manual_labels = {row["paper_id"]: row["manual_label"] for row in read_tsv(Path(args.manual_tsv))}
    rows = []
    for row in read_tsv(Path(args.features_tsv)):
        label = manual_labels.get(row["paper_id"])
        if label:
            row["manual_label"] = label
            rows.append(row)

    fixed_rules: list[tuple[str, Callable[[dict[str, str]], bool]]] = [
        ("all_no_baseline", lambda row: False),
        ("all_yes_baseline", lambda row: True),
        ("fos_candidate", lambda row: is_true(row, "fos_candidate")),
        ("fos_strict", lambda row: is_true(row, "fos_strict")),
        ("default_dispersion_pass", lambda row: is_true(row, "default_dispersion_pass")),
        ("high_precision_dispersion_pass", lambda row: is_true(row, "high_precision_dispersion_pass")),
        ("author_jsd_l0_mean>=0.041", lambda row: parse_float(row.get("author_jsd_l0_mean", "")) >= 0.041),
        (
            "fos_candidate AND high_precision_dispersion",
            lambda row: is_true(row, "fos_candidate") and is_true(row, "high_precision_dispersion_pass"),
        ),
        (
            "fos_candidate AND author_jsd_l0_mean>=0.041",
            lambda row: is_true(row, "fos_candidate") and parse_float(row.get("author_jsd_l0_mean", "")) >= 0.041,
        ),
        (
            "high_precision_dispersion AND author_jsd_l0_mean>=0.041",
            lambda row: is_true(row, "high_precision_dispersion_pass")
            and parse_float(row.get("author_jsd_l0_mean", "")) >= 0.041,
        ),
        (
            "fos_candidate AND high_precision_dispersion AND author_jsd_l0_mean>=0.041",
            lambda row: is_true(row, "fos_candidate")
            and is_true(row, "high_precision_dispersion_pass")
            and parse_float(row.get("author_jsd_l0_mean", "")) >= 0.041,
        ),
    ]
    scan_columns = numeric_columns(rows)

    outputs: list[dict[str, object]] = []
    targets = [
        ("strict_yes", lambda label: label == "yes"),
        ("relaxed_yes_or_borderline", lambda label: label in {"yes", "borderline"}),
    ]
    for target_name, is_positive in targets:
        gold = [is_positive(row["manual_label"]) for row in rows]
        for name, fn in fixed_rules:
            outputs.append(metric_row(name, [fn(row) for row in rows], gold, target_name))
        for objective in ("accuracy", "balanced_accuracy", "f1"):
            for column in scan_columns:
                row = best_threshold(rows, gold, target_name, column, objective)
                row["method"] = f"best_{objective}:{row['method']}"
                outputs.append(row)

    out_path = out_dir / "classifier_method_comparison_smoke_200.tsv"
    fieldnames = [
        "target",
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
        "f1",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(outputs)

    print(f"rows={len(rows)}")
    print(f"wrote={out_path}")


if __name__ == "__main__":
    main()
