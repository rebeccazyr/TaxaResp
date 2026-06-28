#!/usr/bin/env python3
"""Merge fixed year-bin expert profiles into one aggregate profile directory."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List


DEFAULT_BINS = (
    "train_2000_2004",
    "train_2005_2009",
    "valid_2010_2014",
    "test_2015_2019",
)

HEADER = (
    "fos_id\tfos_name\tin_author_papers\tdirect_paper_count\t"
    "direct_weight_sum\tdirect_weight_avg\tpaper_weight_details\n"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge per-expert direct FoS profiles from year-bin folders"
    )
    parser.add_argument(
        "--profile-dir",
        default="output/expert_profile_year_bins",
        help="Root directory containing year-bin expert profile folders",
    )
    parser.add_argument(
        "--out-subdir",
        default="all_2000_2019",
        help="Output subdirectory under profile-dir",
    )
    parser.add_argument(
        "--bins",
        nargs="*",
        default=list(DEFAULT_BINS),
        help="Year-bin subdirectories to merge in order",
    )
    return parser.parse_args()


def load_profile(path: Path, merged: Dict[str, dict]) -> None:
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            fos_id = (row.get("fos_id") or "").strip()
            fos_name = row.get("fos_name") or ""
            key = fos_id if fos_id else f"__NAME__:{fos_name.lower()}"
            if not key or key == "__NAME__:":
                continue

            if key not in merged:
                merged[key] = {
                    "fos_id": fos_id,
                    "fos_name": fos_name,
                    "paper_weights": {},
                }

            try:
                details = json.loads(row.get("paper_weight_details") or "[]")
            except json.JSONDecodeError:
                details = []

            for item in details:
                if not isinstance(item, dict):
                    continue
                paper_id = str(item.get("paper_id", "")).strip()
                if not paper_id:
                    continue
                try:
                    weight = float(item.get("weight", 0.0))
                except (TypeError, ValueError):
                    weight = 0.0
                if weight <= 0.0:
                    continue

                existing = merged[key]["paper_weights"].get(paper_id)
                if existing is None or weight > existing["weight"]:
                    detail = {"paper_id": paper_id, "weight": round(weight, 5)}
                    if "year" in item:
                        detail["year"] = item["year"]
                    merged[key]["paper_weights"][paper_id] = detail


def build_rows(merged: Dict[str, dict]) -> List[dict]:
    rows: List[dict] = []
    for rec in merged.values():
        details = [rec["paper_weights"][pid] for pid in sorted(rec["paper_weights"])]
        weight_sum = sum(float(d["weight"]) for d in details)
        count = len(details)
        rows.append(
            {
                "fos_id": rec["fos_id"],
                "fos_name": rec["fos_name"],
                "in_author_papers": 1,
                "direct_paper_count": count,
                "direct_weight_sum": weight_sum,
                "direct_weight_avg": weight_sum / count if count else 0.0,
                "paper_weight_details": json.dumps(details, ensure_ascii=False),
            }
        )
    rows.sort(
        key=lambda r: (
            -r["direct_weight_sum"],
            -r["direct_paper_count"],
            r["fos_name"].lower(),
        )
    )
    return rows


def write_profile(path: Path, rows: List[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write(HEADER)
        for row in rows:
            f.write(
                f"{row['fos_id']}\t{row['fos_name']}\t{row['in_author_papers']}\t"
                f"{row['direct_paper_count']}\t{row['direct_weight_sum']:.5f}\t"
                f"{row['direct_weight_avg']:.5f}\t{row['paper_weight_details']}\n"
            )


def main() -> None:
    args = parse_args()
    profile_dir = Path(args.profile_dir)
    out_dir = profile_dir / args.out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    first_bin = profile_dir / args.bins[0]
    expert_files = sorted(first_bin.glob("*_direct_fos_nodes.tsv"))

    summary_path = out_dir / "_summary.tsv"
    with summary_path.open("w", encoding="utf-8") as summary:
        summary.write("expert_id\tpapers\tdirect_fos_nodes\toutput_file\n")
        for idx, first_file in enumerate(expert_files, start=1):
            expert_id = first_file.name.replace("_direct_fos_nodes.tsv", "")
            merged: Dict[str, dict] = {}

            for bin_name in args.bins:
                path = profile_dir / bin_name / f"{expert_id}_direct_fos_nodes.tsv"
                if path.exists():
                    load_profile(path, merged)

            rows = build_rows(merged)
            out_path = out_dir / f"{expert_id}_direct_fos_nodes.tsv"
            write_profile(out_path, rows)

            paper_ids = set()
            for rec in merged.values():
                paper_ids.update(rec["paper_weights"].keys())
            summary.write(f"{expert_id}\t{len(paper_ids)}\t{len(rows)}\t{out_path}\n")

            if idx % 1000 == 0:
                print(f"progress experts={idx}/{len(expert_files)}")

    print(f"experts={len(expert_files)}")
    print(f"output_dir={out_dir}")
    print(f"summary={summary_path}")


if __name__ == "__main__":
    main()
