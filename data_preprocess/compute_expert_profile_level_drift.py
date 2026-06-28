#!/usr/bin/env python3
"""Compute cosine drift between expert profile year bins for FoS levels L1-L4.

This script reads per-expert direct FoS profiles produced by
build_expert_profiles_by_year_bins.py and computes:

  drift = 1 - cosine_similarity(w_from, w_to)

for adjacent bins:
- train_2000_2004 -> train_2005_2009
- train_2005_2009 -> valid_2010_2014
- valid_2010_2014 -> test_2015_2019

FoS levels are read from FieldsOfStudy.txt. MAG FoS levels 0, 1, 2, 3 are
reported as L1, L2, L3, L4 respectively.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Dict, Iterable, List, Tuple


BINS = (
    "train_2000_2004",
    "train_2005_2009",
    "valid_2010_2014",
    "test_2015_2019",
)

BIN_PAIRS = tuple(zip(BINS[:-1], BINS[1:]))

MAG_LEVEL_TO_LABEL = {
    0: "L1",
    1: "L2",
    2: "L3",
    3: "L4",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute per-expert FoS level drift between fixed year bins"
    )
    parser.add_argument(
        "--profile-dir",
        default="output/expert_profile_year_bins",
        help="Root directory containing train/valid/test year-bin expert profile folders",
    )
    parser.add_argument(
        "--fos-map",
        default="data/dblp/FieldsOfStudy.txt",
        help="Path to FieldsOfStudy.txt; column 1 is FoS id and column 6 is MAG FoS level",
    )
    parser.add_argument(
        "--out-dir",
        default="output/expert_profile_year_bins_drift",
        help="Output directory for drift TSV files",
    )
    return parser.parse_args()


def load_fos_levels(path: Path) -> Dict[str, int]:
    levels: Dict[str, int] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 6:
                continue
            fos_id = parts[0].strip()
            if not fos_id:
                continue
            try:
                levels[fos_id] = int(parts[5])
            except ValueError:
                continue
    return levels


def list_expert_ids(profile_dir: Path) -> List[str]:
    first_bin = profile_dir / BINS[0]
    expert_ids = []
    for path in sorted(first_bin.glob("*_direct_fos_nodes.tsv")):
        expert_ids.append(path.name.replace("_direct_fos_nodes.tsv", ""))
    return expert_ids


def load_profile_vectors(path: Path, fos_levels: Dict[str, int]) -> Dict[str, Dict[str, float]]:
    vectors: Dict[str, Dict[str, float]] = {label: {} for label in MAG_LEVEL_TO_LABEL.values()}
    if not path.exists():
        return vectors

    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            fos_id = (row.get("fos_id") or "").strip()
            if not fos_id:
                continue
            mag_level = fos_levels.get(fos_id)
            label = MAG_LEVEL_TO_LABEL.get(mag_level)
            if not label:
                continue
            try:
                weight = float(row.get("direct_weight_sum", "0") or 0)
            except ValueError:
                continue
            if weight <= 0:
                continue
            vectors[label][fos_id] = vectors[label].get(fos_id, 0.0) + weight
    return vectors


def cosine_and_drift(a: Dict[str, float], b: Dict[str, float]) -> Tuple[float, float, float, float, int]:
    norm_a = math.sqrt(sum(v * v for v in a.values()))
    norm_b = math.sqrt(sum(v * v for v in b.values()))
    shared = set(a).intersection(b)

    if norm_a == 0 and norm_b == 0:
        return 1.0, 0.0, norm_a, norm_b, 0
    if norm_a == 0 or norm_b == 0:
        return 0.0, 1.0, norm_a, norm_b, len(shared)

    dot = sum(a[k] * b[k] for k in shared)
    cosine = dot / (norm_a * norm_b)
    cosine = max(0.0, min(1.0, cosine))
    return cosine, 1.0 - cosine, norm_a, norm_b, len(shared)


def format_float(v: float) -> str:
    return f"{v:.8f}"


def write_summary(path: Path, values: Dict[Tuple[str, str, str], List[float]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write(
            "from_bin\tto_bin\tlevel\texperts\tmean_drift\tmedian_drift\t"
            "std_drift\tmin_drift\tmax_drift\n"
        )
        for from_bin, to_bin in BIN_PAIRS:
            for level in MAG_LEVEL_TO_LABEL.values():
                drift_values = values.get((from_bin, to_bin, level), [])
                if not drift_values:
                    f.write(f"{from_bin}\t{to_bin}\t{level}\t0\t\t\t\t\t\n")
                    continue
                f.write(
                    f"{from_bin}\t{to_bin}\t{level}\t{len(drift_values)}\t"
                    f"{format_float(mean(drift_values))}\t"
                    f"{format_float(median(drift_values))}\t"
                    f"{format_float(pstdev(drift_values))}\t"
                    f"{format_float(min(drift_values))}\t"
                    f"{format_float(max(drift_values))}\n"
                )


def main() -> None:
    args = parse_args()
    profile_dir = Path(args.profile_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fos_levels = load_fos_levels(Path(args.fos_map))
    expert_ids = list_expert_ids(profile_dir)

    detail_path = out_dir / "expert_level_drift.tsv"
    summary_values: Dict[Tuple[str, str, str], List[float]] = defaultdict(list)

    with detail_path.open("w", encoding="utf-8") as f:
        f.write(
            "expert_id\tfrom_bin\tto_bin\tlevel\tmag_level\tcosine_similarity\t"
            "drift\tfrom_topics\tto_topics\tshared_topics\tfrom_norm\tto_norm\n"
        )

        for idx, expert_id in enumerate(expert_ids, start=1):
            vectors_by_bin = {}
            for bin_name in BINS:
                path = profile_dir / bin_name / f"{expert_id}_direct_fos_nodes.tsv"
                vectors_by_bin[bin_name] = load_profile_vectors(path, fos_levels)

            for from_bin, to_bin in BIN_PAIRS:
                for mag_level, level in MAG_LEVEL_TO_LABEL.items():
                    from_vec = vectors_by_bin[from_bin][level]
                    to_vec = vectors_by_bin[to_bin][level]
                    cosine, drift, norm_from, norm_to, shared_count = cosine_and_drift(
                        from_vec, to_vec
                    )
                    summary_values[(from_bin, to_bin, level)].append(drift)
                    f.write(
                        f"{expert_id}\t{from_bin}\t{to_bin}\t{level}\t{mag_level}\t"
                        f"{format_float(cosine)}\t{format_float(drift)}\t"
                        f"{len(from_vec)}\t{len(to_vec)}\t{shared_count}\t"
                        f"{format_float(norm_from)}\t{format_float(norm_to)}\n"
                    )

            if idx % 1000 == 0:
                print(f"progress experts={idx}/{len(expert_ids)}")

    summary_path = out_dir / "summary_by_level.tsv"
    write_summary(summary_path, summary_values)

    print(f"experts={len(expert_ids)}")
    print(f"detail={detail_path}")
    print(f"summary={summary_path}")


if __name__ == "__main__":
    main()
