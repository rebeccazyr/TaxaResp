#!/usr/bin/env python3
"""Add per-skill FoS weights to a teams CSV from dblp.v12.json.

The input CSV is expected to contain:
- paper_id
- skills: pipe-separated normalized FoS names, e.g. machine_learning|data_mining

The output keeps all original columns and appends:
- skill_weights: pipe-separated weights aligned with `skills`
- skills_with_weights: pipe-separated `skill:weight` entries
- missing_weight_skills: skills not found in the paper's DBLP `fos` list
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Set


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add DBLP FoS weights for each skill in a teams CSV"
    )
    parser.add_argument(
        "--teams-csv",
        default="data_preprocess/teams_2020plus.csv",
        help="Input teams CSV containing paper_id and pipe-separated skills",
    )
    parser.add_argument(
        "--dblp-json",
        default="data/dblp/dblp.v12.json",
        help="Path to dblp.v12.json",
    )
    parser.add_argument(
        "--output",
        default="data_preprocess/teams_2020plus_with_skill_weights.csv",
        help="Output CSV path",
    )
    return parser.parse_args()


def normalize_skill(name: object) -> str:
    return str(name or "").strip().lower().replace(" ", "_")


def safe_float(v: object) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def iter_json_objects(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            s = raw.strip()
            if not s:
                continue
            if s[0] == ",":
                s = s[1:]
            if not s.startswith("{"):
                continue
            try:
                yield json.loads(s)
            except json.JSONDecodeError:
                continue


def load_rows(path: Path) -> tuple[List[dict], List[str], Set[str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])

    paper_ids = {str(row.get("paper_id", "")).strip() for row in rows}
    paper_ids.discard("")
    return rows, fieldnames, paper_ids


def load_paper_skill_weights(dblp_path: Path, target_paper_ids: Set[str]) -> Dict[str, Dict[str, float]]:
    paper_skill_weights: Dict[str, Dict[str, float]] = {}
    remaining = set(target_paper_ids)

    for obj in iter_json_objects(dblp_path):
        paper_id = str(obj.get("id", "")).strip()
        if paper_id not in remaining:
            continue

        weights: Dict[str, float] = {}
        fos_items = obj.get("fos") or []
        if isinstance(fos_items, list):
            for item in fos_items:
                if not isinstance(item, dict):
                    continue
                skill = normalize_skill(item.get("name", ""))
                if not skill:
                    continue
                weight = safe_float(item.get("w", 0.0))
                # If a normalized FoS repeats, keep the strongest assignment.
                if skill not in weights or weight > weights[skill]:
                    weights[skill] = weight

        paper_skill_weights[paper_id] = weights
        remaining.remove(paper_id)
        if not remaining:
            break

    return paper_skill_weights


def main() -> None:
    args = parse_args()
    teams_path = Path(args.teams_csv)
    output_path = Path(args.output)

    rows, fieldnames, paper_ids = load_rows(teams_path)
    paper_skill_weights = load_paper_skill_weights(Path(args.dblp_json), paper_ids)

    added_fields = ["skill_weights", "skills_with_weights", "missing_weight_skills"]
    out_fields = fieldnames + [f for f in added_fields if f not in fieldnames]

    missing_papers = 0
    missing_skill_count = 0
    matched_skill_count = 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=out_fields)
        writer.writeheader()

        for row in rows:
            paper_id = str(row.get("paper_id", "")).strip()
            skills = [s.strip() for s in (row.get("skills") or "").split("|") if s.strip()]
            weights_for_paper = paper_skill_weights.get(paper_id)
            if weights_for_paper is None:
                missing_papers += 1
                weights_for_paper = {}

            aligned_weights: List[str] = []
            skills_with_weights: List[str] = []
            missing_skills: List[str] = []

            for skill in skills:
                weight = weights_for_paper.get(skill)
                if weight is None:
                    aligned_weights.append("")
                    skills_with_weights.append(f"{skill}:")
                    missing_skills.append(skill)
                    missing_skill_count += 1
                    continue
                weight_str = f"{weight:.5f}"
                aligned_weights.append(weight_str)
                skills_with_weights.append(f"{skill}:{weight_str}")
                matched_skill_count += 1

            row["skill_weights"] = "|".join(aligned_weights)
            row["skills_with_weights"] = "|".join(skills_with_weights)
            row["missing_weight_skills"] = "|".join(missing_skills)
            writer.writerow(row)

    print(f"rows={len(rows)}")
    print(f"target_papers={len(paper_ids)}")
    print(f"found_papers={len(paper_skill_weights)}")
    print(f"missing_papers={missing_papers}")
    print(f"matched_skills={matched_skill_count}")
    print(f"missing_skills={missing_skill_count}")
    print(f"output={output_path}")


if __name__ == "__main__":
    main()
