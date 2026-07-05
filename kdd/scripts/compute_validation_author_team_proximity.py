#!/usr/bin/env python3
"""Compute degree-normalized author-to-validation-team citation proximity."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from citation_proximity import load_undirected_adjacency, team_proximity, weighted_degrees


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--graph-dir",
        default="outputs/expert_citation_graph_valid2018_pre2018",
        help="Directory containing edges_undirected.tsv.",
    )
    parser.add_argument(
        "--validation-jsonl",
        default="outputs/temporal_task_splits_full/validation_2018_all_authors_hist_ge5.jsonl",
        help="Validation JSONL containing target teams.",
    )
    parser.add_argument(
        "--candidates-tsv",
        required=True,
        help="TSV with paper_id/id and author_id/expert_id columns to score.",
    )
    parser.add_argument(
        "--out-tsv",
        required=True,
        help="Output TSV for proximity scores.",
    )
    parser.add_argument(
        "--normalization",
        choices=("degree", "sqrt_degree", "none"),
        default="degree",
        help="How to normalize raw proximity by the candidate author's weighted degree.",
    )
    parser.add_argument(
        "--include-self",
        action="store_true",
        help="Include candidate-to-self links if present. Self-loops are usually absent.",
    )
    return parser.parse_args()


def iter_author_ids(authors: object) -> List[str]:
    if not isinstance(authors, list):
        return []
    out: List[str] = []
    seen = set()
    for author in authors:
        if not isinstance(author, dict):
            continue
        author_id = str(author.get("id", "")).strip()
        if author_id and author_id not in seen:
            seen.add(author_id)
            out.append(author_id)
    return out


def load_validation_teams(path: Path) -> Dict[str, List[str]]:
    teams: Dict[str, List[str]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            paper_id = str(obj.get("paper_id") or obj.get("id") or "").strip()
            if not paper_id:
                continue
            teams[paper_id] = iter_author_ids(obj.get("authors"))
    return teams


def row_value(row: dict, names: Iterable[str]) -> str:
    for name in names:
        value = str(row.get(name, "")).strip()
        if value:
            return value
    return ""


def main() -> None:
    args = parse_args()
    graph_dir = Path(args.graph_dir)
    adjacency = load_undirected_adjacency(graph_dir / "edges_undirected.tsv")
    degrees = weighted_degrees(adjacency)
    teams = load_validation_teams(Path(args.validation_jsonl))

    out_path = Path(args.out_tsv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with Path(args.candidates_tsv).open("r", encoding="utf-8", newline="") as in_f, out_path.open(
        "w", encoding="utf-8", newline=""
    ) as out_f:
        reader = csv.DictReader(in_f, delimiter="\t")
        fieldnames = list(reader.fieldnames or []) + [
            "target_team_size",
            "raw_citation_proximity",
            "candidate_weighted_degree",
            "degree_normalization",
            "citation_proximity",
        ]
        writer = csv.DictWriter(out_f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()

        rows = 0
        scored = 0
        for row in reader:
            rows += 1
            paper_id = row_value(row, ("paper_id", "id"))
            author_id = row_value(row, ("author_id", "expert_id", "candidate_author_id"))
            team = teams.get(paper_id, [])
            raw_value, degree, score = team_proximity(
                author_id,
                team,
                adjacency,
                degrees,
                normalization=args.normalization,
                exclude_self=not args.include_self,
            )
            if team and author_id:
                scored += 1
            out_row = dict(row)
            out_row.update(
                {
                    "target_team_size": len(team),
                    "raw_citation_proximity": f"{raw_value:.12g}",
                    "candidate_weighted_degree": f"{degree:.12g}",
                    "degree_normalization": args.normalization,
                    "citation_proximity": f"{score:.12g}",
                }
            )
            writer.writerow(out_row)

    print(f"rows={rows:,}")
    print(f"scored_rows={scored:,}")
    print(f"out_tsv={out_path}")


if __name__ == "__main__":
    main()
