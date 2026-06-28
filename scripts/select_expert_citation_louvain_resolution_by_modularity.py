#!/usr/bin/env python3
"""Select an expert-citation Louvain resolution by modularity."""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import Counter
from pathlib import Path
from statistics import median
from typing import Dict, Iterable, List, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", str(Path("cache/matplotlib").resolve()))
sys.path.insert(0, str(Path("cache/pydeps").resolve()))
sys.path.insert(0, str(Path("scripts").resolve()))

import igraph as ig  # noqa: E402

from evaluate_expert_citation_louvain_blocks import (  # noqa: E402
    build_expert_citation_graph,
    load_paper_experts,
    load_profile_experts,
    parse_resolutions,
)


DEFAULT_RESOLUTIONS = (
    "0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,"
    "1,1.25,1.5,1.75,2,2.5,3,4,5,7.5,10,15,20,30,50,75,100"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--profile-dir", default="output/expert_profile_year_bins/all_2000_2019")
    p.add_argument("--expert-papers-tsv", default="output/all_expert_paper_embeddings/expert_papers.tsv")
    p.add_argument("--dblp-json", default="data/dblp/dblp.v12.json")
    p.add_argument("--out-dir", default="output/expert_citation_louvain_modularity_selection")
    p.add_argument("--resolutions", default=DEFAULT_RESOLUTIONS)
    p.add_argument(
        "--selection-metric",
        choices=("standard_modularity", "generalized_modularity"),
        default="standard_modularity",
        help=(
            "standard_modularity compares each partition with gamma=1. "
            "generalized_modularity compares the objective value at the same "
            "gamma used to detect that partition."
        ),
    )
    return p.parse_args()


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def sort_key(value: str) -> Tuple[int, object]:
    try:
        return (0, int(value))
    except ValueError:
        return (1, value)


def resolution_tag(resolution: float) -> str:
    return f"{resolution:g}".replace(".", "p")


def write_tsv(path: Path, rows: Sequence[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def write_membership(path: Path, expert_ids: Sequence[str], membership: Sequence[int]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["expert_id", "block_id"], delimiter="\t")
        writer.writeheader()
        for expert_id, block_id in sorted(
            zip(expert_ids, membership),
            key=lambda item: sort_key(item[0]),
        ):
            writer.writerow({"expert_id": expert_id, "block_id": int(block_id)})


def write_block_members(path: Path, expert_ids: Sequence[str], membership: Sequence[int]) -> None:
    blocks: Dict[str, List[str]] = {}
    for expert_id, block_id in zip(expert_ids, membership):
        blocks.setdefault(str(int(block_id)), []).append(expert_id)

    rows = []
    for block_id in sorted(blocks, key=sort_key):
        members = sorted(blocks[block_id], key=sort_key)
        rows.append(
            {
                "block_id": block_id,
                "expert_count": len(members),
                "expert_ids": "|".join(members),
            }
        )
    write_tsv(path, rows)


def summarize_partition(
    graph: ig.Graph,
    membership: Sequence[int],
    resolution: float,
) -> dict:
    sizes = list(Counter(int(block_id) for block_id in membership).values())
    standard_modularity = graph.modularity(membership, weights="weight", resolution=1.0)
    generalized_modularity = graph.modularity(
        membership,
        weights="weight",
        resolution=resolution,
    )
    return {
        "resolution": f"{resolution:.6f}",
        "standard_modularity_gamma1": f"{standard_modularity:.12f}",
        "generalized_modularity_at_resolution": f"{generalized_modularity:.12f}",
        "communities": len(sizes),
        "largest_community_experts": max(sizes) if sizes else 0,
        "largest_community_percent": f"{(100.0 * max(sizes) / graph.vcount()) if sizes else 0.0:.6f}",
        "avg_experts_per_community": f"{mean(sizes):.6f}",
        "median_experts_per_community": f"{median(sizes):.6f}" if sizes else "0.000000",
        "singleton_communities": sum(1 for size in sizes if size == 1),
    }


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    profile_experts = load_profile_experts(Path(args.profile_dir))
    paper_experts = load_paper_experts(Path(args.expert_papers_tsv), profile_experts)
    print(
        f"experts={len(profile_experts):,} papers_with_experts={len(paper_experts):,}",
        flush=True,
    )
    graph = build_expert_citation_graph(
        Path(args.dblp_json),
        paper_experts,
        len(profile_experts),
    )

    rows = []
    memberships: Dict[str, List[int]] = {}
    for resolution in parse_resolutions(args.resolutions):
        resolution_text = f"{resolution:.6f}"
        print(f"louvain_resolution={resolution_text}", flush=True)
        clustering = graph.community_multilevel(weights="weight", resolution=resolution)
        membership = [int(block_id) for block_id in clustering.membership]
        memberships[resolution_text] = membership
        row = summarize_partition(graph, membership, resolution)
        rows.append(row)
        print(row, flush=True)

    metric_column = (
        "standard_modularity_gamma1"
        if args.selection_metric == "standard_modularity"
        else "generalized_modularity_at_resolution"
    )
    best = max(rows, key=lambda row: float(row[metric_column]))
    best_resolution = float(best["resolution"])
    best_tag = resolution_tag(best_resolution)
    best_membership = memberships[best["resolution"]]

    write_tsv(out_dir / "expert_graph_louvain_modularity_by_resolution.tsv", rows)
    write_tsv(out_dir / "best_resolution.tsv", [{**best, "selection_metric": args.selection_metric}])
    write_membership(
        out_dir / f"expert_graph_louvain_best_resolution{best_tag}_membership.tsv",
        profile_experts,
        best_membership,
    )
    write_block_members(
        out_dir / f"expert_graph_louvain_best_resolution{best_tag}_block_members.tsv",
        profile_experts,
        best_membership,
    )

    print(f"best_resolution={best['resolution']} metric={args.selection_metric} value={best[metric_column]}")
    print(f"modularity_table={out_dir / 'expert_graph_louvain_modularity_by_resolution.tsv'}")
    print(f"best_summary={out_dir / 'best_resolution.tsv'}")


if __name__ == "__main__":
    main()
