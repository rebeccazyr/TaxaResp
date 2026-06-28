#!/usr/bin/env python3
"""Analyze ground-truth test authors' expert-citation block dispersion."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Dict, Iterable, List, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", str(Path("cache/matplotlib").resolve()))
sys.path.insert(0, str(Path("cache/pydeps").resolve()))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task-nodes-jsonl", default="output/hierec_embedding_server_inputs/task_nodes.jsonl")
    p.add_argument("--out-dir", default="output/test_author_citation_block_dispersion_r2")
    p.add_argument("--resolution", type=float, default=2.0)
    p.add_argument(
        "--membership-tsv",
        default="",
        help=(
            "Existing expert_id/block_id cache. If omitted, the script first "
            "checks common project cache paths, then builds the graph if needed."
        ),
    )
    p.add_argument("--profile-dir", default="output/expert_profile_year_bins/all_2000_2019")
    p.add_argument("--expert-papers-tsv", default="output/all_expert_paper_embeddings/expert_papers.tsv")
    p.add_argument("--dblp-json", default="data/dblp/dblp.v12.json")
    return p.parse_args()


def resolution_tag(resolution: float) -> str:
    return f"{resolution:g}".replace(".", "p")


def cache_candidates(out_dir: Path, resolution: float, explicit: str) -> List[Path]:
    if explicit:
        return [Path(explicit)]
    tag = resolution_tag(resolution)
    return [
        out_dir / f"expert_graph_louvain_resolution{tag}_membership.tsv",
        Path("output/expert_citation_louvain_blocks") / f"expert_graph_louvain_resolution{tag}_membership.tsv",
        Path("output/selected_method_results") / f"expert_graph_louvain_resolution{tag}_membership.tsv",
    ]


def sort_key(value: str) -> Tuple[int, object]:
    try:
        return (0, int(value))
    except ValueError:
        return (1, value)


def dedupe(values: Iterable[str]) -> List[str]:
    seen = set()
    out = []
    for value in values:
        value = str(value)
        if value and value not in seen:
            seen.add(value)
            out.append(value)
    return out


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def percentile(values: Sequence[float], pct: float) -> float:
    if not values:
        return 0.0
    xs = sorted(values)
    pos = (len(xs) - 1) * pct
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    frac = pos - lo
    return xs[lo] * (1.0 - frac) + xs[hi] * frac


def read_membership(path: Path) -> Dict[str, str]:
    membership: Dict[str, str] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if "expert_id" not in (reader.fieldnames or []) or "block_id" not in (reader.fieldnames or []):
            raise SystemExit(f"membership cache must contain expert_id and block_id columns: {path}")
        for row in reader:
            membership[str(row["expert_id"])] = str(row["block_id"])
    return membership


def write_membership(path: Path, membership: Dict[str, str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["expert_id", "block_id"], delimiter="\t")
        writer.writeheader()
        for expert_id in sorted(membership, key=sort_key):
            writer.writerow({"expert_id": expert_id, "block_id": membership[expert_id]})


def load_or_build_membership(args: argparse.Namespace, out_dir: Path) -> Tuple[Dict[str, str], Path, str]:
    for candidate in cache_candidates(out_dir, args.resolution, args.membership_tsv):
        if candidate.exists():
            print(f"using_membership_cache={candidate}", flush=True)
            return read_membership(candidate), candidate, "cache"

    print("membership cache not found; building expert citation graph", flush=True)
    sys.path.insert(0, str(Path("scripts").resolve()))
    from evaluate_expert_citation_louvain_blocks import (  # noqa: WPS433
        build_expert_citation_graph,
        load_paper_experts,
        load_profile_experts,
    )

    profile_experts = load_profile_experts(Path(args.profile_dir))
    paper_experts = load_paper_experts(Path(args.expert_papers_tsv), profile_experts)
    graph = build_expert_citation_graph(Path(args.dblp_json), paper_experts, len(profile_experts))
    clustering = graph.community_multilevel(weights="weight", resolution=float(args.resolution))
    membership = {
        expert_id: str(int(clustering.membership[idx]))
        for idx, expert_id in enumerate(profile_experts)
    }
    path = out_dir / f"expert_graph_louvain_resolution{resolution_tag(args.resolution)}_membership.tsv"
    write_membership(path, membership)
    return membership, path, "built"


def read_test_tasks(path: Path) -> List[dict]:
    tasks: List[dict] = []
    seen = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            paper_id = str(row["paper_id"])
            if paper_id in seen:
                continue
            seen.add(paper_id)
            members = dedupe(row.get("members") or [])
            tasks.append(
                {
                    "paper_id": paper_id,
                    "team_size": int(row.get("team_size") or len(members) or 0),
                    "members": members,
                }
            )
    return tasks


def write_block_members(path: Path, membership: Dict[str, str]) -> List[dict]:
    blocks: Dict[str, List[str]] = defaultdict(list)
    for expert_id, block_id in membership.items():
        blocks[block_id].append(expert_id)

    rows = []
    with path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = ["block_id", "expert_count", "expert_ids"]
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for block_id in sorted(blocks, key=sort_key):
            experts = sorted(blocks[block_id], key=sort_key)
            row = {
                "block_id": block_id,
                "expert_count": len(experts),
                "expert_ids": "|".join(experts),
            }
            writer.writerow(row)
            rows.append(row)
    return rows


def task_dispersion_rows(tasks: Sequence[dict], membership: Dict[str, str]) -> List[dict]:
    rows = []
    for task in tasks:
        by_block: Dict[str, List[str]] = defaultdict(list)
        missing = []
        for expert_id in task["members"]:
            block_id = membership.get(expert_id)
            if block_id is None:
                missing.append(expert_id)
            else:
                by_block[block_id].append(expert_id)

        known_count = sum(len(experts) for experts in by_block.values())
        block_ids = sorted(by_block, key=sort_key)
        authors_by_block = ";".join(
            f"{block_id}:{','.join(sorted(by_block[block_id], key=sort_key))}"
            for block_id in block_ids
        )
        distinct_blocks = len(block_ids)
        rows.append(
            {
                "paper_id": task["paper_id"],
                "team_size": task["team_size"],
                "author_count": len(task["members"]),
                "known_author_count": known_count,
                "missing_author_count": len(missing),
                "distinct_block_count": distinct_blocks,
                "block_dispersion_ratio": f"{(distinct_blocks / known_count) if known_count else 0.0:.12f}",
                "excess_blocks": max(0, distinct_blocks - 1),
                "block_ids": "|".join(block_ids),
                "authors_by_block": authors_by_block,
                "members": "|".join(task["members"]),
                "missing_members": "|".join(missing),
            }
        )
    return rows


def write_tsv(path: Path, rows: Sequence[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def distribution_rows(task_rows: Sequence[dict]) -> List[dict]:
    counts = Counter(int(row["distinct_block_count"]) for row in task_rows)
    total = len(task_rows)
    cumulative = 0
    rows = []
    for distinct_blocks in sorted(counts):
        tasks = counts[distinct_blocks]
        cumulative += tasks
        rows.append(
            {
                "distinct_block_count": distinct_blocks,
                "tasks": tasks,
                "percent": f"{(100.0 * tasks / total) if total else 0.0:.6f}",
                "cumulative_tasks": cumulative,
                "cumulative_percent": f"{(100.0 * cumulative / total) if total else 0.0:.6f}",
            }
        )
    return rows


def summary_rows(
    task_rows: Sequence[dict],
    block_rows: Sequence[dict],
    membership_source: Path,
    source_kind: str,
    resolution: float,
) -> List[dict]:
    distinct_counts = [int(row["distinct_block_count"]) for row in task_rows]
    ratios = [float(row["block_dispersion_ratio"]) for row in task_rows]
    missing_authors = sum(int(row["missing_author_count"]) for row in task_rows)
    known_authors = sum(int(row["known_author_count"]) for row in task_rows)
    total_authors = sum(int(row["author_count"]) for row in task_rows)
    return [
        {
            "resolution": f"{resolution:.6f}",
            "tasks": len(task_rows),
            "total_authors": total_authors,
            "known_authors": known_authors,
            "missing_authors": missing_authors,
            "expert_blocks": len(block_rows),
            "membership_experts": sum(int(row["expert_count"]) for row in block_rows),
            "membership_source": str(membership_source),
            "membership_source_kind": source_kind,
            "min_distinct_blocks": min(distinct_counts) if distinct_counts else 0,
            "max_distinct_blocks": max(distinct_counts) if distinct_counts else 0,
            "mean_distinct_blocks": f"{mean(distinct_counts):.12f}",
            "median_distinct_blocks": f"{median(distinct_counts):.6f}" if distinct_counts else "0.000000",
            "p25_distinct_blocks": f"{percentile(distinct_counts, 0.25):.6f}",
            "p75_distinct_blocks": f"{percentile(distinct_counts, 0.75):.6f}",
            "mean_block_dispersion_ratio": f"{mean(ratios):.12f}",
            "single_block_tasks": sum(1 for value in distinct_counts if value == 1),
            "multi_block_tasks": sum(1 for value in distinct_counts if value > 1),
        }
    ]


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    membership, membership_source, source_kind = load_or_build_membership(args, out_dir)
    normalized_membership_path = (
        out_dir / f"expert_graph_louvain_resolution{resolution_tag(args.resolution)}_membership.tsv"
    )
    write_membership(normalized_membership_path, membership)

    block_rows = write_block_members(
        out_dir / f"expert_graph_louvain_resolution{resolution_tag(args.resolution)}_block_members.tsv",
        membership,
    )
    tasks = read_test_tasks(Path(args.task_nodes_jsonl))
    detail_rows = task_dispersion_rows(tasks, membership)
    dist_rows = distribution_rows(detail_rows)
    summary = summary_rows(
        detail_rows,
        block_rows,
        membership_source,
        source_kind,
        float(args.resolution),
    )

    write_tsv(out_dir / "test_author_citation_block_dispersion_by_task.tsv", detail_rows)
    write_tsv(out_dir / "test_author_citation_block_dispersion_distribution.tsv", dist_rows)
    write_tsv(out_dir / "test_author_citation_block_dispersion_summary.tsv", summary)

    print(f"summary={out_dir / 'test_author_citation_block_dispersion_summary.tsv'}")
    print(f"distribution={out_dir / 'test_author_citation_block_dispersion_distribution.tsv'}")
    print(f"task_detail={out_dir / 'test_author_citation_block_dispersion_by_task.tsv'}")
    print(f"block_members={out_dir / f'expert_graph_louvain_resolution{resolution_tag(args.resolution)}_block_members.tsv'}")


if __name__ == "__main__":
    main()
