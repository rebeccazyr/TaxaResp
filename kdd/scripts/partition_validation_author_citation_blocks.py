#!/usr/bin/env python3
"""Partition a validation-author citation graph into Louvain blocks.

The resolution grid is scored by how well gold coauthor pairs from the
validation split fall in the same block, penalized by the random same-block
probability implied by block sizes:

    separation_score = gold_pair_same_block_rate - random_pair_same_block_rate

This favors blocks that keep true validation teams cohesive without making
communities so large that random authors frequently share a block.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", str(Path("cache/matplotlib").resolve()))
os.environ.setdefault("XDG_CACHE_HOME", str(Path("cache").resolve()))
sys.path.insert(0, str(Path("../cache/pydeps").resolve()))

import igraph as ig  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run weighted Louvain resolution grid on validation author citation graph."
    )
    parser.add_argument(
        "--graph-dir",
        default="outputs/expert_citation_graph_valid2018_pre2018",
        help="Directory containing nodes.tsv and edges_undirected.tsv.",
    )
    parser.add_argument(
        "--validation-jsonl",
        default="outputs/temporal_task_splits_full/validation_2018.jsonl",
        help="Validation JSONL used to evaluate gold coauthor block cohesion.",
    )
    parser.add_argument(
        "--out-dir",
        default="outputs/validation_author_citation_louvain_blocks",
        help="Output directory for memberships, summaries, and best-resolution files.",
    )
    parser.add_argument(
        "--resolutions",
        default="1,2,5,10,20,50,100",
        help="Comma-separated Louvain resolution grid.",
    )
    parser.add_argument(
        "--best-metric",
        choices=["separation_score", "standard_modularity_gamma1"],
        default="separation_score",
        help="Metric used to select best_resolution.tsv and best_membership.tsv.",
    )
    return parser.parse_args()


def parse_resolutions(text: str) -> List[float]:
    values = [float(item.strip()) for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError("resolution grid must not be empty")
    return values


def resolution_label(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def read_node_ids(path: Path) -> Tuple[List[str], Dict[str, int], Dict[str, dict]]:
    rows: List[Tuple[int, str, dict]] = []
    node_rows: Dict[str, dict] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            author_id = row["author_id"]
            rows.append((int(row["author_idx"]), author_id, row))
            node_rows[author_id] = row
    rows.sort(key=lambda item: item[0])
    author_ids = [author_id for _, author_id, _ in rows]
    author_to_idx = {author_id: idx for idx, author_id in enumerate(author_ids)}
    return author_ids, author_to_idx, node_rows


def read_validation_tasks(path: Path, author_to_idx: Dict[str, int]) -> List[List[int]]:
    tasks: List[List[int]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            authors = obj.get("authors") or []
            if not isinstance(authors, list):
                tasks.append([])
                continue
            seen = set()
            task_authors: List[int] = []
            for author in authors:
                if not isinstance(author, dict):
                    continue
                author_id = str(author.get("id", "")).strip()
                if not author_id or author_id in seen or author_id not in author_to_idx:
                    continue
                seen.add(author_id)
                task_authors.append(author_to_idx[author_id])
            tasks.append(task_authors)
    return tasks


def choose2(value: int) -> int:
    return value * (value - 1) // 2


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def author_sort_key(author_id: str) -> Tuple[int, object]:
    if author_id.isdigit():
        return (0, int(author_id))
    return (1, author_id)


def ensure_numeric_ncol(sqlite_path: Path, out_path: Path) -> None:
    if (
        out_path.exists()
        and out_path.stat().st_size > 0
        and out_path.stat().st_mtime >= sqlite_path.stat().st_mtime
    ):
        print(f"using_cached_numeric_edges={out_path}", flush=True)
        return

    print(f"writing_numeric_edges={out_path}", flush=True)
    conn = sqlite3.connect(sqlite_path)
    query = (
        "SELECT "
        "CASE WHEN src < dst THEN src ELSE dst END AS a, "
        "CASE WHEN src < dst THEN dst ELSE src END AS b, "
        "SUM(weight) AS weight "
        "FROM directed_edges "
        "GROUP BY a, b "
        "ORDER BY a, b"
    )
    rows = 0
    with out_path.open("w", encoding="utf-8", newline="") as f:
        for a, b, weight in conn.execute(query):
            f.write(f"{a} {b} {weight}\n")
            rows += 1
            if rows % 5000000 == 0:
                print(f"numeric_edge_write_progress rows={rows:,}", flush=True)
    conn.close()
    print(f"numeric_edges_written={rows:,}", flush=True)


def load_graph(numeric_edges_path: Path, author_count: int) -> ig.Graph:
    print(f"loading_graph_edges={numeric_edges_path}", flush=True)
    graph = ig.Graph.Read_Ncol(str(numeric_edges_path), names=False, weights=True, directed=False)
    if graph.vcount() < author_count:
        graph.add_vertices(author_count - graph.vcount())
    if graph.vcount() != author_count:
        raise ValueError(
            f"graph vertex count mismatch: graph={graph.vcount()} expected={author_count}"
        )
    print(
        f"graph_loaded nodes={graph.vcount():,} edges={graph.ecount():,} "
        f"isolates={sum(1 for degree in graph.degree() if degree == 0):,}",
        flush=True,
    )
    return graph


def summarize_blocks(graph: ig.Graph, membership: Sequence[int], resolution: float) -> dict:
    block_sizes = Counter(int(block_id) for block_id in membership)
    sizes = list(block_sizes.values())
    degrees = graph.degree()
    random_pair_same_block = (
        sum(choose2(size) for size in sizes) / choose2(graph.vcount())
        if graph.vcount() >= 2
        else 0.0
    )
    return {
        "resolution": f"{resolution:.6f}",
        "author_nodes": graph.vcount(),
        "undirected_edges": graph.ecount(),
        "isolated_authors": sum(1 for degree in degrees if degree == 0),
        "blocks": len(sizes),
        "largest_block_authors": max(sizes) if sizes else 0,
        "avg_authors_per_block": f"{mean([float(size) for size in sizes]):.6f}",
        "median_authors_per_block": f"{median(sizes):.6f}" if sizes else "0.000000",
        "singleton_blocks": sum(1 for size in sizes if size == 1),
        "random_pair_same_block_rate": f"{random_pair_same_block:.12f}",
    }


def evaluate_validation_teams(
    tasks: Sequence[Sequence[int]],
    membership: Sequence[int],
) -> dict:
    total_pairs = 0
    same_block_pairs = 0
    tasks_with_two_plus = 0
    all_same_block_tasks = 0
    distinct_block_counts: List[int] = []
    largest_block_shares: List[float] = []

    for idxs in tasks:
        if len(idxs) < 2:
            continue
        tasks_with_two_plus += 1
        block_counts = Counter(int(membership[idx]) for idx in idxs)
        distinct_blocks = len(block_counts)
        distinct_block_counts.append(distinct_blocks)
        largest_block_shares.append(max(block_counts.values()) / len(idxs))
        if distinct_blocks == 1:
            all_same_block_tasks += 1
        total_pairs += choose2(len(idxs))
        same_block_pairs += sum(choose2(count) for count in block_counts.values())

    pair_rate = same_block_pairs / total_pairs if total_pairs else 0.0
    return {
        "validation_tasks": len(tasks),
        "validation_tasks_with_2plus_authors": tasks_with_two_plus,
        "gold_author_pairs": total_pairs,
        "gold_same_block_pairs": same_block_pairs,
        "gold_pair_same_block_rate": f"{pair_rate:.12f}",
        "all_authors_same_block_tasks": all_same_block_tasks,
        "all_authors_same_block_task_rate": (
            f"{all_same_block_tasks / tasks_with_two_plus:.12f}" if tasks_with_two_plus else "0.000000000000"
        ),
        "mean_distinct_blocks_per_team": f"{mean([float(x) for x in distinct_block_counts]):.6f}",
        "median_distinct_blocks_per_team": (
            f"{median(distinct_block_counts):.6f}" if distinct_block_counts else "0.000000"
        ),
        "mean_largest_block_share_per_team": f"{mean(largest_block_shares):.12f}",
    }


def write_dict_rows(path: Path, rows: List[dict]) -> None:
    if not rows:
        return
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def write_membership(
    path: Path,
    author_ids: Sequence[str],
    membership: Sequence[int],
    node_rows: Dict[str, dict],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(
            [
                "author_id",
                "block_id",
                "display_name",
                "validation_papers",
                "historical_papers_pre_cutoff",
                "out_citation_weight",
                "in_citation_weight",
            ]
        )
        for idx, author_id in enumerate(author_ids):
            row = node_rows.get(author_id, {})
            writer.writerow(
                [
                    author_id,
                    int(membership[idx]),
                    row.get("display_name", ""),
                    row.get("validation_papers", "0"),
                    row.get("historical_papers_pre_cutoff", "0"),
                    row.get("out_citation_weight", "0"),
                    row.get("in_citation_weight", "0"),
                ]
            )


def main() -> None:
    args = parse_args()
    graph_dir = Path(args.graph_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    nodes_path = graph_dir / "nodes.tsv"
    sqlite_path = graph_dir / "graph.sqlite"
    if not nodes_path.exists():
        raise FileNotFoundError(f"nodes.tsv not found: {nodes_path}")
    if not sqlite_path.exists():
        raise FileNotFoundError(f"graph.sqlite not found: {sqlite_path}")

    author_ids, author_to_idx, node_rows = read_node_ids(nodes_path)
    tasks = read_validation_tasks(Path(args.validation_jsonl), author_to_idx)
    numeric_edges_path = out_dir / "edges_undirected_numeric.ncol"
    ensure_numeric_ncol(sqlite_path, numeric_edges_path)
    graph = load_graph(numeric_edges_path, len(author_ids))

    rows: List[dict] = []
    best_row: Optional[dict] = None
    best_membership_path: Optional[Path] = None
    best_score = float("-inf")

    for resolution in parse_resolutions(args.resolutions):
        label = resolution_label(resolution)
        print(f"louvain_resolution={resolution:.6f}", flush=True)
        clustering = graph.community_multilevel(weights="weight", resolution=resolution)
        membership = [int(block_id) for block_id in clustering.membership]
        block_row = summarize_blocks(graph, membership, resolution)
        team_row = evaluate_validation_teams(tasks, membership)

        gold_rate = float(team_row["gold_pair_same_block_rate"])
        random_rate = float(block_row["random_pair_same_block_rate"])
        separation_score = gold_rate - random_rate
        lift = gold_rate / random_rate if random_rate > 0 else 0.0
        louvain_modularity = graph.modularity(
            membership, weights="weight", resolution=resolution, directed=False
        )
        standard_modularity_gamma1 = graph.modularity(
            membership, weights="weight", resolution=1.0, directed=False
        )

        row = {
            **block_row,
            **team_row,
            "separation_score": f"{separation_score:.12f}",
            "gold_vs_random_lift": f"{lift:.12f}",
            "louvain_modularity_at_resolution": f"{louvain_modularity:.12f}",
            "standard_modularity_gamma1": f"{standard_modularity_gamma1:.12f}",
        }
        rows.append(row)
        membership_path = out_dir / f"membership_resolution_{label}.tsv"
        write_membership(membership_path, author_ids, membership, node_rows)
        print(row, flush=True)

        metric_score = float(row[args.best_metric])
        if metric_score > best_score:
            best_score = metric_score
            best_row = row
            best_membership_path = membership_path

    summary_path = out_dir / "louvain_resolution_summary.tsv"
    write_dict_rows(summary_path, rows)

    if best_row is not None and best_membership_path is not None:
        best_path = out_dir / "best_membership.tsv"
        shutil.copyfile(best_membership_path, best_path)
        with (out_dir / "best_resolution.tsv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(best_row), delimiter="\t")
            writer.writeheader()
            writer.writerow(best_row)
        print(f"best_metric={args.best_metric}", flush=True)
        print(f"best_resolution={best_row['resolution']}", flush=True)
        print(f"best_membership={best_path}", flush=True)

    print(f"summary={summary_path}")


if __name__ == "__main__":
    main()
