#!/usr/bin/env python3
"""Plot all same-node task/expert embedding cosine-distance pairs.

Pair definition:
  task_node = paper_id::node_id
  expert_node = expert_id::node_id

For every task_node, this script pairs it with every expert_node that has the
same taxonomy/FoS node_id, then plots the distribution of
1 - cosine(task_node_embedding, expert_node_embedding).
"""

from __future__ import annotations

import argparse
import csv
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

os.environ.setdefault("MPLCONFIGDIR", str(Path("cache/matplotlib").resolve()))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--task-node-ids",
        default="output/all_expert_paper_embeddings/task_node_embedding_ids_strict_v2_no_label.tsv",
    )
    p.add_argument(
        "--task-node-embeddings",
        default="output/all_expert_paper_embeddings/task_node_embeddings_strict_v2_no_label.npy",
    )
    p.add_argument(
        "--expert-node-ids",
        default="output/all_expert_paper_embeddings/expert_node_embedding_ids_no_label.tsv",
    )
    p.add_argument(
        "--expert-node-embeddings",
        default="output/all_expert_paper_embeddings/expert_node_embeddings_no_label.npy",
    )
    p.add_argument("--out-dir", default="output/selected_method_results")
    p.add_argument("--threshold", type=float, default=0.30)
    p.add_argument("--bins", type=int, default=120)
    p.add_argument("--min-distance", type=float, default=-0.05)
    p.add_argument("--max-distance", type=float, default=1.05)
    p.add_argument("--expert-chunk-size", type=int, default=2048)
    return p.parse_args()


def read_node_index(path: Path) -> tuple[Dict[str, List[int]], List[dict]]:
    by_node: Dict[str, List[int]] = defaultdict(list)
    rows: List[dict] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row_idx, row in enumerate(reader):
            rows.append(row)
            by_node[str(row["node_id"])].append(row_idx)
    return by_node, rows


def histogram_quantile(counts: np.ndarray, edges: np.ndarray, q: float) -> float:
    total = int(counts.sum())
    if total <= 0:
        return 0.0
    target = q * total
    cumulative = np.cumsum(counts)
    idx = int(np.searchsorted(cumulative, target, side="left"))
    idx = min(max(idx, 0), len(counts) - 1)
    before = int(cumulative[idx - 1]) if idx > 0 else 0
    in_bin = int(counts[idx])
    if in_bin <= 0:
        return float(edges[idx])
    fraction = (target - before) / in_bin
    fraction = min(max(float(fraction), 0.0), 1.0)
    return float(edges[idx] + fraction * (edges[idx + 1] - edges[idx]))


def summarize_histogram(
    counts: np.ndarray,
    edges: np.ndarray,
    mean: float,
    std: float,
    min_x: float,
    max_x: float,
) -> dict:
    return {
        "mean": f"{mean:.9f}",
        "std": f"{std:.9f}",
        "min": f"{min_x:.9f}",
        "p01": f"{histogram_quantile(counts, edges, 0.01):.9f}",
        "p05": f"{histogram_quantile(counts, edges, 0.05):.9f}",
        "p10": f"{histogram_quantile(counts, edges, 0.10):.9f}",
        "p25": f"{histogram_quantile(counts, edges, 0.25):.9f}",
        "median": f"{histogram_quantile(counts, edges, 0.50):.9f}",
        "p75": f"{histogram_quantile(counts, edges, 0.75):.9f}",
        "p90": f"{histogram_quantile(counts, edges, 0.90):.9f}",
        "p95": f"{histogram_quantile(counts, edges, 0.95):.9f}",
        "p99": f"{histogram_quantile(counts, edges, 0.99):.9f}",
        "max": f"{max_x:.9f}",
    }


def write_histogram(path: Path, counts: np.ndarray, edges: np.ndarray) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["bin_left", "bin_right", "count", "density"],
            delimiter="\t",
        )
        writer.writeheader()
        total = int(counts.sum())
        widths = np.diff(edges)
        for left, right, count, width in zip(edges[:-1], edges[1:], counts, widths):
            density = (float(count) / total / float(width)) if total else 0.0
            writer.writerow(
                {
                    "bin_left": f"{float(left):.9f}",
                    "bin_right": f"{float(right):.9f}",
                    "count": int(count),
                    "density": f"{density:.12f}",
                }
            )


def write_summary(
    path: Path,
    total_pairs: int,
    underflow: int,
    overflow: int,
    within_threshold: int,
    threshold: float,
    moments: dict,
) -> None:
    row = {
        "scope": "all_same_node_task_expert_pairs",
        "pairs": total_pairs,
        "within_threshold": within_threshold,
        "within_threshold_rate": f"{(within_threshold / total_pairs) if total_pairs else 0.0:.9f}",
        "threshold": f"{threshold:.6f}",
        "hist_underflow": underflow,
        "hist_overflow": overflow,
        **moments,
    }
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row), delimiter="\t")
        writer.writeheader()
        writer.writerow(row)


def plot_histogram(
    path: Path,
    counts: np.ndarray,
    edges: np.ndarray,
    total_pairs: int,
    within_threshold: int,
    threshold: float,
) -> None:
    widths = np.diff(edges)
    density = counts / counts.sum() / widths if counts.sum() else counts
    centers = (edges[:-1] + edges[1:]) / 2.0
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(centers, density, align="center", width=widths, color="#4C78A8", alpha=0.84)
    ax.axvline(threshold, color="#D62728", linestyle="--", linewidth=1.8)
    ax.set_title("Task-node vs Same-node Expert-node Embedding Distance Distribution")
    ax.set_xlabel("Cosine distance: 1 - cosine(task_node_embedding, expert_node_embedding)")
    ax.set_ylabel("Density")
    ax.grid(axis="y", alpha=0.25)
    rate = (within_threshold / total_pairs) if total_pairs else 0.0
    ax.text(
        0.98,
        0.90,
        f"pairs={total_pairs:,}\n<= {threshold:.2f}: {rate:.3%}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("loading id indexes", flush=True)
    task_by_node, _task_rows = read_node_index(Path(args.task_node_ids))
    expert_by_node, _expert_rows = read_node_index(Path(args.expert_node_ids))

    print("opening embeddings", flush=True)
    task_arr = np.load(args.task_node_embeddings, mmap_mode="r")
    expert_arr = np.load(args.expert_node_embeddings, mmap_mode="r")

    edges = np.linspace(args.min_distance, args.max_distance, args.bins + 1)
    counts = np.zeros(args.bins, dtype=np.int64)
    underflow = 0
    overflow = 0
    total_pairs = 0
    within_threshold = 0
    sum_x = 0.0
    sum_x2 = 0.0
    min_x = float("inf")
    max_x = float("-inf")

    matching_nodes = sorted(set(task_by_node) & set(expert_by_node))
    pair_counts = Counter(
        {
            node_id: len(task_by_node[node_id]) * len(expert_by_node[node_id])
            for node_id in matching_nodes
        }
    )
    print(
        f"task_nodes={sum(len(v) for v in task_by_node.values()):,} "
        f"expert_nodes={sum(len(v) for v in expert_by_node.values()):,} "
        f"matching_nodes={len(matching_nodes):,} "
        f"pairs={sum(pair_counts.values()):,}",
        flush=True,
    )

    for node_idx, node_id in enumerate(
        sorted(matching_nodes, key=lambda n: pair_counts[n], reverse=True),
        start=1,
    ):
        task_indices = task_by_node[node_id]
        expert_indices = expert_by_node[node_id]
        if node_idx == 1 or node_idx % 50 == 0:
            print(
                f"progress nodes={node_idx:,}/{len(matching_nodes):,} "
                f"node_id={node_id} pairs={pair_counts[node_id]:,}",
                flush=True,
            )

        task_mat = np.asarray(task_arr[task_indices], dtype=np.float32)
        for start in range(0, len(expert_indices), args.expert_chunk_size):
            chunk_indices = expert_indices[start : start + args.expert_chunk_size]
            expert_mat = np.asarray(expert_arr[chunk_indices], dtype=np.float32)
            distances = 1.0 - (task_mat @ expert_mat.T).ravel()

            total_pairs += int(distances.size)
            within_threshold += int(np.count_nonzero(distances <= args.threshold))
            distances64 = np.asarray(distances, dtype=np.float64)
            sum_x += float(distances64.sum())
            sum_x2 += float(np.dot(distances64, distances64))
            min_x = min(min_x, float(distances.min()))
            max_x = max(max_x, float(distances.max()))

            hist, _ = np.histogram(distances, bins=edges)
            counts += hist
            underflow += int(np.count_nonzero(distances < edges[0]))
            overflow += int(np.count_nonzero(distances > edges[-1]))

    if total_pairs == 0:
        raise SystemExit("No same-node task/expert pairs found.")

    mean = sum_x / total_pairs
    variance = max(0.0, (sum_x2 / total_pairs) - mean * mean)
    moments = summarize_histogram(counts, edges, mean, variance ** 0.5, min_x, max_x)

    hist_path = out_dir / "task_expert_same_node_embedding_distance_histogram.tsv"
    summary_path = out_dir / "task_expert_same_node_embedding_distance_summary.tsv"
    plot_path = out_dir / "task_expert_same_node_embedding_distance_distribution.png"
    write_histogram(hist_path, counts, edges)
    write_summary(
        summary_path,
        total_pairs,
        underflow,
        overflow,
        within_threshold,
        args.threshold,
        moments,
    )
    plot_histogram(plot_path, counts, edges, total_pairs, within_threshold, args.threshold)

    print(f"histogram={hist_path}")
    print(f"summary={summary_path}")
    print(f"plot={plot_path}")


if __name__ == "__main__":
    main()
