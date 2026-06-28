#!/usr/bin/env python3
"""Plot pairwise expert embedding cosine-distance distribution.

This matches the expert representation used by the Embedding Similarity P/R
soft-groundtruth metric: an expert is represented by the L2-normalized mean of
their historical paper embeddings.
"""

from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

os.environ.setdefault("MPLCONFIGDIR", str(Path("cache/matplotlib").resolve()))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--expert-papers-tsv", default="output/all_expert_paper_embeddings/expert_papers.tsv")
    p.add_argument("--paper-ids-tsv", default="output/all_expert_paper_embeddings/paper_embedding_ids.tsv")
    p.add_argument("--paper-embeddings", default="output/all_expert_paper_embeddings/paper_embeddings.npy")
    p.add_argument("--out-dir", default="output/selected_method_results")
    p.add_argument("--threshold", type=float, default=0.30)
    p.add_argument("--bins", type=int, default=120)
    p.add_argument("--min-distance", type=float, default=-0.05)
    p.add_argument("--max-distance", type=float, default=1.05)
    p.add_argument("--chunk-size", type=int, default=512)
    return p.parse_args()


def read_paper_id_to_row(path: Path) -> Dict[str, int]:
    out: Dict[str, int] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for idx, row in enumerate(reader):
            out[str(row["id"])] = idx
    return out


def read_expert_papers(path: Path) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = defaultdict(list)
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            out[str(row["expert_id"])].append(str(row["paper_id"]))
    return dict(out)


def build_expert_embeddings(
    expert_papers: Dict[str, List[str]],
    paper_id_to_row: Dict[str, int],
    paper_arr: np.ndarray,
) -> tuple[List[str], np.ndarray, List[int]]:
    expert_ids: List[str] = []
    embeddings: List[np.ndarray] = []
    paper_counts: List[int] = []
    for idx, (expert_id, papers) in enumerate(sorted(expert_papers.items()), start=1):
        if idx % 1000 == 0:
            print(f"expert_embedding_progress experts={idx:,}/{len(expert_papers):,}", flush=True)
        rows = [paper_id_to_row[p] for p in papers if p in paper_id_to_row]
        if not rows:
            continue
        vec = np.asarray(paper_arr[rows], dtype=np.float32).mean(axis=0)
        norm = float(np.linalg.norm(vec))
        if norm <= 0:
            continue
        expert_ids.append(expert_id)
        embeddings.append((vec / norm).astype(np.float32))
        paper_counts.append(len(rows))
    if not embeddings:
        raise SystemExit("No expert embeddings could be built.")
    return expert_ids, np.vstack(embeddings).astype(np.float32), paper_counts


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
    fraction = min(max(float((target - before) / in_bin), 0.0), 1.0)
    return float(edges[idx] + fraction * (edges[idx + 1] - edges[idx]))


def write_histogram(path: Path, counts: np.ndarray, edges: np.ndarray) -> None:
    total = int(counts.sum())
    widths = np.diff(edges)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["bin_left", "bin_right", "count", "density"],
            delimiter="\t",
        )
        writer.writeheader()
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
    expert_count: int,
    pair_count: int,
    within_threshold: int,
    threshold: float,
    counts: np.ndarray,
    edges: np.ndarray,
    mean: float,
    std: float,
    min_x: float,
    max_x: float,
    avg_papers_per_expert: float,
) -> None:
    row = {
        "scope": "all_pairwise_expert_mean_paper_embeddings",
        "experts": expert_count,
        "pairs": pair_count,
        "within_threshold": within_threshold,
        "within_threshold_rate": f"{(within_threshold / pair_count) if pair_count else 0.0:.9f}",
        "threshold": f"{threshold:.6f}",
        "avg_papers_per_expert": f"{avg_papers_per_expert:.6f}",
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
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row), delimiter="\t")
        writer.writeheader()
        writer.writerow(row)


def plot_histogram(
    path: Path,
    counts: np.ndarray,
    edges: np.ndarray,
    pair_count: int,
    within_threshold: int,
    threshold: float,
) -> None:
    widths = np.diff(edges)
    density = counts / counts.sum() / widths if counts.sum() else counts
    centers = (edges[:-1] + edges[1:]) / 2.0
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(centers, density, align="center", width=widths, color="#4C78A8", alpha=0.84)
    ax.axvline(threshold, color="#D62728", linestyle="--", linewidth=1.8)
    ax.set_title("Pairwise Expert Embedding Distance Distribution")
    ax.set_xlabel("Cosine distance: 1 - cosine(expert_mean_paper_embedding_i, expert_mean_paper_embedding_j)")
    ax.set_ylabel("Density")
    ax.grid(axis="y", alpha=0.25)
    rate = (within_threshold / pair_count) if pair_count else 0.0
    ax.text(
        0.98,
        0.90,
        f"pairs={pair_count:,}\n<= {threshold:.2f}: {rate:.3%}",
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

    print("loading paper id index", flush=True)
    paper_id_to_row = read_paper_id_to_row(Path(args.paper_ids_tsv))
    print("loading expert paper lists", flush=True)
    expert_papers = read_expert_papers(Path(args.expert_papers_tsv))
    print("opening paper embeddings", flush=True)
    paper_arr = np.load(args.paper_embeddings, mmap_mode="r")
    print("building expert mean-paper embeddings", flush=True)
    expert_ids, expert_arr, paper_counts = build_expert_embeddings(
        expert_papers,
        paper_id_to_row,
        paper_arr,
    )

    n = expert_arr.shape[0]
    expected_pairs = n * (n - 1) // 2
    print(
        f"expert_embeddings={n:,} dim={expert_arr.shape[1]:,} "
        f"unordered_pairs={expected_pairs:,}",
        flush=True,
    )

    edges = np.linspace(args.min_distance, args.max_distance, args.bins + 1)
    counts = np.zeros(args.bins, dtype=np.int64)
    pair_count = 0
    within_threshold = 0
    sum_x = 0.0
    sum_x2 = 0.0
    min_x = float("inf")
    max_x = float("-inf")

    for start in range(0, n, args.chunk_size):
        end = min(start + args.chunk_size, n)
        print(f"pairwise_progress experts={start:,}/{n:,}", flush=True)
        sims = np.asarray(expert_arr[start:end] @ expert_arr.T, dtype=np.float32)
        for local_idx in range(end - start):
            global_idx = start + local_idx
            if global_idx + 1 >= n:
                continue
            distances = 1.0 - sims[local_idx, global_idx + 1 :]
            distances64 = np.asarray(distances, dtype=np.float64)
            pair_count += int(distances.size)
            within_threshold += int(np.count_nonzero(distances <= args.threshold))
            sum_x += float(distances64.sum())
            sum_x2 += float(np.dot(distances64, distances64))
            min_x = min(min_x, float(distances.min()))
            max_x = max(max_x, float(distances.max()))
            hist, _ = np.histogram(distances, bins=edges)
            counts += hist

    if pair_count != expected_pairs:
        raise RuntimeError(f"pair count mismatch: {pair_count} vs {expected_pairs}")

    mean = sum_x / pair_count
    variance = max(0.0, (sum_x2 / pair_count) - mean * mean)
    hist_path = out_dir / "expert_mean_paper_embedding_pair_distance_histogram.tsv"
    summary_path = out_dir / "expert_mean_paper_embedding_pair_distance_summary.tsv"
    plot_path = out_dir / "expert_mean_paper_embedding_pair_distance_distribution.png"
    write_histogram(hist_path, counts, edges)
    write_summary(
        summary_path,
        n,
        pair_count,
        within_threshold,
        args.threshold,
        counts,
        edges,
        mean,
        variance ** 0.5,
        min_x,
        max_x,
        float(np.mean(paper_counts)),
    )
    plot_histogram(plot_path, counts, edges, pair_count, within_threshold, args.threshold)
    print(f"histogram={hist_path}")
    print(f"summary={summary_path}")
    print(f"plot={plot_path}")


if __name__ == "__main__":
    main()
