#!/usr/bin/env python3
"""Plot pred-gold expert embedding cosine-distance distributions.

This uses the same expert embedding representation as
scripts/evaluate_soft_groundtruth_methods.py: each expert is represented by the
L2-normalized mean of their historical paper embeddings.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Sequence

os.environ.setdefault("MPLCONFIGDIR", str(Path("cache/matplotlib").resolve()))
sys.path.insert(0, str(Path("scripts").resolve()))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from evaluate_soft_groundtruth_methods import (  # noqa: E402
    add_random_predictions,
    build_user_embeddings,
    dedupe,
    load_all_predictions,
    load_expert_papers,
    read_tasks,
)


DISPLAY_METHODS = {
    "embedding_bfs:embedding_bfs_unique_assign_each_node_then_top_team_size_by_weighted_score": "Embedding BFS",
    "responsibility_cut_assign": "TaxaResp-GainCut",
    "expert_distribution_cut_assign": "TaxaResp-DistCut",
    "seq2seq_epoch15527": "Seq2seq",
    "random_seed_13": "Random mean 5",
    "random_seed_14": "Random mean 5",
    "random_seed_15": "Random mean 5",
    "random_seed_16": "Random mean 5",
    "random_seed_17": "Random mean 5",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task-nodes-jsonl", default="output/hierec_embedding_server_inputs/task_nodes.jsonl")
    p.add_argument("--seq2seq-pred-csv", default="output/test.fold0.epoch15527.pred.csv")
    p.add_argument("--indexes-pkl", default="output/indexes.pkl")
    p.add_argument("--expert-papers-tsv", default="output/all_expert_paper_embeddings/expert_papers.tsv")
    p.add_argument("--paper-ids-tsv", default="output/all_expert_paper_embeddings/paper_embedding_ids.tsv")
    p.add_argument("--paper-embeddings", default="output/all_expert_paper_embeddings/paper_embeddings.npy")
    p.add_argument("--out-dir", default="output/selected_method_results")
    p.add_argument("--threshold", type=float, default=0.30)
    p.add_argument("--random-runs", type=int, default=5)
    p.add_argument("--random-seed", type=int, default=13)
    return p.parse_args()


def selected_predictions(
    predictions: Dict[str, Dict[str, List[str]]],
) -> Dict[str, Dict[str, List[str]]]:
    selected: Dict[str, Dict[str, List[str]]] = {}
    for method, by_paper in predictions.items():
        if method not in DISPLAY_METHODS:
            continue
        selected[method] = by_paper
    return selected


def method_display_name(method: str) -> str:
    return DISPLAY_METHODS.get(method, method)


def collect_needed_experts(
    members_by_paper: Dict[str, List[str]],
    predictions: Dict[str, Dict[str, List[str]]],
) -> set[str]:
    needed = set()
    for members in members_by_paper.values():
        needed.update(members)
    for by_paper in predictions.values():
        for experts in by_paper.values():
            needed.update(experts)
    return needed


def write_pair_rows(
    path: Path,
    paper_order: Sequence[str],
    members_by_paper: Dict[str, List[str]],
    predictions: Dict[str, Dict[str, List[str]]],
    user_embeddings: Dict[str, np.ndarray],
    threshold: float,
) -> Dict[str, List[float]]:
    distances_by_method: Dict[str, List[float]] = {}
    fieldnames = [
        "display_method",
        "source_method",
        "paper_id",
        "pred_expert_id",
        "gold_expert_id",
        "cosine_distance",
        "within_threshold",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for method, by_paper in sorted(predictions.items()):
            display = method_display_name(method)
            distances = distances_by_method.setdefault(display, [])
            for paper_id in paper_order:
                preds = dedupe(by_paper.get(paper_id, []))
                golds = dedupe(members_by_paper.get(paper_id, []))
                for pred in preds:
                    pred_vec = user_embeddings.get(pred)
                    if pred_vec is None:
                        continue
                    for gold in golds:
                        gold_vec = user_embeddings.get(gold)
                        if gold_vec is None:
                            continue
                        distance = 1.0 - float(np.dot(pred_vec, gold_vec))
                        distances.append(distance)
                        writer.writerow(
                            {
                                "display_method": display,
                                "source_method": method,
                                "paper_id": paper_id,
                                "pred_expert_id": pred,
                                "gold_expert_id": gold,
                                "cosine_distance": f"{distance:.9f}",
                                "within_threshold": int(distance <= threshold),
                            }
                        )
    return distances_by_method


def write_summary(path: Path, distances_by_method: Dict[str, List[float]], threshold: float) -> None:
    fieldnames = [
        "display_method",
        "pairs",
        "within_threshold",
        "within_threshold_rate",
        "mean",
        "median",
        "p10",
        "p25",
        "p75",
        "p90",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for display, distances in sorted(distances_by_method.items()):
            arr = np.asarray(distances, dtype=np.float32)
            if arr.size == 0:
                continue
            writer.writerow(
                {
                    "display_method": display,
                    "pairs": int(arr.size),
                    "within_threshold": int((arr <= threshold).sum()),
                    "within_threshold_rate": f"{float((arr <= threshold).mean()):.6f}",
                    "mean": f"{float(arr.mean()):.6f}",
                    "median": f"{float(np.median(arr)):.6f}",
                    "p10": f"{float(np.quantile(arr, 0.10)):.6f}",
                    "p25": f"{float(np.quantile(arr, 0.25)):.6f}",
                    "p75": f"{float(np.quantile(arr, 0.75)):.6f}",
                    "p90": f"{float(np.quantile(arr, 0.90)):.6f}",
                }
            )


def flatten_distances(
    distances_by_method: Dict[str, List[float]],
    include_random: bool,
) -> List[float]:
    out: List[float] = []
    for display, distances in distances_by_method.items():
        if not include_random and display == "Random mean 5":
            continue
        out.extend(distances)
    return out


def write_combined_summary(
    path: Path,
    all_distances: Sequence[float],
    nonrandom_distances: Sequence[float],
    threshold: float,
) -> None:
    fieldnames = [
        "scope",
        "pairs",
        "within_threshold",
        "within_threshold_rate",
        "mean",
        "median",
        "p10",
        "p25",
        "p75",
        "p90",
    ]

    def row(scope: str, distances: Sequence[float]) -> dict:
        arr = np.asarray(distances, dtype=np.float32)
        return {
            "scope": scope,
            "pairs": int(arr.size),
            "within_threshold": int((arr <= threshold).sum()),
            "within_threshold_rate": f"{float((arr <= threshold).mean()):.6f}" if arr.size else "0.000000",
            "mean": f"{float(arr.mean()):.6f}" if arr.size else "0.000000",
            "median": f"{float(np.median(arr)):.6f}" if arr.size else "0.000000",
            "p10": f"{float(np.quantile(arr, 0.10)):.6f}" if arr.size else "0.000000",
            "p25": f"{float(np.quantile(arr, 0.25)):.6f}" if arr.size else "0.000000",
            "p75": f"{float(np.quantile(arr, 0.75)):.6f}" if arr.size else "0.000000",
            "p90": f"{float(np.quantile(arr, 0.90)):.6f}" if arr.size else "0.000000",
        }

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerow(row("all_pairs", all_distances))
        writer.writerow(row("nonrandom_pairs", nonrandom_distances))


def plot_single_distribution(
    path: Path,
    distances: Sequence[float],
    threshold: float,
    title: str,
) -> None:
    arr = np.asarray(distances, dtype=np.float32)
    fig, ax = plt.subplots(figsize=(9, 5))
    bins = np.linspace(0.0, 1.0, 51)
    ax.hist(arr, bins=bins, color="#4C78A8", alpha=0.84, density=True)
    ax.axvline(threshold, color="#D62728", linestyle="--", linewidth=1.8)
    ax.set_title(title)
    ax.set_xlabel("Cosine distance between predicted expert and ground-truth expert embeddings")
    ax.set_ylabel("Density")
    ax.grid(axis="y", alpha=0.25)
    if arr.size:
        rate = float((arr <= threshold).mean())
        ax.text(
            0.98,
            0.90,
            f"pairs={arr.size:,}\n<= {threshold:.2f}: {rate:.1%}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=10,
        )
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_distribution(path: Path, distances_by_method: Dict[str, List[float]], threshold: float) -> None:
    order = ["Embedding BFS", "TaxaResp-GainCut", "TaxaResp-DistCut", "Seq2seq", "Random mean 5"]
    bins = np.linspace(0.0, 1.0, 41)
    fig, axes = plt.subplots(len(order), 1, figsize=(9, 11), sharex=True, sharey=True)
    for ax, display in zip(axes, order):
        distances = distances_by_method.get(display, [])
        ax.hist(distances, bins=bins, color="#4C78A8", alpha=0.82, density=True)
        ax.axvline(threshold, color="#D62728", linestyle="--", linewidth=1.6)
        ax.set_ylabel(display)
        ax.grid(axis="y", alpha=0.25)
        if distances:
            rate = float((np.asarray(distances) <= threshold).mean())
            ax.text(
                0.98,
                0.78,
                f"pairs={len(distances):,}\n<= {threshold:.2f}: {rate:.1%}",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=9,
            )
    axes[-1].set_xlabel("Cosine distance between predicted expert and ground-truth expert embeddings")
    fig.suptitle("Embedding Similarity Pair Distance Distribution", y=0.995)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    paper_order, members_by_paper, team_size_by_paper = read_tasks(Path(args.task_nodes_jsonl))
    soft_args = SimpleNamespace(
        seq2seq_pred_csv=args.seq2seq_pred_csv,
        indexes_pkl=args.indexes_pkl,
        method_prediction=[],
    )
    all_predictions = load_all_predictions(soft_args, paper_order, members_by_paper)
    method_pool = collect_needed_experts(members_by_paper, all_predictions)
    add_random_predictions(
        all_predictions,
        paper_order,
        team_size_by_paper,
        sorted(method_pool),
        args.random_runs,
        args.random_seed,
    )
    predictions = selected_predictions(all_predictions)

    needed_experts = collect_needed_experts(members_by_paper, predictions)
    expert_papers = load_expert_papers(Path(args.expert_papers_tsv), needed_experts)
    user_embeddings = build_user_embeddings(
        Path(args.paper_ids_tsv),
        Path(args.paper_embeddings),
        expert_papers,
    )

    pair_path = out_dir / "embedding_similarity_pair_distances.tsv"
    summary_path = out_dir / "embedding_similarity_pair_distance_summary.tsv"
    combined_summary_path = out_dir / "embedding_similarity_all_pair_distance_summary.tsv"
    plot_path = out_dir / "embedding_similarity_pair_distance_distribution.png"
    all_plot_path = out_dir / "embedding_similarity_all_pair_distance_distribution.png"
    nonrandom_plot_path = out_dir / "embedding_similarity_nonrandom_pair_distance_distribution.png"
    distances_by_method = write_pair_rows(
        pair_path,
        paper_order,
        members_by_paper,
        predictions,
        user_embeddings,
        args.threshold,
    )
    write_summary(summary_path, distances_by_method, args.threshold)
    all_distances = flatten_distances(distances_by_method, include_random=True)
    nonrandom_distances = flatten_distances(distances_by_method, include_random=False)
    write_combined_summary(combined_summary_path, all_distances, nonrandom_distances, args.threshold)
    plot_distribution(plot_path, distances_by_method, args.threshold)
    plot_single_distribution(
        all_plot_path,
        all_distances,
        args.threshold,
        "Embedding Similarity Pair Distance Distribution - All Pairs",
    )
    plot_single_distribution(
        nonrandom_plot_path,
        nonrandom_distances,
        args.threshold,
        "Embedding Similarity Pair Distance Distribution - Non-random Pairs",
    )

    print(f"pairs={pair_path}")
    print(f"summary={summary_path}")
    print(f"combined_summary={combined_summary_path}")
    print(f"plot={plot_path}")
    print(f"all_plot={all_plot_path}")
    print(f"nonrandom_plot={nonrandom_plot_path}")


if __name__ == "__main__":
    main()
