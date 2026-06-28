#!/usr/bin/env python3
"""Evaluate direct log-sum IDF cosine with node embeddings.

This is the embedding analogue of direct_log_sum_with_idf_cosine:
- task vector: weighted sum of direct task leaf-node embeddings with IDF
- expert vector: weighted sum of direct expert node embeddings with log(1+w) and IDF
- ranking score: cosine between the aggregated task/expert vectors
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from embedding_pipeline_utils import load_fos_map, load_tasks, read_jsonl


VARIANT = "embedding_direct_log_sum_with_idf_cosine_no_label"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate embedding direct log-sum IDF cosine")
    p.add_argument("--tasks-csv", default="data_preprocess/teams_2020plus_with_skill_weights.csv")
    p.add_argument("--fos-map", default="data/dblp/FieldsOfStudy.txt")
    p.add_argument("--expert-node-evidence-jsonl", required=True)
    p.add_argument("--task-node-ids", required=True)
    p.add_argument("--task-node-embeddings", required=True)
    p.add_argument("--expert-node-ids", required=True)
    p.add_argument("--expert-node-embeddings", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--top-k-output", type=int, default=20)
    p.add_argument("--comparison-tsv", default="")
    return p.parse_args()


def mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def read_id_rows(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def load_embedding_id_to_row(path: Path) -> Dict[str, int]:
    out = {}
    with path.open("r", encoding="utf-8") as f:
        for idx, row in enumerate(csv.DictReader(f, delimiter="\t")):
            out[str(row["id"])] = idx
    return out


def l2_rows(arr: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.maximum(norms, 1e-12)


def metrics_at_ks(ranked_ids: Sequence[str], positives: set, ks: Sequence[int]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    n_pos = len(positives)
    for k in ks:
        top = ranked_ids[:k]
        hits = [1 if expert_id in positives else 0 for expert_id in top]
        hit_count = sum(hits)
        precision = hit_count / k if k else 0.0
        recall = hit_count / n_pos if n_pos else 0.0

        dcg = sum(rel / math.log2(rank + 2) for rank, rel in enumerate(hits))
        ideal_hits = min(n_pos, k)
        idcg = sum(1.0 / math.log2(rank + 2) for rank in range(ideal_hits))
        ndcg = dcg / idcg if idcg > 0 else 0.0

        running_hits = 0
        precision_sum = 0.0
        for rank, rel in enumerate(hits, start=1):
            if rel:
                running_hits += 1
                precision_sum += running_hits / rank
        map_k = precision_sum / min(n_pos, k) if n_pos else 0.0

        out[f"mean_precision_at_{k}"] = precision
        out[f"mean_recall_at_{k}"] = recall
        out[f"mean_ndcg_at_{k}"] = ndcg
        out[f"mean_map_at_{k}"] = map_k
    return out


def load_expert_direct_weights(path: Path) -> Tuple[Dict[str, Dict[str, float]], Dict[str, str]]:
    by_expert: Dict[str, Dict[str, float]] = defaultdict(dict)
    expert_names: Dict[str, str] = {}
    for row in read_jsonl(path):
        expert_id = str(row["expert_id"])
        node_id = str(row["node_id"])
        weight = float(row.get("direct_weight_sum") or 0.0)
        if weight <= 0:
            continue
        by_expert[expert_id][node_id] = weight
        expert_names.setdefault(expert_id, row.get("expert_name") or expert_id)
    return by_expert, expert_names


def build_idf(expert_direct_weights: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    n = len(expert_direct_weights)
    df: Dict[str, int] = defaultdict(int)
    for weights in expert_direct_weights.values():
        for node_id in weights:
            df[node_id] += 1
    return {node_id: math.log((n + 1) / (cnt + 1)) + 1.0 for node_id, cnt in df.items()}


def build_task_matrix(
    tasks: List[dict],
    task_node_id_to_row: Dict[str, int],
    task_arr: np.ndarray,
    idf: Dict[str, float],
) -> Tuple[np.ndarray, List[dict]]:
    vectors = []
    kept_tasks = []
    dim = task_arr.shape[1]
    for task in tasks:
        vec = np.zeros(dim, dtype=np.float32)
        for node_id, weight, _ in task["direct"]:
            emb_id = f"{task['paper_id']}::{node_id}"
            row_idx = task_node_id_to_row.get(emb_id)
            if row_idx is None:
                continue
            coeff = float(weight) * idf.get(node_id, 1.0)
            if coeff > 0:
                vec += coeff * task_arr[row_idx]
        norm = np.linalg.norm(vec)
        if norm <= 0:
            continue
        vectors.append(vec / norm)
        kept_tasks.append(task)
    return np.vstack(vectors).astype(np.float32), kept_tasks


def build_expert_matrix(
    expert_direct_weights: Dict[str, Dict[str, float]],
    expert_node_id_to_row: Dict[str, int],
    expert_arr: np.ndarray,
    idf: Dict[str, float],
) -> Tuple[np.ndarray, List[str]]:
    expert_ids = sorted(expert_direct_weights)
    dim = expert_arr.shape[1]
    matrix = np.zeros((len(expert_ids), dim), dtype=np.float32)
    for idx, expert_id in enumerate(expert_ids, start=1):
        if idx % 1000 == 0:
            print(f"expert_vector_progress {idx:,}/{len(expert_ids):,}", flush=True)
        vec = matrix[idx - 1]
        for node_id, weight in expert_direct_weights[expert_id].items():
            emb_id = f"{expert_id}::{node_id}"
            row_idx = expert_node_id_to_row.get(emb_id)
            if row_idx is None:
                continue
            coeff = math.log1p(weight) * idf.get(node_id, 1.0)
            if coeff > 0:
                vec += coeff * expert_arr[row_idx]
    matrix = l2_rows(matrix).astype(np.float32)
    return matrix, expert_ids


def append_to_comparison(comparison_path: Path, metric_row: dict) -> None:
    if not comparison_path:
        return
    with comparison_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        fieldnames = reader.fieldnames
        rows = [row for row in reader if row.get("variant") != VARIANT]
    if not fieldnames:
        raise ValueError(f"missing header in {comparison_path}")
    row = {k: "" for k in fieldnames}
    row.update(
        {
            "variant": VARIANT,
            "expert_value": "sum log(1 + direct_weight_sum) * idf * expert_node_embedding; no taxonomy node label",
            "task_zero_or_missing_weight": "0",
            "uses_idf": "yes",
            "uses_cosine": "yes",
            "mean_recall_at_team_size": metric_row["mean_recall_at_team_size"],
            "mean_precision_at_team_size": metric_row["mean_precision_at_team_size"],
            "mean_precision_at_2": metric_row["mean_precision_at_2"],
            "mean_recall_at_2": metric_row["mean_recall_at_2"],
            "mean_ndcg_at_2": metric_row["mean_ndcg_at_2"],
            "mean_map_at_2": metric_row["mean_map_at_2"],
            "mean_precision_at_5": metric_row["mean_precision_at_5"],
            "mean_recall_at_5": metric_row["mean_recall_at_5"],
            "mean_ndcg_at_5": metric_row["mean_ndcg_at_5"],
            "mean_map_at_5": metric_row["mean_map_at_5"],
            "mean_precision_at_10": metric_row["mean_precision_at_10"],
            "mean_recall_at_10": metric_row["mean_recall_at_10"],
            "mean_ndcg_at_10": metric_row["mean_ndcg_at_10"],
            "mean_map_at_10": metric_row["mean_map_at_10"],
            "avg_selected_after_top_team_size": metric_row["avg_selected_after_top_team_size"],
            "micro_precision_at_team_size": metric_row["micro_precision_at_team_size"],
            "micro_recall_at_team_size": metric_row["micro_recall_at_team_size"],
        }
    )
    rows.append(row)
    with comparison_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("loading tasks", flush=True)
    name_to_id, id_to_name, _ = load_fos_map(Path(args.fos_map))
    tasks = [t for t in load_tasks(Path(args.tasks_csv), name_to_id, id_to_name) if t["direct"]]

    print("loading direct expert weights", flush=True)
    expert_direct_weights, expert_names = load_expert_direct_weights(
        Path(args.expert_node_evidence_jsonl)
    )
    idf = build_idf(expert_direct_weights)

    print("loading task embeddings", flush=True)
    task_node_id_to_row = load_embedding_id_to_row(Path(args.task_node_ids))
    task_arr = np.load(args.task_node_embeddings, mmap_mode="r")
    task_matrix, tasks = build_task_matrix(tasks, task_node_id_to_row, task_arr, idf)

    print("loading expert embeddings", flush=True)
    expert_node_id_to_row = load_embedding_id_to_row(Path(args.expert_node_ids))
    expert_arr = np.load(args.expert_node_embeddings, mmap_mode="r")
    expert_matrix, expert_ids = build_expert_matrix(
        expert_direct_weights, expert_node_id_to_row, expert_arr, idf
    )

    print("scoring", flush=True)
    scores = task_matrix @ expert_matrix.T
    ks = (2, 5, 10)
    metric_lists: Dict[str, List[float]] = defaultdict(list)
    prediction_rows = []
    micro_hits = 0
    micro_selected = 0
    micro_positives = 0
    selected_counts = []

    for task_idx, task in enumerate(tasks):
        row_scores = scores[task_idx]
        order = np.argsort(-row_scores)
        ranked_ids = [expert_ids[i] for i in order]
        positives = set(task["members"])
        team_size = max(1, int(task["team_size"]))
        selected = ranked_ids[:team_size]
        hits = len(positives.intersection(selected))
        metric_lists["mean_precision_at_team_size"].append(hits / len(selected) if selected else 0.0)
        metric_lists["mean_recall_at_team_size"].append(hits / len(positives) if positives else 0.0)
        micro_hits += hits
        micro_selected += len(selected)
        micro_positives += len(positives)
        selected_counts.append(len(selected))
        for key, value in metrics_at_ks(ranked_ids, positives, ks).items():
            metric_lists[key].append(value)
        for rank, expert_idx in enumerate(order[: args.top_k_output], start=1):
            expert_id = expert_ids[int(expert_idx)]
            prediction_rows.append(
                {
                    "method": VARIANT,
                    "paper_id": task["paper_id"],
                    "rank": rank,
                    "expert_id": expert_id,
                    "expert_name": expert_names.get(expert_id, expert_id),
                    "score": f"{float(row_scores[expert_idx]):.6f}",
                    "is_actual_member": "1" if expert_id in positives else "0",
                }
            )

    metric_row = {
        "variant": VARIANT,
        "tasks": len(tasks),
        "mean_precision_at_team_size": f"{mean(metric_lists['mean_precision_at_team_size']):.12f}",
        "mean_recall_at_team_size": f"{mean(metric_lists['mean_recall_at_team_size']):.12f}",
        "micro_precision_at_team_size": f"{(micro_hits / micro_selected) if micro_selected else 0.0:.12f}",
        "micro_recall_at_team_size": f"{(micro_hits / micro_positives) if micro_positives else 0.0:.12f}",
        "avg_selected_after_top_team_size": f"{mean(selected_counts):.6f}",
    }
    for metric in (
        "mean_precision_at_2",
        "mean_recall_at_2",
        "mean_ndcg_at_2",
        "mean_map_at_2",
        "mean_precision_at_5",
        "mean_recall_at_5",
        "mean_ndcg_at_5",
        "mean_map_at_5",
        "mean_precision_at_10",
        "mean_recall_at_10",
        "mean_ndcg_at_10",
        "mean_map_at_10",
    ):
        metric_row[metric] = f"{mean(metric_lists[metric]):.12f}"

    with (out_dir / "metrics_summary.tsv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(metric_row), delimiter="\t")
        writer.writeheader()
        writer.writerow(metric_row)

    with (out_dir / "predictions_topk.tsv").open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "method",
            "paper_id",
            "rank",
            "expert_id",
            "expert_name",
            "score",
            "is_actual_member",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(prediction_rows)

    if args.comparison_tsv:
        append_to_comparison(Path(args.comparison_tsv), metric_row)

    print(f"tasks={len(tasks)}")
    print(
        f"{VARIANT} p={100 * float(metric_row['mean_precision_at_team_size']):.4f}% "
        f"r={100 * float(metric_row['mean_recall_at_team_size']):.4f}%"
    )
    print(f"out_dir={out_dir}")


if __name__ == "__main__":
    main()
