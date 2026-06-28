#!/usr/bin/env python3
"""Evaluate BFS unique node assignment with embedding similarity.

For each task, taxonomy nodes are visited in level/BFS order. Each node assigns
the best same-node expert whose expert id has not already been assigned for the
task. Final team prediction is the top ground-truth team-size assigned experts
by either similarity * node_importance or similarity * log-sum subtree weight.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from embedding_pipeline_utils import read_jsonl


METHOD_WEIGHTED = "embedding_bfs_unique_assign_each_node_then_top_team_size_by_weighted_score"
METHOD_LOG_SUM = "embedding_bfs_unique_assign_each_node_then_top_team_size_by_weighted_score_log_sum"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate embedding BFS unique assignment")
    p.add_argument("--task-nodes-jsonl", required=True)
    p.add_argument("--task-node-ids", required=True)
    p.add_argument("--task-node-embeddings", required=True)
    p.add_argument("--expert-node-ids", required=True)
    p.add_argument("--expert-node-embeddings", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--top-k-output", type=int, default=20)
    p.add_argument("--candidate-chunk-size", type=int, default=8192)
    p.add_argument(
        "--precompute-top-candidates",
        type=int,
        default=512,
        help="Keep this many same-node candidates per task-node query before BFS unique assignment.",
    )
    return p.parse_args()


def mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def read_id_rows(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def load_task_embedding_table(ids_path: Path, npy_path: Path) -> Dict[str, np.ndarray]:
    ids = [row["id"] for row in read_id_rows(ids_path)]
    arr = np.load(npy_path, mmap_mode="r")
    if len(ids) != arr.shape[0]:
        raise ValueError(f"task ids/embedding mismatch: {len(ids)} vs {arr.shape[0]}")
    return {id_: np.array(arr[i], dtype=np.float32) for i, id_ in enumerate(ids)}


def as_members(value) -> set:
    if isinstance(value, list):
        return {str(x) for x in value}
    if isinstance(value, str):
        return {x for x in value.replace("|", ",").split(",") if x}
    return set()


def safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int(value, default: int = 99) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def parse_subtree_log_sum(text: str) -> float:
    total = 0.0
    for part in str(text or "").split(";"):
        part = part.strip()
        if not part or ":" not in part:
            continue
        _, raw_weight = part.rsplit(":", 1)
        weight = safe_float(raw_weight, 0.0)
        if weight > 0:
            total += math.log1p(weight)
    return total


def build_expert_node_index(
    path: Path,
) -> Tuple[Dict[str, List[int]], List[str], Dict[str, str], int]:
    by_node: Dict[str, List[int]] = defaultdict(list)
    expert_ids: List[str] = []
    expert_names: Dict[str, str] = {}
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row_idx, row in enumerate(reader):
            node_id = str(row["node_id"])
            expert_id = str(row["expert_id"])
            by_node[node_id].append(row_idx)
            expert_ids.append(expert_id)
            expert_names.setdefault(expert_id, row.get("expert_name") or expert_id)
            if (row_idx + 1) % 500000 == 0:
                print(f"expert_id_index_progress rows={row_idx + 1:,}", flush=True)
    return by_node, expert_ids, expert_names, len(expert_ids)


def best_unassigned_candidate(
    ranked_candidates: List[Tuple[str, float]],
    used_experts: set,
) -> Tuple[str, float] | None:
    for expert_id, score in ranked_candidates:
        if expert_id not in used_experts:
            return expert_id, score
    return None


def precompute_task_node_rankings(
    task_rows_by_paper: Dict[str, list],
    task_embeddings: Dict[str, np.ndarray],
    experts_by_node: Dict[str, List[int]],
    expert_ids: List[str],
    expert_arr: np.ndarray,
    top_candidates: int,
) -> Dict[str, List[Tuple[str, float]]]:
    rows_by_node: Dict[str, list] = defaultdict(list)
    for rows in task_rows_by_paper.values():
        for row in rows:
            if row["task_node_id"] in task_embeddings:
                rows_by_node[str(row["node_id"])].append(row)

    rankings: Dict[str, List[Tuple[str, float]]] = {}
    total_nodes = len(rows_by_node)
    for idx, (node_id, rows) in enumerate(sorted(rows_by_node.items()), start=1):
        if idx % 25 == 0:
            print(f"precompute_progress nodes={idx:,}/{total_nodes:,}", flush=True)
        candidate_rows = experts_by_node.get(node_id, [])
        if not candidate_rows:
            continue
        mat = np.asarray(expert_arr[candidate_rows], dtype=np.float32)
        queries = np.vstack([task_embeddings[row["task_node_id"]] for row in rows]).astype(np.float32)
        scores = queries @ mat.T
        keep = min(top_candidates, len(candidate_rows))
        for q_idx, row in enumerate(rows):
            row_scores = scores[q_idx]
            if keep < len(candidate_rows):
                top_pos = np.argpartition(-row_scores, keep - 1)[:keep]
                top_pos = top_pos[np.argsort(-row_scores[top_pos])]
            else:
                top_pos = np.argsort(-row_scores)
            rankings[row["task_node_id"]] = [
                (expert_ids[candidate_rows[pos]], float(row_scores[pos]))
                for pos in top_pos
            ]
    return rankings


def rank_nodes_bfs(rows: Iterable[dict]) -> List[dict]:
    return sorted(
        rows,
        key=lambda r: (
            safe_int(r.get("node_level"), 99),
            -safe_float(r.get("node_importance"), 0.0),
            str(r.get("node_id")),
        ),
    )


def summarize_method(
    method: str,
    task_results: List[dict],
) -> dict:
    macro_p = [r["precision"] for r in task_results]
    macro_r = [r["recall"] for r in task_results]
    micro_hits = sum(r["hits"] for r in task_results)
    micro_selected = sum(r["selected"] for r in task_results)
    micro_positives = sum(r["positives"] for r in task_results)
    return {
        "method": method,
        "tasks": len(task_results),
        "mean_precision_at_team_size": f"{mean(macro_p):.12f}",
        "mean_recall_at_team_size": f"{mean(macro_r):.12f}",
        "percent_precision_at_team_size": f"{100 * mean(macro_p):.6f}",
        "percent_recall_at_team_size": f"{100 * mean(macro_r):.6f}",
        "micro_precision_at_team_size": f"{(micro_hits / micro_selected) if micro_selected else 0.0:.12f}",
        "micro_recall_at_team_size": f"{(micro_hits / micro_positives) if micro_positives else 0.0:.12f}",
        "avg_assigned_nodes": f"{mean([r['assigned'] for r in task_results]):.6f}",
        "avg_selected_experts": f"{mean([r['selected'] for r in task_results]):.6f}",
    }


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("loading task embeddings", flush=True)
    task_embeddings = load_task_embedding_table(
        Path(args.task_node_ids), Path(args.task_node_embeddings)
    )

    print("loading expert ids/index", flush=True)
    experts_by_node, expert_ids, expert_names, expert_row_count = build_expert_node_index(
        Path(args.expert_node_ids)
    )
    print("opening expert embeddings", flush=True)
    expert_arr = np.load(args.expert_node_embeddings, mmap_mode="r")
    if expert_row_count != expert_arr.shape[0]:
        raise ValueError(
            f"expert ids/embedding mismatch: {expert_row_count} vs {expert_arr.shape[0]}"
        )

    task_rows_by_paper: Dict[str, list] = defaultdict(list)
    task_info = {}
    for row in read_jsonl(Path(args.task_nodes_jsonl)):
        paper_id = str(row["paper_id"])
        node_id = str(row["node_id"])
        row = dict(row)
        row["task_node_id"] = f"{paper_id}::{node_id}"
        row["node_importance_value"] = safe_float(row.get("node_importance"), 0.0)
        row["node_log_sum_value"] = parse_subtree_log_sum(row.get("subtree_skills", ""))
        task_rows_by_paper[paper_id].append(row)
        task_info[paper_id] = {
            "team_size": int(row["team_size"]),
            "members": as_members(row.get("members")),
        }

    print("precomputing same-node embedding rankings", flush=True)
    task_node_rankings = precompute_task_node_rankings(
        task_rows_by_paper,
        task_embeddings,
        experts_by_node,
        expert_ids,
        expert_arr,
        args.precompute_top_candidates,
    )

    assignment_rows = []
    predictions_by_method = {METHOD_WEIGHTED: [], METHOD_LOG_SUM: []}
    results_by_method = {METHOD_WEIGHTED: [], METHOD_LOG_SUM: []}

    for task_idx, (paper_id, rows) in enumerate(sorted(task_rows_by_paper.items()), start=1):
        if task_idx % 25 == 0:
            print(f"eval_progress tasks={task_idx:,}/{len(task_rows_by_paper):,}", flush=True)
        positives = task_info[paper_id]["members"]
        team_size = max(1, task_info[paper_id]["team_size"])
        used_experts = set()
        assignments = []

        for row in rank_nodes_bfs(rows):
            ranked_candidates = task_node_rankings.get(row["task_node_id"], [])
            if not ranked_candidates:
                continue
            best = best_unassigned_candidate(ranked_candidates, used_experts)
            if best is None:
                continue
            expert_id, similarity = best
            used_experts.add(expert_id)
            weighted_score = similarity * row["node_importance_value"]
            log_sum_score = similarity * row["node_log_sum_value"]
            record = {
                "paper_id": paper_id,
                "node_id": str(row["node_id"]),
                "node_name": row.get("node_name", row["node_id"]),
                "node_level": row.get("node_level", ""),
                "node_importance": f"{row['node_importance_value']:.6f}",
                "node_log_sum": f"{row['node_log_sum_value']:.6f}",
                "subtree_skill_count": row.get("subtree_skill_count", ""),
                "expert_id": expert_id,
                "expert_name": expert_names.get(expert_id, expert_id),
                "similarity": f"{similarity:.6f}",
                "weighted_score": f"{weighted_score:.6f}",
                "log_sum_score": f"{log_sum_score:.6f}",
                "is_actual_member": "1" if expert_id in positives else "0",
            }
            assignments.append((record, weighted_score, log_sum_score))
            assignment_rows.append(record)

        for method, score_idx in ((METHOD_WEIGHTED, 1), (METHOD_LOG_SUM, 2)):
            ranked = sorted(assignments, key=lambda item: item[score_idx], reverse=True)
            selected = ranked[:team_size]
            selected_ids = [rec["expert_id"] for rec, _, _ in selected]
            hits = len(positives.intersection(selected_ids))
            results_by_method[method].append(
                {
                    "hits": hits,
                    "selected": len(selected_ids),
                    "positives": len(positives),
                    "precision": hits / len(selected_ids) if selected_ids else 0.0,
                    "recall": hits / len(positives) if positives else 0.0,
                    "assigned": len(assignments),
                }
            )
            for rank, (rec, weighted_score, log_sum_score) in enumerate(
                selected[: args.top_k_output], start=1
            ):
                predictions_by_method[method].append(
                    {
                        "method": method,
                        "paper_id": paper_id,
                        "rank": rank,
                        "expert_id": rec["expert_id"],
                        "expert_name": rec["expert_name"],
                        "score": f"{weighted_score if method == METHOD_WEIGHTED else log_sum_score:.6f}",
                        "similarity": rec["similarity"],
                        "best_node_id": rec["node_id"],
                        "best_node_name": rec["node_name"],
                        "is_actual_member": rec["is_actual_member"],
                    }
                )

    metric_rows = [
        summarize_method(METHOD_WEIGHTED, results_by_method[METHOD_WEIGHTED]),
        summarize_method(METHOD_LOG_SUM, results_by_method[METHOD_LOG_SUM]),
    ]

    with (out_dir / "metrics_summary.tsv").open("w", encoding="utf-8", newline="") as f:
        fieldnames = list(metric_rows[0])
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(metric_rows)

    with (out_dir / "node_assignments.tsv").open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "paper_id",
            "node_id",
            "node_name",
            "node_level",
            "node_importance",
            "node_log_sum",
            "subtree_skill_count",
            "expert_id",
            "expert_name",
            "similarity",
            "weighted_score",
            "log_sum_score",
            "is_actual_member",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(assignment_rows)

    with (out_dir / "predictions_team_size.tsv").open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "method",
            "paper_id",
            "rank",
            "expert_id",
            "expert_name",
            "score",
            "similarity",
            "best_node_id",
            "best_node_name",
            "is_actual_member",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for method in (METHOD_WEIGHTED, METHOD_LOG_SUM):
            writer.writerows(predictions_by_method[method])

    print(f"tasks={len(task_rows_by_paper)}")
    for row in metric_rows:
        print(
            f"{row['method']} p={float(row['percent_precision_at_team_size']):.4f}% "
            f"r={float(row['percent_recall_at_team_size']):.4f}%"
        )
    print(f"out_dir={out_dir}")


if __name__ == "__main__":
    main()
