#!/usr/bin/env python3
"""Evaluate fixed-budget node assignment using cached HieRec embeddings."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, Sequence

import numpy as np

from embedding_pipeline_utils import load_embedding_table, read_jsonl


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate cached HieRec embedding assignments")
    p.add_argument("--task-nodes-jsonl", required=True)
    p.add_argument("--task-node-ids", required=True)
    p.add_argument("--task-node-embeddings", required=True)
    p.add_argument("--expert-node-ids", required=True)
    p.add_argument("--expert-node-embeddings", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--top-k-output", type=int, default=20)
    return p.parse_args()


def mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def read_id_metadata(path: Path) -> Dict[str, dict]:
    out = {}
    with path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            out[row["id"]] = row
    return out


def as_members(value) -> set:
    if isinstance(value, list):
        return {str(x) for x in value}
    if isinstance(value, str):
        return {x for x in value.replace("|", ",").split(",") if x}
    return set()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    task_node_embeddings = load_embedding_table(
        Path(args.task_node_ids), Path(args.task_node_embeddings)
    )
    expert_node_embeddings = load_embedding_table(
        Path(args.expert_node_ids), Path(args.expert_node_embeddings)
    )
    expert_meta = read_id_metadata(Path(args.expert_node_ids))

    experts_by_node: Dict[str, list] = defaultdict(list)
    for emb_id, vec in expert_node_embeddings.items():
        meta = expert_meta[emb_id]
        node_id = meta["node_id"]
        experts_by_node[node_id].append((meta["expert_id"], emb_id, vec, meta))

    task_rows_by_paper: Dict[str, list] = defaultdict(list)
    task_info = {}
    for row in read_jsonl(Path(args.task_nodes_jsonl)):
        paper_id = str(row["paper_id"])
        node_id = str(row["node_id"])
        row["task_node_id"] = f"{paper_id}::{node_id}"
        task_rows_by_paper[paper_id].append(row)
        task_info[paper_id] = {
            "team_size": int(row["team_size"]),
            "members": as_members(row.get("members")),
        }

    assignment_rows = []
    prediction_rows = []
    macro_p = []
    macro_r = []
    micro_hits = 0
    micro_selected = 0
    micro_positives = 0
    assigned_counts = []
    unique_counts = []
    selected_counts = []

    for paper_id, rows in task_rows_by_paper.items():
        positives = task_info[paper_id]["members"]
        team_size = max(1, task_info[paper_id]["team_size"])
        node_assignments = []
        for row in rows:
            task_vec = task_node_embeddings.get(row["task_node_id"])
            if task_vec is None:
                continue
            node_id = str(row["node_id"])
            candidates = experts_by_node.get(node_id, [])
            if not candidates:
                continue
            best = None
            for expert_id, emb_id, expert_vec, meta in candidates:
                score = float(np.dot(task_vec, expert_vec))
                if best is None or score > best[1]:
                    best = (expert_id, score, emb_id, meta)
            if best is None:
                continue
            expert_id, score, emb_id, meta = best
            node_importance = float(row.get("node_importance") or 0.0)
            weighted_score = score * node_importance
            node_assignments.append((node_id, expert_id, score, weighted_score, row, meta))
            assignment_rows.append(
                {
                    "paper_id": paper_id,
                    "node_id": node_id,
                    "node_name": row.get("node_name", node_id),
                    "node_level": row.get("node_level", ""),
                    "node_importance": f"{node_importance:.6f}",
                    "subtree_skill_count": row.get("subtree_skill_count", ""),
                    "expert_id": expert_id,
                    "expert_name": meta.get("expert_name", expert_id),
                    "score": f"{score:.6f}",
                    "weighted_score": f"{weighted_score:.6f}",
                    "is_actual_member": "1" if expert_id in positives else "0",
                }
            )

        assigned_counts.append(len(node_assignments))
        best_by_expert = {}
        for node_id, expert_id, score, weighted_score, row, meta in node_assignments:
            prev = best_by_expert.get(expert_id)
            if prev is None or weighted_score > prev[0]:
                best_by_expert[expert_id] = (weighted_score, node_id, score, row, meta)
        ranked = sorted(best_by_expert.items(), key=lambda x: x[1][0], reverse=True)
        unique_counts.append(len(ranked))
        selected = ranked[:team_size]
        selected_ids = [expert_id for expert_id, _ in selected]
        selected_counts.append(len(selected_ids))

        hits = len(positives.intersection(selected_ids))
        macro_p.append(hits / len(selected_ids) if selected_ids else 0.0)
        macro_r.append(hits / len(positives) if positives else 0.0)
        micro_hits += hits
        micro_selected += len(selected_ids)
        micro_positives += len(positives)

        for rank, (expert_id, (weighted_score, node_id, score, row, meta)) in enumerate(
            selected[: args.top_k_output], start=1
        ):
            prediction_rows.append(
                {
                    "paper_id": paper_id,
                    "rank": rank,
                    "expert_id": expert_id,
                    "expert_name": meta.get("expert_name", expert_id),
                    "score": f"{weighted_score:.6f}",
                    "raw_node_score": f"{score:.6f}",
                    "best_node_id": node_id,
                    "best_node_name": row.get("node_name", node_id),
                    "is_actual_member": "1" if expert_id in positives else "0",
                }
            )

    metrics = {
        "method": "hierec_embedding_node_assign_then_team_size_cut",
        "tasks": len(task_rows_by_paper),
        "avg_assigned_nodes": f"{mean(assigned_counts):.6f}",
        "avg_unique_assigned_experts": f"{mean(unique_counts):.6f}",
        "avg_selected_experts": f"{mean(selected_counts):.6f}",
        "macro_precision_at_team_size": f"{mean(macro_p):.6f}",
        "macro_recall_at_team_size": f"{mean(macro_r):.6f}",
        "micro_precision_at_team_size": f"{(micro_hits / micro_selected) if micro_selected else 0.0:.6f}",
        "micro_recall_at_team_size": f"{(micro_hits / micro_positives) if micro_positives else 0.0:.6f}",
    }

    with (out_dir / "metrics_summary.tsv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics), delimiter="\t")
        writer.writeheader()
        writer.writerow(metrics)

    with (out_dir / "node_assignments.tsv").open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "paper_id",
            "node_id",
            "node_name",
            "node_level",
            "node_importance",
            "subtree_skill_count",
            "expert_id",
            "expert_name",
            "score",
            "weighted_score",
            "is_actual_member",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(assignment_rows)

    with (out_dir / "predictions_team_size.tsv").open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "paper_id",
            "rank",
            "expert_id",
            "expert_name",
            "score",
            "raw_node_score",
            "best_node_id",
            "best_node_name",
            "is_actual_member",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(prediction_rows)

    print(f"tasks={len(task_rows_by_paper)}")
    print(f"macro_p={100 * mean(macro_p):.4f}%")
    print(f"macro_r={100 * mean(macro_r):.4f}%")
    print(f"out_dir={out_dir}")


if __name__ == "__main__":
    main()
