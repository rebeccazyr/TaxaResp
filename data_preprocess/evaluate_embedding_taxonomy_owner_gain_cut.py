#!/usr/bin/env python3
"""Evaluate taxonomy region cuts selected by owner-assignment gain.

This keeps the region-cut idea but replaces JSD edge scoring with the direct
gain in final region-owner assignment objective. At each step, the method cuts
the taxonomy edge whose removal yields the largest increase in unique owner
matching score over all current regions.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from embedding_pipeline_utils import load_child_to_parents, read_jsonl
from evaluate_embedding_taxonomy_region_cut import (
    VIRTUAL_ROOT,
    as_members,
    assign_region_owners,
    build_expert_node_index,
    choose_task_parent,
    connected_regions,
    load_task_embedding_table,
    mean,
    parse_subtree_log_sum,
    precompute_task_node_rankings,
    safe_float,
    score_regions,
)


METHOD = "embedding_taxonomy_owner_gain_region_cut"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate owner-gain taxonomy region cuts")
    p.add_argument("--task-nodes-jsonl", required=True)
    p.add_argument("--task-node-ids", required=True)
    p.add_argument("--task-node-embeddings", required=True)
    p.add_argument("--expert-node-ids", required=True)
    p.add_argument("--expert-node-embeddings", required=True)
    p.add_argument("--fos-children", default="data/dblp/13.FieldOfStudyChildren.nt")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--top-m", type=int, default=256)
    p.add_argument("--top-k-output", type=int, default=20)
    return p.parse_args()


def load_task_rows(path: Path) -> tuple[Dict[str, list], Dict[str, dict]]:
    task_rows_by_paper: Dict[str, list] = defaultdict(list)
    task_info: Dict[str, dict] = {}
    for row in read_jsonl(path):
        paper_id = str(row["paper_id"])
        node_id = str(row["node_id"])
        row = dict(row)
        row["task_node_id"] = f"{paper_id}::{node_id}"
        row["node_importance"] = safe_float(row.get("node_importance"), 0.0)
        row["node_log_sum"] = parse_subtree_log_sum(row.get("subtree_skills", ""))
        task_rows_by_paper[paper_id].append(row)
        task_info[paper_id] = {
            "team_size": int(row["team_size"]),
            "members": as_members(row.get("members")),
        }
    return task_rows_by_paper, task_info


def build_task_edges(
    rows: List[dict],
    child_to_parents: Dict[str, List[str]],
) -> List[Tuple[str, str, float]]:
    row_by_node = {str(row["node_id"]): row for row in rows}
    node_ids = set(row_by_node)
    edges = []
    for row in rows:
        child_id = str(row["node_id"])
        parent_id = choose_task_parent(row, node_ids, row_by_node, child_to_parents)
        edges.append((parent_id, child_id, 0.0))
    return edges


def assignment_objective(
    regions: List[List[str]],
    row_by_node: Dict[str, dict],
    rankings: Dict[str, list],
) -> tuple[float, list]:
    region_scores, _ = score_regions(regions, row_by_node, rankings, "importance")
    selected = assign_region_owners(region_scores, allow_repeat=False)
    return sum(score for _, _, score in selected), selected


def greedy_owner_gain_cuts(
    rows: List[dict],
    edges: List[Tuple[str, str, float]],
    row_by_node: Dict[str, dict],
    rankings: Dict[str, list],
    target_regions: int,
) -> List[Tuple[str, str, float]]:
    n_cuts = max(0, min(target_regions - 1, len(edges)))
    cuts: List[Tuple[str, str, float]] = []
    cut_pairs = set()

    for _ in range(n_cuts):
        current_regions = connected_regions(rows, edges, cuts)
        current_obj, _ = assignment_objective(current_regions, row_by_node, rankings)
        best = None

        for parent, child, _score in edges:
            if (parent, child) in cut_pairs:
                continue
            candidate_cut = (parent, child, 0.0)
            candidate_regions = connected_regions(rows, edges, cuts + [candidate_cut])
            if len(candidate_regions) <= len(current_regions):
                continue
            candidate_obj, _ = assignment_objective(
                candidate_regions, row_by_node, rankings
            )
            gain = candidate_obj - current_obj
            key = (gain, candidate_obj, parent, child)
            if best is None or key > best[0]:
                best = (key, (parent, child, gain))

        if best is None:
            break
        cut = best[1]
        cuts.append(cut)
        cut_pairs.add((cut[0], cut[1]))

    return cuts


def summarize(task_results: List[dict]) -> dict:
    micro_hits = sum(r["hits"] for r in task_results)
    micro_selected = sum(r["selected"] for r in task_results)
    micro_positives = sum(r["positives"] for r in task_results)
    return {
        "method": METHOD,
        "tasks": len(task_results),
        "mean_precision_at_team_size": f"{mean([r['precision'] for r in task_results]):.12f}",
        "mean_recall_at_team_size": f"{mean([r['recall'] for r in task_results]):.12f}",
        "percent_precision_at_team_size": f"{100 * mean([r['precision'] for r in task_results]):.6f}",
        "percent_recall_at_team_size": f"{100 * mean([r['recall'] for r in task_results]):.6f}",
        "micro_precision_at_team_size": f"{(micro_hits / micro_selected) if micro_selected else 0.0:.12f}",
        "micro_recall_at_team_size": f"{(micro_hits / micro_positives) if micro_positives else 0.0:.12f}",
        "avg_task_nodes": f"{mean([r['task_nodes'] for r in task_results]):.6f}",
        "avg_cut_edges": f"{mean([r['cut_edges'] for r in task_results]):.6f}",
        "avg_regions": f"{mean([r['regions'] for r in task_results]):.6f}",
        "avg_selected_experts": f"{mean([r['selected'] for r in task_results]):.6f}",
        "avg_cut_gain": f"{mean([r['avg_cut_gain'] for r in task_results]):.12f}",
        "duplicate_expert_assignments": f"{sum(r['duplicates'] for r in task_results)}",
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

    print("loading task taxonomy rows", flush=True)
    task_rows_by_paper, task_info = load_task_rows(Path(args.task_nodes_jsonl))
    child_to_parents = load_child_to_parents(Path(args.fos_children))

    print("precomputing same-node top-M embedding rankings", flush=True)
    task_node_rankings = precompute_task_node_rankings(
        task_rows_by_paper,
        task_embeddings,
        experts_by_node,
        expert_ids,
        expert_arr,
        args.top_m,
    )

    task_results = []
    edge_rows = []
    region_rows = []
    prediction_rows = []

    for task_idx, (paper_id, rows) in enumerate(sorted(task_rows_by_paper.items()), start=1):
        if task_idx % 25 == 0:
            print(f"eval_progress tasks={task_idx:,}/{len(task_rows_by_paper):,}", flush=True)
        positives = task_info[paper_id]["members"]
        team_size = max(1, task_info[paper_id]["team_size"])
        row_by_node = {str(row["node_id"]): row for row in rows}
        edges = build_task_edges(rows, child_to_parents)
        cuts = greedy_owner_gain_cuts(
            rows, edges, row_by_node, task_node_rankings, team_size
        )
        cut_pairs = {(u, v) for u, v, _ in cuts}
        cut_gain_by_pair = {(u, v): gain for u, v, gain in cuts}

        for parent, child, _ in edges:
            edge_rows.append(
                {
                    "paper_id": paper_id,
                    "parent_node_id": parent,
                    "parent_node_name": "Task"
                    if parent == VIRTUAL_ROOT
                    else row_by_node[parent].get("node_name", parent),
                    "child_node_id": child,
                    "child_node_name": row_by_node[child].get("node_name", child),
                    "owner_gain": f"{cut_gain_by_pair.get((parent, child), 0.0):.12f}",
                    "is_cut": "1" if (parent, child) in cut_pairs else "0",
                }
            )

        regions = connected_regions(rows, edges, cuts)
        region_scores, best_node = score_regions(
            regions, row_by_node, task_node_rankings, "importance"
        )
        selected = assign_region_owners(region_scores, allow_repeat=False)
        selected_ids = [expert_id for _, expert_id, _ in selected]
        hits = len(positives.intersection(selected_ids))
        duplicates = len(selected_ids) - len(set(selected_ids))
        task_results.append(
            {
                "hits": hits,
                "selected": len(selected_ids),
                "positives": len(positives),
                "precision": hits / len(selected_ids) if selected_ids else 0.0,
                "recall": hits / len(positives) if positives else 0.0,
                "task_nodes": len(rows),
                "cut_edges": len(cuts),
                "regions": len(regions),
                "avg_cut_gain": mean([gain for _, _, gain in cuts]),
                "duplicates": duplicates,
            }
        )

        owner_by_region = {region_idx: (expert_id, score) for region_idx, expert_id, score in selected}
        for region_idx, region in enumerate(regions):
            expert_id, score = owner_by_region.get(region_idx, ("", 0.0))
            best_node_id, best_node_score = best_node.get((region_idx, expert_id), ("", 0.0))
            region_rows.append(
                {
                    "paper_id": paper_id,
                    "region_id": region_idx + 1,
                    "node_ids": "|".join(region),
                    "node_names": "|".join(row_by_node[n].get("node_name", n) for n in region),
                    "node_count": len(region),
                    "owner_expert_id": expert_id,
                    "owner_expert_name": expert_names.get(expert_id, expert_id) if expert_id else "",
                    "owner_score": f"{score:.6f}",
                    "best_owner_node_id": best_node_id,
                    "best_owner_node_name": row_by_node[best_node_id].get("node_name", best_node_id) if best_node_id else "",
                    "best_owner_node_score": f"{best_node_score:.6f}",
                    "is_actual_member": "1" if expert_id in positives else "0",
                }
            )

        ranked_selected = sorted(selected, key=lambda item: item[2], reverse=True)
        for rank, (region_idx, expert_id, score) in enumerate(
            ranked_selected[: args.top_k_output], start=1
        ):
            region = regions[region_idx]
            best_node_id, best_node_score = best_node.get((region_idx, expert_id), ("", 0.0))
            prediction_rows.append(
                {
                    "method": METHOD,
                    "paper_id": paper_id,
                    "rank": rank,
                    "region_id": region_idx + 1,
                    "expert_id": expert_id,
                    "expert_name": expert_names.get(expert_id, expert_id),
                    "score": f"{score:.6f}",
                    "best_node_id": best_node_id,
                    "best_node_name": row_by_node[best_node_id].get("node_name", best_node_id) if best_node_id else "",
                    "best_node_score": f"{best_node_score:.6f}",
                    "region_node_ids": "|".join(region),
                    "is_actual_member": "1" if expert_id in positives else "0",
                }
            )

    metric_row = summarize(task_results)
    with (out_dir / "metrics_summary.tsv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(metric_row), delimiter="\t")
        writer.writeheader()
        writer.writerow(metric_row)

    with (out_dir / "edge_owner_gains.tsv").open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "paper_id",
            "parent_node_id",
            "parent_node_name",
            "child_node_id",
            "child_node_name",
            "owner_gain",
            "is_cut",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(edge_rows)

    with (out_dir / "regions.tsv").open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "paper_id",
            "region_id",
            "node_ids",
            "node_names",
            "node_count",
            "owner_expert_id",
            "owner_expert_name",
            "owner_score",
            "best_owner_node_id",
            "best_owner_node_name",
            "best_owner_node_score",
            "is_actual_member",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(region_rows)

    with (out_dir / "predictions_team_size.tsv").open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "method",
            "paper_id",
            "rank",
            "region_id",
            "expert_id",
            "expert_name",
            "score",
            "best_node_id",
            "best_node_name",
            "best_node_score",
            "region_node_ids",
            "is_actual_member",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(prediction_rows)

    print(f"tasks={len(task_results)}")
    print(
        f"{METHOD} p={float(metric_row['percent_precision_at_team_size']):.4f}% "
        f"r={float(metric_row['percent_recall_at_team_size']):.4f}%"
    )
    print(f"out_dir={out_dir}")


if __name__ == "__main__":
    main()
