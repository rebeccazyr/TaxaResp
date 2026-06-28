#!/usr/bin/env python3
"""Grid-search tunable parameters for taxonomy region-cut embedding evaluation."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

from embedding_pipeline_utils import load_child_to_parents, read_jsonl
from evaluate_embedding_taxonomy_region_cut import (
    METHOD,
    REGION_WEIGHT_CHOICES,
    as_members,
    assign_region_owners,
    build_expert_node_index,
    build_task_tree,
    connected_regions,
    load_task_embedding_table,
    mean,
    parse_subtree_log_sum,
    precompute_task_node_rankings,
    responsibility_overlap,
    safe_float,
    score_distribution,
    score_regions,
    selected_cut_edges,
)


def parse_int_grid(text: str) -> List[int]:
    values = sorted({int(x.strip()) for x in text.split(",") if x.strip()})
    if not values or any(v <= 0 for v in values):
        raise ValueError(f"invalid int grid: {text}")
    return values


def parse_float_grid(text: str) -> List[float]:
    values = sorted({float(x.strip()) for x in text.split(",") if x.strip()})
    if not values or any(v <= 0 for v in values):
        raise ValueError(f"invalid float grid: {text}")
    return values


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Tune taxonomy region-cut embedding parameters on a labeled task set"
    )
    p.add_argument("--task-nodes-jsonl", required=True)
    p.add_argument("--task-node-ids", required=True)
    p.add_argument("--task-node-embeddings", required=True)
    p.add_argument("--expert-node-ids", required=True)
    p.add_argument("--expert-node-embeddings", required=True)
    p.add_argument("--fos-children", default="data/dblp/13.FieldOfStudyChildren.nt")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--top-m-grid", default="8,16,32,64,128,256")
    p.add_argument(
        "--temperature-grid",
        default="0.01,0.02,0.03,0.05,0.08,0.1,0.15,0.2,0.3,0.5",
    )
    p.add_argument(
        "--repeat-grid",
        default="unique,repeat",
        help="Comma-separated values from: unique, repeat.",
    )
    p.add_argument(
        "--region-weight-grid",
        default="importance",
        help="Comma-separated values from: importance, log_sum.",
    )
    p.add_argument(
        "--objective",
        choices=["mean_recall_at_team_size", "mean_precision_at_team_size"],
        default="mean_recall_at_team_size",
    )
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


def sliced_rankings(
    rankings: Dict[str, list],
    top_m: int,
) -> Dict[str, list]:
    return {key: value[:top_m] for key, value in rankings.items()}


def evaluate_combo(
    task_rows_by_paper: Dict[str, list],
    task_info: Dict[str, dict],
    child_to_parents: Dict[str, List[str]],
    rankings: Dict[str, list],
    top_m: int,
    temperature: float,
    allow_repeat: bool,
    region_weight: str,
) -> dict:
    use_rankings = sliced_rankings(rankings, top_m)
    distributions = {
        task_node_id: score_distribution(ranking, temperature)
        for task_node_id, ranking in use_rankings.items()
    }

    task_results = []
    for paper_id, rows in task_rows_by_paper.items():
        positives = task_info[paper_id]["members"]
        team_size = max(1, task_info[paper_id]["team_size"])
        row_by_node = {str(row["node_id"]): row for row in rows}
        edges, _ = build_task_tree(rows, child_to_parents, distributions)
        cuts = selected_cut_edges(edges, team_size)
        regions = connected_regions(rows, edges, cuts)
        region_scores, _ = score_regions(
            regions, row_by_node, use_rankings, region_weight
        )
        selected = assign_region_owners(region_scores, allow_repeat)

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
                "avg_cut_boundary_score": mean([boundary for _, _, boundary in cuts]),
                "team_responsibility_overlap": responsibility_overlap(
                    selected, rows, use_rankings, region_weight
                ),
                "duplicates": duplicates,
            }
        )

    micro_hits = sum(r["hits"] for r in task_results)
    micro_selected = sum(r["selected"] for r in task_results)
    micro_positives = sum(r["positives"] for r in task_results)
    return {
        "method": METHOD,
        "top_m": top_m,
        "temperature": f"{temperature:.6g}",
        "region_weight": region_weight,
        "allow_repeat_experts": "1" if allow_repeat else "0",
        "tasks": len(task_results),
        "mean_precision_at_team_size": mean([r["precision"] for r in task_results]),
        "mean_recall_at_team_size": mean([r["recall"] for r in task_results]),
        "micro_precision_at_team_size": (micro_hits / micro_selected)
        if micro_selected
        else 0.0,
        "micro_recall_at_team_size": (micro_hits / micro_positives)
        if micro_positives
        else 0.0,
        "avg_task_nodes": mean([r["task_nodes"] for r in task_results]),
        "avg_cut_edges": mean([r["cut_edges"] for r in task_results]),
        "avg_regions": mean([r["regions"] for r in task_results]),
        "avg_selected_experts": mean([r["selected"] for r in task_results]),
        "avg_cut_boundary_score": mean(
            [r["avg_cut_boundary_score"] for r in task_results]
        ),
        "avg_team_responsibility_overlap": mean(
            [r["team_responsibility_overlap"] for r in task_results]
        ),
        "duplicate_expert_assignments": sum(r["duplicates"] for r in task_results),
    }


def sortable_metric(row: dict, objective: str) -> tuple:
    return (
        row[objective],
        row["mean_precision_at_team_size"],
        row["micro_recall_at_team_size"],
        -row["avg_team_responsibility_overlap"],
        -row["duplicate_expert_assignments"],
    )


def format_row(row: dict) -> dict:
    out = dict(row)
    for key in (
        "mean_precision_at_team_size",
        "mean_recall_at_team_size",
        "micro_precision_at_team_size",
        "micro_recall_at_team_size",
        "avg_task_nodes",
        "avg_cut_edges",
        "avg_regions",
        "avg_selected_experts",
        "avg_cut_boundary_score",
        "avg_team_responsibility_overlap",
    ):
        out[key] = f"{float(out[key]):.12f}"
    out["percent_precision_at_team_size"] = f"{100 * float(row['mean_precision_at_team_size']):.6f}"
    out["percent_recall_at_team_size"] = f"{100 * float(row['mean_recall_at_team_size']):.6f}"
    return out


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    top_m_grid = parse_int_grid(args.top_m_grid)
    temperature_grid = parse_float_grid(args.temperature_grid)
    repeat_tokens = {token.strip().lower() for token in args.repeat_grid.split(",") if token.strip()}
    bad_tokens = repeat_tokens - {"unique", "repeat"}
    if bad_tokens:
        raise ValueError(f"invalid repeat-grid values: {sorted(bad_tokens)}")
    repeat_grid = []
    if "unique" in repeat_tokens:
        repeat_grid.append(False)
    if "repeat" in repeat_tokens:
        repeat_grid.append(True)
    if not repeat_grid:
        raise ValueError("repeat-grid must include unique, repeat, or both")

    region_weight_grid = [
        token.strip().lower()
        for token in args.region_weight_grid.split(",")
        if token.strip()
    ]
    bad_weights = set(region_weight_grid) - set(REGION_WEIGHT_CHOICES)
    if bad_weights:
        raise ValueError(f"invalid region-weight-grid values: {sorted(bad_weights)}")
    if not region_weight_grid:
        raise ValueError("region-weight-grid must include at least one value")

    print("loading task embeddings", flush=True)
    task_embeddings = load_task_embedding_table(
        Path(args.task_node_ids), Path(args.task_node_embeddings)
    )

    print("loading expert ids/index", flush=True)
    experts_by_node, expert_ids, _, expert_row_count = build_expert_node_index(
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

    max_top_m = max(top_m_grid)
    print(f"precomputing same-node top-{max_top_m} embedding rankings", flush=True)
    rankings = precompute_task_node_rankings(
        task_rows_by_paper,
        task_embeddings,
        experts_by_node,
        expert_ids,
        expert_arr,
        max_top_m,
    )

    rows = []
    total = (
        len(top_m_grid)
        * len(temperature_grid)
        * len(repeat_grid)
        * len(region_weight_grid)
    )
    done = 0
    for top_m in top_m_grid:
        for temperature in temperature_grid:
            for region_weight in region_weight_grid:
                for allow_repeat in repeat_grid:
                    done += 1
                    print(
                        "grid_progress "
                        f"{done}/{total} top_m={top_m} temperature={temperature} "
                        f"region_weight={region_weight} "
                        f"allow_repeat={int(allow_repeat)}",
                        flush=True,
                    )
                    rows.append(
                        evaluate_combo(
                            task_rows_by_paper,
                            task_info,
                            child_to_parents,
                            rankings,
                            top_m,
                            temperature,
                            allow_repeat,
                            region_weight,
                        )
                    )

    rows.sort(key=lambda row: sortable_metric(row, args.objective), reverse=True)
    formatted = [format_row(row) for row in rows]
    fieldnames = list(formatted[0])
    with (out_dir / "grid_results.tsv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(formatted)

    with (out_dir / "best_params.tsv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerow(formatted[0])

    best = formatted[0]
    print(
        "best "
        f"top_m={best['top_m']} temperature={best['temperature']} "
        f"region_weight={best['region_weight']} "
        f"allow_repeat={best['allow_repeat_experts']} "
        f"p={best['percent_precision_at_team_size']}% "
        f"r={best['percent_recall_at_team_size']}% "
        f"overlap={best['avg_team_responsibility_overlap']}",
        flush=True,
    )
    print(f"out_dir={out_dir}", flush=True)


if __name__ == "__main__":
    main()
