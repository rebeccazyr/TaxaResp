#!/usr/bin/env python3
"""Add skill coverage totals to direct_weighting_comparison.tsv."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from embedding_pipeline_utils import load_fos_map, load_tasks
from taxonomy_team_formation_experiment import (
    build_idf,
    build_index,
    rank_all_positive_positions,
    score_task,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Add skillcoverage_total_score to comparison table")
    p.add_argument("--comparison-tsv", default="output/taxonomy_team_formation_experiment/direct_weighting_comparison.tsv")
    p.add_argument("--tasks-csv", default="data_preprocess/teams_2020plus_with_skill_weights.csv")
    p.add_argument("--profile-dir", default="output/expert_profile_year_bins/all_2000_2019")
    p.add_argument("--fos-map", default="data/dblp/FieldsOfStudy.txt")
    p.add_argument("--direct-predictions", default="output/taxonomy_team_formation_experiment/predictions_topk.tsv")
    p.add_argument("--bfs-predictions", default="output/taxonomy_team_formation_experiment/bfs_unique_team_size_predictions.tsv")
    p.add_argument("--embedding-bfs-predictions", default="output/embedding_bfs_unique_assignment_no_label/predictions_team_size.tsv")
    p.add_argument("--embedding-direct-predictions", default="output/embedding_direct_log_sum_no_label/predictions_topk.tsv")
    p.add_argument("--out-tsv", default="")
    return p.parse_args()


def safe_float(v: object, default: float = 0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def read_profile_direct_nodes(profile_dir: Path) -> Tuple[Dict[str, set], Dict[str, Dict[str, float]]]:
    direct_nodes: Dict[str, set] = {}
    direct_weights: Dict[str, Dict[str, float]] = {}
    for idx, path in enumerate(sorted(profile_dir.glob("*_direct_fos_nodes.tsv")), start=1):
        if path.name.startswith("_"):
            continue
        expert_id = path.name.replace("_direct_fos_nodes.tsv", "")
        nodes = set()
        weights = {}
        with path.open("r", encoding="utf-8") as f:
            for row in csv.DictReader(f, delimiter="\t"):
                node_id = (row.get("fos_id") or "").strip()
                weight = safe_float(row.get("direct_weight_sum"), 0.0)
                if node_id and weight > 0:
                    nodes.add(node_id)
                    weights[node_id] = weight
        direct_nodes[expert_id] = nodes
        direct_weights[expert_id] = weights
        if idx % 2000 == 0:
            print(f"profile_progress {idx:,}", flush=True)
    return direct_nodes, direct_weights


def direct_vector(items: Sequence[Tuple[str, float, str]], transform: str) -> Dict[str, float]:
    vec: Dict[str, float] = defaultdict(float)
    for node_id, weight, _ in items:
        if transform == "log":
            vec[node_id] += math.log1p(weight)
        else:
            vec[node_id] += weight
    return dict(vec)


def build_expert_vectors(
    direct_weights: Dict[str, Dict[str, float]],
    transform: str,
) -> Dict[str, Dict[str, float]]:
    out = {}
    for expert_id, weights in direct_weights.items():
        if transform == "normalized":
            total = sum(weights.values())
            out[expert_id] = {
                node_id: (weight / total)
                for node_id, weight in weights.items()
                if total > 0 and weight > 0
            }
        elif transform == "log":
            out[expert_id] = {
                node_id: math.log1p(weight)
                for node_id, weight in weights.items()
                if weight > 0
            }
        else:
            out[expert_id] = {node_id: weight for node_id, weight in weights.items() if weight > 0}
    return out


def selected_from_prediction_file(path: Path, method: str, team_sizes: Dict[str, int]) -> Dict[str, List[str]]:
    selected: Dict[str, List[str]] = defaultdict(list)
    if not path.exists():
        return selected
    with path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if row.get("method") != method:
                continue
            paper_id = str(row.get("paper_id"))
            rank = int(float(row.get("rank") or 0))
            if rank <= team_sizes.get(paper_id, 0):
                selected[paper_id].append(str(row.get("expert_id")))
    return dict(selected)


def selected_direct_log_sum(
    tasks: List[dict],
    direct_weights: Dict[str, Dict[str, float]],
) -> Dict[str, List[str]]:
    expert_vectors = build_expert_vectors(direct_weights, transform="log")
    idf = build_idf(expert_vectors)
    index, norms = build_index(expert_vectors, idf)
    selected = {}
    for task in tasks:
        task_vec = direct_vector(task["direct"], transform="raw")
        ranked = score_task(task_vec, index, norms, idf, top_k=len(norms))
        selected[task["paper_id"]] = [
            expert_id for expert_id, _ in ranked[: max(1, int(task["team_size"]))]
        ]
    return selected


def skillcoverage_total(
    tasks: List[dict],
    selected_by_paper: Dict[str, List[str]],
    direct_nodes: Dict[str, set],
) -> Tuple[float, float]:
    total = 0.0
    normalized_scores = []
    for task in tasks:
        paper_id = str(task["paper_id"])
        selected = selected_by_paper.get(paper_id, [])
        covered = set()
        for expert_id in selected:
            covered.update(direct_nodes.get(expert_id, set()))
        task_total = sum(weight for _, weight, _ in task["direct"])
        score = sum(weight for node_id, weight, _ in task["direct"] if node_id in covered)
        total += score
        normalized_scores.append(score / task_total if task_total > 0 else 0.0)
    return total, sum(normalized_scores) / len(normalized_scores) if normalized_scores else 0.0


def main() -> None:
    args = parse_args()
    comparison_path = Path(args.comparison_tsv)
    out_path = Path(args.out_tsv) if args.out_tsv else comparison_path

    name_to_id, id_to_name, _ = load_fos_map(Path(args.fos_map))
    tasks = [t for t in load_tasks(Path(args.tasks_csv), name_to_id, id_to_name) if t["direct"]]
    team_sizes = {str(t["paper_id"]): max(1, int(t["team_size"])) for t in tasks}

    print("loading expert direct profiles", flush=True)
    direct_nodes, direct_weights = read_profile_direct_nodes(Path(args.profile_dir))

    selected_by_variant = {
        "direct_log_sum_with_idf_cosine": selected_direct_log_sum(tasks, direct_weights),
        "direct_normalized_sum_with_idf_cosine": selected_from_prediction_file(
            Path(args.direct_predictions), "direct", team_sizes
        ),
        "bfs_unique_assign_each_node_then_top_team_size_by_weighted_score": selected_from_prediction_file(
            Path(args.bfs_predictions), "bfs_unique_top_team_size", team_sizes
        ),
        "bfs_unique_assign_each_node_then_top_team_size_by_weighted_score_log_sum": selected_from_prediction_file(
            Path(args.bfs_predictions), "bfs_unique_top_team_size", team_sizes
        ),
        "embedding_bfs_unique_assign_each_node_then_top_team_size_by_weighted_score": selected_from_prediction_file(
            Path(args.embedding_bfs_predictions),
            "embedding_bfs_unique_assign_each_node_then_top_team_size_by_weighted_score",
            team_sizes,
        ),
        "embedding_bfs_unique_assign_each_node_then_top_team_size_by_weighted_score_log_sum": selected_from_prediction_file(
            Path(args.embedding_bfs_predictions),
            "embedding_bfs_unique_assign_each_node_then_top_team_size_by_weighted_score_log_sum",
            team_sizes,
        ),
        "embedding_direct_log_sum_with_idf_cosine_no_label": selected_from_prediction_file(
            Path(args.embedding_direct_predictions),
            "embedding_direct_log_sum_with_idf_cosine_no_label",
            team_sizes,
        ),
    }

    coverage = {}
    for variant, selected in selected_by_variant.items():
        total, mean_norm = skillcoverage_total(tasks, selected, direct_nodes)
        coverage[variant] = (total, mean_norm)
        print(f"{variant}\tskillcoverage_total={total:.6f}\tmean_norm={mean_norm:.6f}", flush=True)

    with comparison_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    for field in ("skillcoverage_total_score", "skillcoverage_mean_normalized_score"):
        if field not in fieldnames:
            fieldnames.append(field)
    for row in rows:
        total, mean_norm = coverage.get(row.get("variant"), ("", ""))
        row["skillcoverage_total_score"] = f"{total:.6f}" if total != "" else ""
        row["skillcoverage_mean_normalized_score"] = f"{mean_norm:.6f}" if mean_norm != "" else ""

    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote={out_path}")


if __name__ == "__main__":
    main()
