#!/usr/bin/env python3
"""Compute missing selected-method coverage and team-structure metrics.

The selected methods are the presentation/table methods:

- Embedding BFS
- responsibility cut
- expert distribution cut
- Seq2seq
- Random mean 5

This script complements the existing exact/soft matching outputs by computing
task coverage for all selected methods, weighted SpecScore diversity, and
taxonomy-structure compactness under specialty/authority node assignment.
"""

from __future__ import annotations

import csv
import json
import pickle
import random
import re
import sys
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "data_preprocess"))

from add_skillcoverage_to_comparison import read_profile_direct_nodes, safe_float  # noqa: E402
from compute_weighted_specscore import (  # noqa: E402
    load_needed_expert_rows,
    load_task_rows,
    load_id_to_row,
    specscore_for_team,
    summarize as summarize_specscore,
)
from embedding_pipeline_utils import load_child_to_parents, load_fos_map, load_tasks  # noqa: E402


METHODS = [
    "Embedding BFS",
    "职责切割",
    "Expert 分布切割",
    "Seq2seq",
    "Random mean 5",
]


def mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def write_tsv(path: Path, rows: List[dict], fieldnames: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_task_order(path: Path) -> Tuple[List[str], Dict[str, int], Dict[str, List[str]]]:
    order: List[str] = []
    team_size_by_paper: Dict[str, int] = {}
    members_by_paper: Dict[str, List[str]] = {}
    seen = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            paper_id = str(row["paper_id"])
            if paper_id in seen:
                continue
            seen.add(paper_id)
            order.append(paper_id)
            members = row.get("members") or []
            members_by_paper[paper_id] = [str(x) for x in members]
            team_size_by_paper[paper_id] = int(row.get("team_size") or len(members) or 1)
    return order, team_size_by_paper, members_by_paper


def selected_from_prediction_file(path: Path, method: str, team_sizes: Dict[str, int]) -> Dict[str, List[str]]:
    selected: Dict[str, List[str]] = defaultdict(list)
    with path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if row.get("method") != method:
                continue
            paper_id = str(row.get("paper_id") or "")
            expert_id = str(row.get("expert_id") or "")
            if not paper_id or not expert_id:
                continue
            rank = int(float(row.get("rank") or 0))
            if rank <= team_sizes.get(paper_id, 0):
                selected[paper_id].append(expert_id)
    return {paper_id: dedupe(experts) for paper_id, experts in selected.items()}


def dedupe(values: Iterable[str]) -> List[str]:
    seen = set()
    out = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            out.append(value)
    return out


def load_seq2seq_teams(path: Path, indexes_pkl: Path, paper_order: Sequence[str]) -> Dict[str, List[str]]:
    with indexes_pkl.open("rb") as f:
        indexes = pickle.load(f)
    i2c = indexes["i2c"]

    teams: Dict[str, List[str]] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        for row_idx, row in enumerate(reader):
            if row_idx >= len(paper_order):
                break
            selected = []
            for raw_idx in re.findall(r"m(\d+)", row[0] if row else ""):
                idname = i2c.get(int(raw_idx))
                if idname:
                    selected.append(str(idname).split("_", 1)[0])
            teams[paper_order[row_idx]] = dedupe(selected)
    return teams


def add_random_mean5(
    teams_by_method: Dict[str, Dict[str, List[str]]],
    paper_order: Sequence[str],
    team_size_by_paper: Dict[str, int],
    members_by_paper: Dict[str, List[str]],
    seeds: Sequence[int],
) -> Tuple[Dict[str, Dict[str, List[str]]], Dict[str, List[Dict[str, List[str]]]]]:
    pool = set()
    for members in members_by_paper.values():
        pool.update(members)
    for teams in teams_by_method.values():
        for experts in teams.values():
            pool.update(experts)
    pool_list = sorted(pool)

    random_runs: Dict[str, List[Dict[str, List[str]]]] = {"Random mean 5": []}
    for seed in seeds:
        rng = random.Random(seed)
        teams: Dict[str, List[str]] = {}
        for paper_id in paper_order:
            team_size = max(1, team_size_by_paper[paper_id])
            if len(pool_list) <= team_size:
                teams[paper_id] = list(pool_list)
            else:
                teams[paper_id] = rng.sample(pool_list, team_size)
        random_runs["Random mean 5"].append(teams)
    return teams_by_method, random_runs


def load_selected_teams() -> Tuple[
    List[str],
    Dict[str, int],
    Dict[str, List[str]],
    Dict[str, Dict[str, List[str]]],
    Dict[str, List[Dict[str, List[str]]]],
]:
    task_nodes_jsonl = ROOT / "output/hierec_embedding_server_inputs/task_nodes.jsonl"
    paper_order, team_sizes, members_by_paper = read_task_order(task_nodes_jsonl)
    teams_by_method = {
        "Embedding BFS": selected_from_prediction_file(
            ROOT / "output/embedding_bfs_unique_assignment_no_label/predictions_team_size.tsv",
            "embedding_bfs_unique_assign_each_node_then_top_team_size_by_weighted_score",
            team_sizes,
        ),
        "职责切割": selected_from_prediction_file(
            ROOT / "output/embedding_taxonomy_owner_gain_cut_topm256_no_label/predictions_team_size.tsv",
            "embedding_taxonomy_owner_gain_region_cut",
            team_sizes,
        ),
        "Expert 分布切割": selected_from_prediction_file(
            ROOT / "output/embedding_taxonomy_region_cut_jsd_topm256_temp015_no_label/predictions_team_size.tsv",
            "embedding_taxonomy_region_cut_jsd_topm_owner_matching",
            team_sizes,
        ),
        "Seq2seq": load_seq2seq_teams(
            ROOT / "output/test.fold0.epoch15527.pred.csv",
            ROOT / "output/indexes.pkl",
            paper_order,
        ),
    }
    teams_by_method, random_runs = add_random_mean5(
        teams_by_method,
        paper_order,
        team_sizes,
        members_by_paper,
        seeds=[13, 14, 15, 16, 17],
    )
    return paper_order, team_sizes, members_by_paper, teams_by_method, random_runs


def build_best_weight_by_node(direct_weights: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    best: Dict[str, float] = defaultdict(float)
    for weights in direct_weights.values():
        for node_id, weight in weights.items():
            if weight > best[node_id]:
                best[node_id] = weight
    return dict(best)


def expert_total_weights(direct_weights: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    return {expert_id: sum(weights.values()) for expert_id, weights in direct_weights.items()}


def build_specialty_weight_by_expert_node(
    direct_weights: Dict[str, Dict[str, float]],
) -> Dict[str, Dict[str, float]]:
    specialty_weights: Dict[str, Dict[str, float]] = {}
    for expert_id, weights in direct_weights.items():
        total = sum(weights.values())
        specialty_weights[expert_id] = {
            node_id: weight / total
            for node_id, weight in weights.items()
            if total > 0 and weight > 0
        }
    return specialty_weights


def load_expert_node_authority(
    path: Path,
    needed_experts: set[str],
    needed_nodes: set[str],
) -> Tuple[Dict[Tuple[str, str], float], Dict[str, str]]:
    scores: Dict[Tuple[str, str], float] = {}
    names: Dict[str, str] = {}
    with path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            expert_id = str(row.get("expert_id") or "")
            if expert_id not in needed_experts:
                continue
            node_id = str(row.get("node_id") or "")
            if node_id not in needed_nodes:
                continue
            scores[(expert_id, node_id)] = safe_float(row.get("subtree_weight_sum"), 0.0)
            names.setdefault(expert_id, row.get("expert_name") or expert_id)
    return scores, names


def make_ancestor_paths(child_to_parents: Dict[str, List[str]]):
    cache: Dict[str, Dict[str, Tuple[str, ...]]] = {}

    def paths(node_id: str) -> Dict[str, Tuple[str, ...]]:
        if node_id in cache:
            return cache[node_id]
        best: Dict[str, Tuple[str, ...]] = {node_id: (node_id,)}
        q = deque([(node_id, (node_id,))])
        while q:
            child, child_path = q.popleft()
            parents = child_to_parents.get(child, [])
            if not parents and child != "ROOT_FOS":
                parents = ["ROOT_FOS"]
            for parent in parents:
                parent_path = (parent,) + child_path
                old = best.get(parent)
                if old is not None and len(old) <= len(parent_path):
                    continue
                best[parent] = parent_path
                q.append((parent, parent_path))
        cache[node_id] = best
        return best

    return paths


def compact_closure_size(assigned_nodes: Sequence[str], ancestor_paths) -> int:
    unique_nodes = sorted(set(assigned_nodes))
    if not unique_nodes:
        return 0
    if len(unique_nodes) == 1:
        return 1

    paths_by_node = {node_id: ancestor_paths(node_id) for node_id in unique_nodes}
    common = set.intersection(*(set(paths) for paths in paths_by_node.values()))
    if not common:
        return len(unique_nodes)

    best_size = None
    for ancestor in common:
        closure = set()
        for node_id in unique_nodes:
            closure.update(paths_by_node[node_id][ancestor])
        size = len(closure)
        if best_size is None or size < best_size:
            best_size = size
    return int(best_size or len(unique_nodes))


def assign_nodes_by_score(
    task_rows: Sequence[dict],
    team: Sequence[str],
    score_by_expert_node: Dict[Tuple[str, str], float],
) -> List[dict]:
    assignments = []
    team = list(team)
    if not team:
        return assignments
    for row in task_rows:
        node_id = str(row["node_id"])
        best_expert = min(team)
        best_score = -1.0
        for expert_id in team:
            score = score_by_expert_node.get((expert_id, node_id), 0.0)
            if score > best_score or (score == best_score and expert_id < best_expert):
                best_expert = expert_id
                best_score = score
        assignments.append(
            {
                "paper_id": str(row["paper_id"]),
                "node_id": node_id,
                "node_name": row.get("node_name", node_id),
                "assigned_expert_id": best_expert,
                "assignment_score": best_score,
            }
        )
    return assignments


def load_native_embedding_bfs_assignments(path: Path, method: str) -> Dict[str, List[dict]]:
    by_paper: Dict[str, List[dict]] = defaultdict(list)
    with path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            node_id = str(row.get("node_id") or "")
            expert_id = str(row.get("expert_id") or "")
            paper_id = str(row.get("paper_id") or "")
            if row.get("method") != method or not paper_id or not node_id or not expert_id:
                continue
            by_paper[paper_id].append(
                {
                    "paper_id": paper_id,
                    "node_id": node_id,
                    "node_name": row.get("node_name") or node_id,
                    "assigned_expert_id": expert_id,
                    "assignment_score": safe_float(row.get("weighted_score"), 0.0),
                }
            )
    return dict(by_paper)


def load_native_region_assignments(path: Path) -> Dict[str, List[dict]]:
    by_paper: Dict[str, List[dict]] = defaultdict(list)
    with path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            paper_id = str(row.get("paper_id") or "")
            expert_id = str(row.get("owner_expert_id") or "")
            if not paper_id or not expert_id:
                continue
            node_ids = [x for x in str(row.get("node_ids") or "").split("|") if x]
            node_names = str(row.get("node_names") or "").split("|")
            for idx, node_id in enumerate(node_ids):
                by_paper[paper_id].append(
                    {
                        "paper_id": paper_id,
                        "node_id": node_id,
                        "node_name": node_names[idx] if idx < len(node_names) and node_names[idx] else node_id,
                        "assigned_expert_id": expert_id,
                        "assignment_score": safe_float(row.get("owner_score"), 0.0),
                        "region_id": row.get("region_id", ""),
                    }
                )
    return dict(by_paper)


def compactness_for_task(
    paper_id: str,
    team: Sequence[str],
    task_rows: Sequence[dict],
    score_by_expert_node: Dict[Tuple[str, str], float],
    ancestor_paths,
) -> Tuple[dict, List[dict], List[dict]]:
    team = dedupe(team)
    assignments = assign_nodes_by_score(task_rows, team, score_by_expert_node)
    assigned_by_expert: Dict[str, List[str]] = defaultdict(list)
    for row in assignments:
        assigned_by_expert[row["assigned_expert_id"]].append(row["node_id"])

    expert_rows = []
    compactness_values = []
    for expert_id in team:
        assigned_nodes = assigned_by_expert.get(expert_id, [])
        assigned_count = len(set(assigned_nodes))
        closure_size = compact_closure_size(assigned_nodes, ancestor_paths)
        extra_connector_nodes = max(0, closure_size - assigned_count)
        compactness = (assigned_count / closure_size) if closure_size > 0 else 0.0
        compactness_values.append(compactness)
        expert_rows.append(
            {
                "paper_id": paper_id,
                "expert_id": expert_id,
                "assigned_node_count": assigned_count,
                "tree_closure_node_count": closure_size,
                "extra_connector_node_count": extra_connector_nodes,
                "compactness": compactness,
                "assigned_node_ids": "|".join(sorted(set(assigned_nodes))),
            }
        )

    task_result = {
        "paper_id": paper_id,
        "team_size": len(team),
        "task_node_count": len(task_rows),
        "assigned_node_count": len(assignments),
        "team_compactness": mean(compactness_values),
    }
    return task_result, assignments, expert_rows


def compactness_for_native_assignments(
    paper_id: str,
    team: Sequence[str],
    task_rows: Sequence[dict],
    assignments: Sequence[dict],
    ancestor_paths,
) -> Tuple[dict, List[dict], List[dict]]:
    team = dedupe(team)
    task_node_ids = {str(row["node_id"]) for row in task_rows}
    filtered_assignments = [
        dict(row)
        for row in assignments
        if str(row.get("node_id") or "") in task_node_ids
        and str(row.get("assigned_expert_id") or "") in set(team)
    ]
    assigned_by_expert: Dict[str, List[str]] = defaultdict(list)
    for row in filtered_assignments:
        assigned_by_expert[str(row["assigned_expert_id"])].append(str(row["node_id"]))

    expert_rows = []
    compactness_values = []
    for expert_id in team:
        assigned_nodes = assigned_by_expert.get(expert_id, [])
        assigned_count = len(set(assigned_nodes))
        closure_size = compact_closure_size(assigned_nodes, ancestor_paths)
        extra_connector_nodes = max(0, closure_size - assigned_count)
        compactness = (assigned_count / closure_size) if closure_size > 0 else 0.0
        compactness_values.append(compactness)
        expert_rows.append(
            {
                "paper_id": paper_id,
                "expert_id": expert_id,
                "assigned_node_count": assigned_count,
                "tree_closure_node_count": closure_size,
                "extra_connector_node_count": extra_connector_nodes,
                "compactness": compactness,
                "assigned_node_ids": "|".join(sorted(set(assigned_nodes))),
            }
        )

    task_result = {
        "paper_id": paper_id,
        "team_size": len(team),
        "task_node_count": len(task_rows),
        "assigned_node_count": len(filtered_assignments),
        "team_compactness": mean(compactness_values),
    }
    return task_result, filtered_assignments, expert_rows


def summarize_compactness(label: str, rows: List[dict]) -> dict:
    return {
        "label": label,
        "tasks": len(rows),
        "mean_team_compactness": f"{mean([float(r['team_compactness']) for r in rows]):.12f}",
        "avg_team_size": f"{mean([int(r['team_size']) for r in rows]):.6f}",
        "avg_task_nodes": f"{mean([int(r['task_node_count']) for r in rows]):.6f}",
        "avg_assigned_nodes": f"{mean([int(r['assigned_node_count']) for r in rows]):.6f}",
    }


def coverage_for_teams(
    tasks: List[dict],
    selected_by_paper: Dict[str, List[str]],
    direct_nodes: Dict[str, set],
    direct_weights: Dict[str, Dict[str, float]],
    best_weight_by_node: Dict[str, float],
    specialty_weights: Dict[str, Dict[str, float]],
) -> dict:
    specialty_totals = []
    specialty_weighted_norms = []
    binary_counts = 0
    binary_norms = []
    authority_totals = []
    authority_weighted_norms = []
    authority_unweighted_norms = []

    for task in tasks:
        paper_id = str(task["paper_id"])
        selected = selected_by_paper.get(paper_id, [])
        task_nodes = [(str(node_id), float(weight)) for node_id, weight, _ in task["direct"]]
        covered = set()
        for expert_id in selected:
            covered.update(direct_nodes.get(expert_id, set()))

        task_total = sum(weight for _, weight in task_nodes)
        specialty_num = 0.0
        for node_id, task_weight in task_nodes:
            selected_best_specialty = max(
                (specialty_weights.get(expert_id, {}).get(node_id, 0.0) for expert_id in selected),
                default=0.0,
            )
            specialty_num += task_weight * selected_best_specialty
        specialty_totals.append(specialty_num)
        specialty_weighted_norms.append(specialty_num / task_total if task_total > 0 else 0.0)

        binary_score = sum(1 for node_id, _ in task_nodes if node_id in covered)
        binary_counts += binary_score
        binary_norms.append(binary_score / len(task_nodes) if task_nodes else 0.0)

        authority_num = 0.0
        authority_unweighted = []
        for node_id, task_weight in task_nodes:
            selected_best = max(
                (direct_weights.get(expert_id, {}).get(node_id, 0.0) for expert_id in selected),
                default=0.0,
            )
            global_best = best_weight_by_node.get(node_id, 0.0)
            ratio = selected_best / global_best if global_best > 0 else 0.0
            authority_num += task_weight * ratio
            authority_unweighted.append(ratio)
        authority_totals.append(authority_num)
        authority_weighted_norms.append(authority_num / task_total if task_total > 0 else 0.0)
        authority_unweighted_norms.append(mean(authority_unweighted))

    return {
        "tasks": len(tasks),
        "binary_coverage_total_count": binary_counts,
        "binary_coverage_mean_normalized": mean(binary_norms),
        "specialty_coverage_total_score": sum(specialty_totals),
        "specialty_coverage_mean_normalized": mean(specialty_weighted_norms),
        "authority_coverage_total_score": sum(authority_totals),
        "authority_coverage_mean_weighted_normalized": mean(authority_weighted_norms),
        "authority_coverage_mean_unweighted_normalized": mean(authority_unweighted_norms),
    }


def compute_task_coverage(
    teams_by_method: Dict[str, Dict[str, List[str]]],
    random_runs: Dict[str, List[Dict[str, List[str]]]],
    out_dir: Path,
) -> None:
    name_to_id, id_to_name, _ = load_fos_map(ROOT / "data/dblp/FieldsOfStudy.txt")
    tasks = [
        task
        for task in load_tasks(ROOT / "data_preprocess/teams_2020plus_with_skill_weights.csv", name_to_id, id_to_name)
        if task["direct"]
    ]

    print("loading expert direct profiles", flush=True)
    direct_nodes, direct_weights = read_profile_direct_nodes(
        ROOT / "output/expert_profile_year_bins/all_2000_2019"
    )
    best_weight_by_node = build_best_weight_by_node(direct_weights)
    specialty_weights = build_specialty_weight_by_expert_node(direct_weights)

    rows = []
    for method in METHODS:
        if method == "Random mean 5":
            per_seed = [
                coverage_for_teams(
                    tasks,
                    teams,
                    direct_nodes,
                    direct_weights,
                    best_weight_by_node,
                    specialty_weights,
                )
                for teams in random_runs[method]
            ]
            row = {"method": method, "random_runs": len(per_seed)}
            for field in per_seed[0]:
                if field == "tasks":
                    row[field] = per_seed[0][field]
                else:
                    row[field] = mean([float(r[field]) for r in per_seed])
        else:
            row = {"method": method, "random_runs": ""}
            row.update(
                coverage_for_teams(
                    tasks,
                    teams_by_method[method],
                    direct_nodes,
                    direct_weights,
                    best_weight_by_node,
                    specialty_weights,
                )
            )
        rows.append(format_float_row(row))

    fields = [
        "method",
        "tasks",
        "random_runs",
        "binary_coverage_total_count",
        "binary_coverage_mean_normalized",
        "specialty_coverage_total_score",
        "specialty_coverage_mean_normalized",
        "authority_coverage_total_score",
        "authority_coverage_mean_weighted_normalized",
        "authority_coverage_mean_unweighted_normalized",
    ]
    write_tsv(out_dir / "task_coverage_completed_selected_methods.tsv", rows, fields)
    write_task_coverage_table2(rows, out_dir)


def write_task_coverage_table2(rows: List[dict], out_dir: Path) -> None:
    metric_specs = [
        (
            "Binary Coverage",
            "binary_coverage_mean_normalized",
            "binary_coverage_total_count",
        ),
        (
            "Soft Coverage / Authority Score Coverage",
            "authority_coverage_mean_weighted_normalized",
            "authority_coverage_total_score",
        ),
        (
            "Soft Coverage / Specialty Score Coverage",
            "specialty_coverage_mean_normalized",
            "specialty_coverage_total_score",
        ),
    ]
    table_rows = []
    for metric_label, value_field, total_field in metric_specs:
        for row in rows:
            total_value = row[total_field]
            if metric_label != "Binary Coverage":
                total_value = f"{float(total_value):.6f}"
            table_rows.append(
                {
                    "category": "Task Coverage",
                    "metric": metric_label,
                    "method": row["method"],
                    "value": f"{float(row[value_field]):.6f}",
                    "total": total_value,
                    "tasks": row["tasks"],
                    "random_runs": row["random_runs"],
                }
            )
    write_tsv(
        out_dir / "table2_task_coverage_all_results.tsv",
        table_rows,
        ["category", "metric", "method", "value", "total", "tasks", "random_runs"],
    )


def format_float_row(row: dict) -> dict:
    out = {}
    for key, value in row.items():
        if isinstance(value, float):
            out[key] = f"{value:.12f}"
        else:
            out[key] = value
    return out


def aggregate_specscore_random(rows: List[dict]) -> dict:
    out = {"label": "Random mean 5", "random_runs": len(rows)}
    for key in rows[0]:
        if key == "label":
            continue
        values = [float(row[key]) for row in rows]
        if key in {"tasks", "tasks_with_valid_pairs"}:
            out[key] = int(round(mean(values)))
        elif key in {"valid_pairs", "possible_pairs", "negative_similarities"}:
            out[key] = f"{mean(values):.6f}"
        else:
            out[key] = f"{mean(values):.12f}"
    return out


def aggregate_compactness_random(rows: List[dict]) -> dict:
    out = {"label": "Random mean 5", "random_runs": len(rows)}
    for key in rows[0]:
        if key == "label":
            continue
        values = [float(row[key]) for row in rows]
        if key == "tasks":
            out[key] = int(round(mean(values)))
        else:
            out[key] = f"{mean(values):.12f}"
    return out


def compute_specscore_and_compactness(
    teams_by_method: Dict[str, Dict[str, List[str]]],
    random_runs: Dict[str, List[Dict[str, List[str]]]],
    out_dir: Path,
) -> None:
    print("loading task rows and embeddings", flush=True)
    rows_by_paper, _ = load_task_rows(ROOT / "output/hierec_embedding_server_inputs/task_nodes.jsonl")
    needed_nodes = {row["node_id"] for rows in rows_by_paper.values() for row in rows}
    task_id_to_row = load_id_to_row(ROOT / "output/all_expert_paper_embeddings/task_node_embedding_ids_strict_v2_no_label.tsv")
    task_arr = np.load(
        ROOT / "output/all_expert_paper_embeddings/task_node_embeddings_strict_v2_no_label.npy",
        mmap_mode="r",
    )

    all_team_sets = list(teams_by_method.values())
    all_team_sets.extend(teams for runs in random_runs.values() for teams in runs)
    needed_experts = {
        expert_id
        for teams in all_team_sets
        for team in teams.values()
        for expert_id in team
    }
    print(f"loading expert rows for {len(needed_experts):,} selected experts", flush=True)
    expert_index, _ = load_needed_expert_rows(
        ROOT / "output/all_expert_paper_embeddings/expert_node_embedding_ids_no_label.tsv",
        needed_experts,
    )
    expert_arr = np.load(
        ROOT / "output/all_expert_paper_embeddings/expert_node_embeddings_no_label.npy",
        mmap_mode="r",
    )
    authority_scores, expert_names = load_expert_node_authority(
        ROOT / "output/all_expert_paper_embeddings/expert_node_embedding_ids_no_label.tsv",
        needed_experts,
        needed_nodes,
    )
    _, direct_weights = read_profile_direct_nodes(ROOT / "output/expert_profile_year_bins/all_2000_2019")
    total_by_expert = expert_total_weights(direct_weights)
    specialty_scores = {
        key: (score / total_by_expert.get(key[0], 0.0)) if total_by_expert.get(key[0], 0.0) > 0 else 0.0
        for key, score in authority_scores.items()
    }
    ancestor_paths = make_ancestor_paths(
        load_child_to_parents(ROOT / "data/dblp/13.FieldOfStudyChildren.nt")
    )
    native_assignments_by_method = {
        "职责切割": load_native_region_assignments(
            ROOT / "output/embedding_taxonomy_owner_gain_cut_topm256_no_label/regions.tsv"
        ),
        "Expert 分布切割": load_native_region_assignments(
            ROOT / "output/embedding_taxonomy_region_cut_jsd_topm256_temp015_no_label/regions.tsv"
        ),
    }

    summary_rows = []
    detail_fields = [
        "paper_id",
        "team_size",
        "experts_with_distribution",
        "valid_pairs",
        "possible_pairs",
        "specscore",
        "avg_matched_nodes_per_expert",
        "negative_similarities",
    ]

    def detail_for_teams(label: str, teams: Dict[str, List[str]]) -> List[dict]:
        print(f"computing specscore label={label}", flush=True)
        detail_rows = [
            specscore_for_team(
                paper_id,
                team,
                rows_by_paper,
                task_id_to_row,
                task_arr,
                expert_index,
                expert_arr,
                clamp_negative=True,
            )
            for paper_id, team in sorted(teams.items())
        ]
        for row in detail_rows:
            row["specscore"] = f"{float(row['specscore']):.12f}"
            row["avg_matched_nodes_per_expert"] = f"{float(row['avg_matched_nodes_per_expert']):.6f}"
        write_tsv(out_dir / f"{label}.task_specscore.tsv", detail_rows, detail_fields)
        return detail_rows

    assignment_fields = [
        "assignment_type",
        "method",
        "paper_id",
        "node_id",
        "node_name",
        "assigned_expert_id",
        "assigned_expert_name",
        "assignment_score",
    ]
    expert_compactness_fields = [
        "assignment_type",
        "method",
        "paper_id",
        "expert_id",
        "expert_name",
        "assigned_node_count",
        "tree_closure_node_count",
        "extra_connector_node_count",
        "compactness",
        "assigned_node_ids",
    ]
    compactness_summary_by_assignment = {}
    assignment_detail_rows = []
    expert_compactness_rows = []

    def compactness_summary_for_teams(
        assignment_type: str,
        label: str,
        teams: Dict[str, List[str]],
        scores: Dict[Tuple[str, str], float],
    ) -> dict:
        task_rows_out = []
        for paper_id, team in sorted(teams.items()):
            task_result, assignments, expert_rows = compactness_for_task(
                paper_id,
                team,
                rows_by_paper.get(paper_id, []),
                scores,
                ancestor_paths,
            )
            task_rows_out.append(task_result)
            for row in assignments:
                assignment_detail_rows.append(
                    {
                        "assignment_type": assignment_type,
                        "method": label,
                        "assigned_expert_name": expert_names.get(row["assigned_expert_id"], row["assigned_expert_id"]),
                        **format_float_row(row),
                    }
                )
            for row in expert_rows:
                expert_compactness_rows.append(
                    {
                        "assignment_type": assignment_type,
                        "method": label,
                        "expert_name": expert_names.get(row["expert_id"], row["expert_id"]),
                        **format_float_row(row),
                    }
                )
        return summarize_compactness(label, task_rows_out)

    def native_compactness_summary_for_teams(
        label: str,
        teams: Dict[str, List[str]],
        assignments_by_paper: Dict[str, List[dict]],
    ) -> dict:
        task_rows_out = []
        for paper_id, team in sorted(teams.items()):
            task_result, assignments, expert_rows = compactness_for_native_assignments(
                paper_id,
                team,
                rows_by_paper.get(paper_id, []),
                assignments_by_paper.get(paper_id, []),
                ancestor_paths,
            )
            task_rows_out.append(task_result)
            for row in assignments:
                assignment_detail_rows.append(
                    {
                        "assignment_type": "native_assignment",
                        "method": label,
                        "assigned_expert_name": expert_names.get(row["assigned_expert_id"], row["assigned_expert_id"]),
                        **format_float_row(row),
                    }
                )
            for row in expert_rows:
                expert_compactness_rows.append(
                    {
                        "assignment_type": "native_assignment",
                        "method": label,
                        "expert_name": expert_names.get(row["expert_id"], row["expert_id"]),
                        **format_float_row(row),
                    }
                )
        return summarize_compactness(label, task_rows_out)

    for method in METHODS:
        if method == "Random mean 5":
            random_summaries = []
            random_specialty_compactness = []
            random_authority_compactness = []
            for idx, teams in enumerate(random_runs[method], start=1):
                label = f"Random seed run {idx}"
                random_summaries.append(summarize_specscore(label, detail_for_teams(label, teams)))
                random_specialty_compactness.append(
                    compactness_summary_for_teams("specialty_assignment", label, teams, specialty_scores)
                )
                random_authority_compactness.append(
                    compactness_summary_for_teams("authority_assignment", label, teams, authority_scores)
                )
            row = aggregate_specscore_random(random_summaries)
            compactness_summary_by_assignment[(method, "specialty_assignment")] = aggregate_compactness_random(
                random_specialty_compactness
            )
            compactness_summary_by_assignment[(method, "authority_assignment")] = aggregate_compactness_random(
                random_authority_compactness
            )
        else:
            row = summarize_specscore(method, detail_for_teams(method, teams_by_method[method]))
            row["random_runs"] = ""
            if method in native_assignments_by_method:
                compactness_summary_by_assignment[(method, "native_assignment")] = native_compactness_summary_for_teams(
                    method,
                    teams_by_method[method],
                    native_assignments_by_method[method],
                )
            compactness_summary_by_assignment[(method, "specialty_assignment")] = compactness_summary_for_teams(
                "specialty_assignment",
                method,
                teams_by_method[method],
                specialty_scores,
            )
            compactness_summary_by_assignment[(method, "authority_assignment")] = compactness_summary_for_teams(
                "authority_assignment",
                method,
                teams_by_method[method],
                authority_scores,
            )
        row["method"] = method
        row["responsibility_diversity_mean_specscore"] = row["mean_specscore_by_task"]
        row["responsibility_diversity_pair_weighted_specscore"] = row["pair_weighted_specscore"]
        row["responsibility_compactness_specialty_assignment"] = compactness_summary_by_assignment[
            (method, "specialty_assignment")
        ]["mean_team_compactness"]
        row["responsibility_compactness_authority_assignment"] = compactness_summary_by_assignment[
            (method, "authority_assignment")
        ]["mean_team_compactness"]
        row["responsibility_compactness_native_assignment"] = (
            compactness_summary_by_assignment.get((method, "native_assignment"), {}).get("mean_team_compactness", "")
        )
        summary_rows.append(row)

    fields = [
        "method",
        "tasks",
        "random_runs",
        "tasks_with_valid_pairs",
        "responsibility_diversity_mean_specscore",
        "responsibility_diversity_pair_weighted_specscore",
        "responsibility_compactness_native_assignment",
        "responsibility_compactness_specialty_assignment",
        "responsibility_compactness_authority_assignment",
        "valid_pairs",
        "possible_pairs",
        "pair_coverage",
        "avg_experts_with_distribution",
        "avg_matched_nodes_per_expert",
        "negative_similarities",
    ]
    write_tsv(out_dir / "team_structure_completed_selected_methods.tsv", summary_rows, fields)
    write_tsv(
        out_dir / "responsibility_node_assignments_taxonomy_compactness.tsv",
        assignment_detail_rows,
        assignment_fields,
    )
    write_tsv(
        out_dir / "responsibility_expert_compactness_details.tsv",
        expert_compactness_rows,
        expert_compactness_fields,
    )


def write_definitions(out_dir: Path) -> None:
    rows = [
        {
            "metric": "binary_coverage_mean_normalized",
            "definition": "Per task, fraction of direct task FoS nodes covered by at least one selected expert direct profile node; averaged over tasks.",
        },
        {
            "metric": "specialty_coverage_mean_normalized",
            "definition": "For each task skill, max selected-expert specialty score, where specialty is the expert's direct_weight_sum for that skill divided by the expert's total direct profile weight; weighted by task skill weight and averaged over tasks.",
        },
        {
            "metric": "authority_coverage_mean_weighted_normalized",
            "definition": "For each task skill, max selected-expert direct_weight_sum divided by global best direct_weight_sum for that skill, weighted by task skill weight and averaged over tasks.",
        },
        {
            "metric": "responsibility_diversity_mean_specscore",
            "definition": "Existing weighted SpecScore: mean pairwise JSD between selected experts' task-node responsibility distributions.",
        },
        {
            "metric": "responsibility_compactness_native_assignment",
            "definition": "Taxonomy-structure compactness using method-native region responsibility assignment where available. Responsibility cut and expert-distribution cut use their method-generated regions.tsv region-owner assignments. Embedding BFS, Seq2seq, and Random mean 5 do not have comparable native region partitions, so this field is blank.",
        },
        {
            "metric": "responsibility_compactness_specialty_assignment",
            "definition": "For each task taxonomy node, assign it to the selected expert with the highest specialty score, where specialty is expert subtree_weight_sum at that node divided by the expert's total direct profile weight. For each expert, compactness is assigned nodes divided by the minimal taxonomy tree-closure nodes needed to connect them. Team compactness is the mean over selected experts.",
        },
        {
            "metric": "responsibility_compactness_authority_assignment",
            "definition": "For each task taxonomy node, assign it to the selected expert with the highest authority score, where authority is expert subtree_weight_sum at that node. Compactness then uses assigned nodes divided by minimal taxonomy tree-closure nodes, averaged over selected experts.",
        },
        {
            "metric": "Random mean 5",
            "definition": "Mean over five random runs with seeds 13-17, sampling from the same method/history expert pool used by the soft-groundtruth random baseline.",
        },
    ]
    write_tsv(out_dir / "completed_metric_definitions.tsv", rows, ["metric", "definition"])


def main() -> None:
    out_dir = ROOT / "output/selected_method_results"
    _, _, _, teams_by_method, random_runs = load_selected_teams()
    compute_task_coverage(teams_by_method, random_runs, out_dir)
    compute_specscore_and_compactness(teams_by_method, random_runs, out_dir)
    write_definitions(out_dir)
    print(f"completed outputs in {out_dir}", flush=True)


if __name__ == "__main__":
    main()
