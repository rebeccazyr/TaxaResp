#!/usr/bin/env python3
"""Evaluate taxonomy region cuts with embedding candidate distributions.

Pipeline:
1. Keep top-M same-node experts for every task taxonomy node.
2. Convert each node's top-M similarities into an expert distribution.
3. Score each taxonomy edge by Jensen-Shannon divergence between endpoint
   distributions.
4. Cut k-1 high-boundary edges, where k is the target team size.
5. Assign one expert owner to each region with maximum weight bipartite
   matching by summed node-owner score.

The overlap/redundancy metrics are reported after assignment only; they are not
used in optimization.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import math
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from embedding_pipeline_utils import load_child_to_parents, read_jsonl


METHOD = "embedding_taxonomy_region_cut_jsd_topm_owner_matching"
VIRTUAL_ROOT = "__task_root__"
REGION_WEIGHT_CHOICES = ("importance", "log_sum")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate embedding taxonomy region cut owner assignment"
    )
    p.add_argument("--task-nodes-jsonl", required=True)
    p.add_argument("--task-node-ids", required=True)
    p.add_argument("--task-node-embeddings", required=True)
    p.add_argument("--expert-node-ids", required=True)
    p.add_argument("--expert-node-embeddings", required=True)
    p.add_argument("--fos-children", default="data/dblp/13.FieldOfStudyChildren.nt")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--top-m", type=int, default=64)
    p.add_argument("--top-k-output", type=int, default=20)
    p.add_argument("--distribution-temperature", type=float, default=0.05)
    p.add_argument(
        "--region-weight",
        choices=REGION_WEIGHT_CHOICES,
        default="importance",
        help="Node weight used in Score(e, R).",
    )
    p.add_argument(
        "--allow-repeat-experts",
        action="store_true",
        help="Assign each region independently instead of enforcing unique experts.",
    )
    return p.parse_args()


def mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


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


def precompute_task_node_rankings(
    task_rows_by_paper: Dict[str, list],
    task_embeddings: Dict[str, np.ndarray],
    experts_by_node: Dict[str, List[int]],
    expert_ids: List[str],
    expert_arr: np.ndarray,
    top_m: int,
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
        queries = np.vstack([task_embeddings[row["task_node_id"]] for row in rows]).astype(
            np.float32
        )
        scores = queries @ mat.T
        keep = min(top_m, len(candidate_rows))
        for q_idx, row in enumerate(rows):
            row_scores = scores[q_idx]
            if keep < len(candidate_rows):
                top_pos = np.argpartition(-row_scores, keep - 1)[:keep]
                top_pos = top_pos[np.argsort(-row_scores[top_pos])]
            else:
                top_pos = np.argsort(-row_scores)
            rankings[row["task_node_id"]] = [
                (expert_ids[candidate_rows[pos]], float(row_scores[pos])) for pos in top_pos
            ]
    return rankings


def score_distribution(
    ranking: Sequence[Tuple[str, float]], temperature: float
) -> Dict[str, float]:
    if not ranking:
        return {}
    temp = max(float(temperature), 1e-6)
    vals = np.array([score for _, score in ranking], dtype=np.float64) / temp
    vals -= vals.max()
    probs = np.exp(vals)
    total = float(probs.sum())
    if total <= 0:
        return {}
    probs /= total
    return {expert_id: float(prob) for (expert_id, _), prob in zip(ranking, probs)}


def js_divergence(p: Dict[str, float], q: Dict[str, float]) -> float:
    keys = set(p) | set(q)
    if not keys:
        return 0.0
    out = 0.0
    for key in keys:
        pv = p.get(key, 0.0)
        qv = q.get(key, 0.0)
        mv = 0.5 * (pv + qv)
        if pv > 0:
            out += 0.5 * pv * math.log(pv / mv, 2)
        if qv > 0:
            out += 0.5 * qv * math.log(qv / mv, 2)
    return out


def weighted_mixture(distributions: Iterable[Tuple[Dict[str, float], float]]) -> Dict[str, float]:
    out: Dict[str, float] = defaultdict(float)
    total = 0.0
    for dist, weight in distributions:
        if weight <= 0:
            continue
        total += weight
        for expert_id, prob in dist.items():
            out[expert_id] += weight * prob
    if total <= 0:
        return {}
    return {expert_id: value / total for expert_id, value in out.items()}


def parse_subtree_skill_ids(row: dict) -> set:
    ids = set()
    for part in str(row.get("subtree_skills") or "").split(";"):
        name = part.rsplit(":", 1)[0].strip()
        if name:
            ids.add(name)
    return ids


def choose_task_parent(
    row: dict,
    node_ids: set,
    row_by_node: Dict[str, dict],
    child_to_parents: Dict[str, List[str]],
) -> str:
    node_id = str(row["node_id"])
    direct_parents = [p for p in child_to_parents.get(node_id, []) if p in node_ids]
    if direct_parents:
        child_skills = parse_subtree_skill_ids(row)

        def key(parent_id: str) -> tuple:
            parent = row_by_node[parent_id]
            level_gap = safe_int(row.get("node_level"), 99) - safe_int(
                parent.get("node_level"), 99
            )
            parent_skills = parse_subtree_skill_ids(parent)
            overlap = len(child_skills & parent_skills)
            return (level_gap, -overlap, str(parent_id))

        return sorted(direct_parents, key=key)[0]
    return VIRTUAL_ROOT


def build_task_tree(
    rows: List[dict],
    child_to_parents: Dict[str, List[str]],
    distributions: Dict[str, Dict[str, float]],
) -> Tuple[List[Tuple[str, str, float]], Dict[str, List[str]]]:
    row_by_node = {str(row["node_id"]): row for row in rows}
    node_ids = set(row_by_node)
    root_dist = weighted_mixture(
        (
            distributions.get(row["task_node_id"], {}),
            max(safe_float(row.get("node_importance"), 0.0), 1e-6),
        )
        for row in rows
    )
    dist_by_node = {
        str(row["node_id"]): distributions.get(row["task_node_id"], {}) for row in rows
    }
    dist_by_node[VIRTUAL_ROOT] = root_dist

    edges: List[Tuple[str, str, float]] = []
    children: Dict[str, List[str]] = defaultdict(list)
    for row in rows:
        child_id = str(row["node_id"])
        parent_id = choose_task_parent(row, node_ids, row_by_node, child_to_parents)
        boundary = js_divergence(dist_by_node.get(parent_id, {}), dist_by_node.get(child_id, {}))
        edges.append((parent_id, child_id, boundary))
        children[parent_id].append(child_id)
    return edges, children


def selected_cut_edges(
    edges: List[Tuple[str, str, float]],
    k: int,
) -> List[Tuple[str, str, float]]:
    n_cuts = max(0, min(k - 1, len(edges)))
    if n_cuts == 0:
        return []
    selected = sorted(edges, key=lambda e: (e[2], e[0], e[1]), reverse=True)[:n_cuts]
    virtual_edges = [edge for edge in edges if edge[0] == VIRTUAL_ROOT]
    if virtual_edges and len(selected) == len(virtual_edges) and all(
        edge[0] == VIRTUAL_ROOT for edge in selected
    ):
        replacement = next((edge for edge in sorted(edges, key=lambda e: e[2], reverse=True) if edge not in selected), None)
        if replacement is not None:
            selected[-1] = replacement
    return selected


def connected_regions(
    rows: List[dict],
    edges: List[Tuple[str, str, float]],
    cuts: List[Tuple[str, str, float]],
) -> List[List[str]]:
    cut_pairs = {(u, v) for u, v, _ in cuts}
    graph: Dict[str, List[str]] = defaultdict(list)
    all_nodes = {VIRTUAL_ROOT}
    all_nodes.update(str(row["node_id"]) for row in rows)
    for parent, child, _ in edges:
        if (parent, child) in cut_pairs:
            continue
        graph[parent].append(child)
        graph[child].append(parent)
    regions: List[List[str]] = []
    seen = set()
    for start in sorted(all_nodes):
        if start in seen:
            continue
        q = deque([start])
        seen.add(start)
        comp = []
        while q:
            node = q.popleft()
            if node != VIRTUAL_ROOT:
                comp.append(node)
            for nxt in graph.get(node, []):
                if nxt in seen:
                    continue
                seen.add(nxt)
                q.append(nxt)
        if comp:
            regions.append(sorted(comp))
    regions.sort(key=lambda region: (-len(region), region[0]))
    return regions


def node_weight(row: dict, weight_mode: str = "importance") -> float:
    if weight_mode == "log_sum":
        return max(safe_float(row.get("node_log_sum"), 0.0), 0.0)
    return max(safe_float(row.get("node_importance"), 0.0), 0.0)


def score_regions(
    regions: List[List[str]],
    row_by_node: Dict[str, dict],
    rankings: Dict[str, List[Tuple[str, float]]],
    weight_mode: str = "importance",
) -> Tuple[List[Dict[str, float]], Dict[Tuple[int, str], Tuple[str, float]]]:
    region_scores: List[Dict[str, float]] = []
    best_node: Dict[Tuple[int, str], Tuple[str, float]] = {}
    for region_idx, region in enumerate(regions):
        scores: Dict[str, float] = defaultdict(float)
        best: Dict[str, Tuple[str, float]] = {}
        for node_id in region:
            row = row_by_node[node_id]
            weight = node_weight(row, weight_mode)
            for expert_id, sim in rankings.get(row["task_node_id"], []):
                contribution = weight * sim
                scores[expert_id] += contribution
                prev = best.get(expert_id)
                if prev is None or contribution > prev[1]:
                    best[expert_id] = (node_id, contribution)
        for expert_id, item in best.items():
            best_node[(region_idx, expert_id)] = item
        region_scores.append(dict(scores))
    return region_scores, best_node


def assign_region_owners(
    region_scores: List[Dict[str, float]],
    allow_repeat: bool,
) -> List[Tuple[int, str, float]]:
    if allow_repeat:
        out = []
        for region_idx, scores in enumerate(region_scores):
            if not scores:
                continue
            expert_id, score = max(scores.items(), key=lambda item: item[1])
            out.append((region_idx, expert_id, score))
        return out

    expert_ids = sorted({expert_id for scores in region_scores for expert_id in scores})
    if not expert_ids:
        return []
    expert_to_col = {expert_id: idx for idx, expert_id in enumerate(expert_ids)}
    score_matrix = np.full((len(region_scores), len(expert_ids)), -1e9, dtype=np.float64)
    for region_idx, scores in enumerate(region_scores):
        for expert_id, score in scores.items():
            score_matrix[region_idx, expert_to_col[expert_id]] = score
    row_ind, col_ind = linear_sum_assignment(-score_matrix)
    out = []
    for region_idx, expert_col in zip(row_ind, col_ind):
        score = float(score_matrix[region_idx, expert_col])
        if score <= -1e8:
            continue
        out.append((int(region_idx), expert_ids[int(expert_col)], score))
    out.sort(key=lambda item: item[2], reverse=True)
    return out


def region_label(
    region: List[str],
    row_by_node: Dict[str, dict],
    max_names: int = 4,
    weight_mode: str = "importance",
) -> str:
    rows = [row_by_node[node_id] for node_id in region]
    rows.sort(
        key=lambda row: (
            -node_weight(row, weight_mode),
            safe_int(row.get("node_level"), 99),
            str(row.get("node_name") or row.get("node_id")),
        )
    )
    names = [str(row.get("node_name") or row["node_id"]) for row in rows[:max_names]]
    return " / ".join(names)


def responsibility_overlap(
    selected: List[Tuple[int, str, float]],
    rows: List[dict],
    rankings: Dict[str, List[Tuple[str, float]]],
    weight_mode: str = "importance",
) -> float:
    if len(selected) < 2:
        return 0.0
    selected_experts = [expert_id for _, expert_id, _ in selected]
    vectors: Dict[str, List[float]] = {expert_id: [] for expert_id in selected_experts}
    for row in rows:
        score_by_expert = dict(rankings.get(row["task_node_id"], []))
        weight = node_weight(row, weight_mode)
        for expert_id in selected_experts:
            vectors[expert_id].append(max(score_by_expert.get(expert_id, 0.0), 0.0) * weight)

    normalized = {}
    for expert_id, values in vectors.items():
        total = sum(values)
        normalized[expert_id] = [v / total for v in values] if total > 0 else values

    overlaps = []
    for a, b in itertools.combinations(selected_experts, 2):
        overlaps.append(sum(min(x, y) for x, y in zip(normalized[a], normalized[b])))
    return mean(overlaps)


def summarize_results(task_results: List[dict], region_weight: str) -> dict:
    micro_hits = sum(r["hits"] for r in task_results)
    micro_selected = sum(r["selected"] for r in task_results)
    micro_positives = sum(r["positives"] for r in task_results)
    return {
        "method": METHOD,
        "region_weight": region_weight,
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
        "avg_cut_boundary_score": f"{mean([r['avg_cut_boundary_score'] for r in task_results]):.12f}",
        "avg_team_responsibility_overlap": f"{mean([r['team_responsibility_overlap'] for r in task_results]):.12f}",
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
    task_rows_by_paper: Dict[str, list] = defaultdict(list)
    task_info = {}
    for row in read_jsonl(Path(args.task_nodes_jsonl)):
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

    print("loading taxonomy parent links", flush=True)
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

    print("building node expert distributions", flush=True)
    distributions = {
        task_node_id: score_distribution(ranking, args.distribution_temperature)
        for task_node_id, ranking in task_node_rankings.items()
    }

    node_candidate_rows = []
    edge_rows = []
    region_rows = []
    prediction_rows = []
    task_results = []

    for task_idx, (paper_id, rows) in enumerate(sorted(task_rows_by_paper.items()), start=1):
        if task_idx % 25 == 0:
            print(f"eval_progress tasks={task_idx:,}/{len(task_rows_by_paper):,}", flush=True)
        positives = task_info[paper_id]["members"]
        team_size = max(1, task_info[paper_id]["team_size"])
        row_by_node = {str(row["node_id"]): row for row in rows}

        for row in rows:
            for rank, (expert_id, score) in enumerate(
                task_node_rankings.get(row["task_node_id"], [])[: args.top_m], start=1
            ):
                node_candidate_rows.append(
                    {
                        "paper_id": paper_id,
                        "node_id": row["node_id"],
                        "node_name": row.get("node_name", row["node_id"]),
                        "rank": rank,
                        "expert_id": expert_id,
                        "expert_name": expert_names.get(expert_id, expert_id),
                        "similarity": f"{score:.6f}",
                        "distribution_prob": f"{distributions.get(row['task_node_id'], {}).get(expert_id, 0.0):.12f}",
                    }
                )

        edges, _ = build_task_tree(rows, child_to_parents, distributions)
        cuts = selected_cut_edges(edges, team_size)
        cut_pairs = {(u, v) for u, v, _ in cuts}
        for parent, child, boundary in edges:
            edge_rows.append(
                {
                    "paper_id": paper_id,
                    "parent_node_id": parent,
                    "parent_node_name": "Task" if parent == VIRTUAL_ROOT else row_by_node[parent].get("node_name", parent),
                    "child_node_id": child,
                    "child_node_name": row_by_node[child].get("node_name", child),
                    "boundary_score_jsd": f"{boundary:.12f}",
                    "is_cut": "1" if (parent, child) in cut_pairs else "0",
                }
            )

        regions = connected_regions(rows, edges, cuts)
        region_scores, best_node = score_regions(
            regions, row_by_node, task_node_rankings, args.region_weight
        )
        selected = assign_region_owners(region_scores, args.allow_repeat_experts)

        selected_ids = [expert_id for _, expert_id, _ in selected]
        hits = len(positives.intersection(selected_ids))
        duplicates = len(selected_ids) - len(set(selected_ids))
        overlap = responsibility_overlap(
            selected, rows, task_node_rankings, args.region_weight
        )
        avg_cut_boundary = mean([boundary for _, _, boundary in cuts])

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
                "avg_cut_boundary_score": avg_cut_boundary,
                "team_responsibility_overlap": overlap,
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
                    "region_label": region_label(
                        region, row_by_node, weight_mode=args.region_weight
                    ),
                    "node_ids": "|".join(region),
                    "node_names": "|".join(row_by_node[n].get("node_name", n) for n in region),
                    "node_count": len(region),
                    "region_weight": f"{sum(node_weight(row_by_node[n], args.region_weight) for n in region):.6f}",
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
                    "region_label": region_label(
                        region, row_by_node, weight_mode=args.region_weight
                    ),
                    "expert_id": expert_id,
                    "expert_name": expert_names.get(expert_id, expert_id),
                    "score": f"{score:.6f}",
                    "best_node_id": best_node_id,
                    "best_node_name": row_by_node[best_node_id].get("node_name", best_node_id) if best_node_id else "",
                    "best_node_score": f"{best_node_score:.6f}",
                    "is_actual_member": "1" if expert_id in positives else "0",
                }
            )

    metric_row = summarize_results(task_results, args.region_weight)
    with (out_dir / "metrics_summary.tsv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(metric_row), delimiter="\t")
        writer.writeheader()
        writer.writerow(metric_row)

    with (out_dir / "node_topm_candidates.tsv").open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "paper_id",
            "node_id",
            "node_name",
            "rank",
            "expert_id",
            "expert_name",
            "similarity",
            "distribution_prob",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(node_candidate_rows)

    with (out_dir / "edge_boundaries.tsv").open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "paper_id",
            "parent_node_id",
            "parent_node_name",
            "child_node_id",
            "child_node_name",
            "boundary_score_jsd",
            "is_cut",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(edge_rows)

    with (out_dir / "regions.tsv").open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "paper_id",
            "region_id",
            "region_label",
            "node_ids",
            "node_names",
            "node_count",
            "region_weight",
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
            "region_label",
            "expert_id",
            "expert_name",
            "score",
            "best_node_id",
            "best_node_name",
            "best_node_score",
            "is_actual_member",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(prediction_rows)

    print(f"tasks={len(task_results)}")
    print(
        f"{METHOD} p={float(metric_row['percent_precision_at_team_size']):.4f}% "
        f"r={float(metric_row['percent_recall_at_team_size']):.4f}% "
        f"overlap={float(metric_row['avg_team_responsibility_overlap']):.6f}"
    )
    print(f"out_dir={out_dir}")


if __name__ == "__main__":
    main()
