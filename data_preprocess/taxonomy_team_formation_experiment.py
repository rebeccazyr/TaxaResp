#!/usr/bin/env python3
"""Taxonomy-aware expert retrieval experiment for 2020+ team tasks.

The task side uses ``teams_2020plus_with_skill_weights.csv`` as a proxy for a
task description: each paper's weighted skills are treated as the required
capabilities. The expert side uses pre-2020 per-expert FoS profiles.

Two retrieval methods are compared:
- direct: weighted overlap between task skills and expert direct FoS nodes.
- taxonomy: direct nodes are propagated upward through the MAG FoS taxonomy.

The propagated vectors are sparse "taxonomy embeddings": each dimension is a
FoS node in the taxonomy, and values describe how strongly a task or expert
occupies that node under its current context. Expert direct node weights are
normalized within each expert profile. IDF downweights overly broad nodes such
as Computer science.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


DEFAULT_TASKS = "data_preprocess/teams_2020plus_with_skill_weights.csv"
DEFAULT_PROFILE_DIR = "output/expert_profile_year_bins/all_2000_2019"
DEFAULT_EXPERTS = "data/dblp/expert_id_name.tsv"
DEFAULT_FOS_MAP = "data/dblp/FieldsOfStudy.txt"
DEFAULT_FOS_CHILDREN = "data/dblp/13.FieldOfStudyChildren.nt"
DEFAULT_OUT = "output/taxonomy_team_formation_experiment"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run a taxonomy-aware team-formation retrieval experiment"
    )
    p.add_argument("--tasks-csv", default=DEFAULT_TASKS)
    p.add_argument("--profile-dir", default=DEFAULT_PROFILE_DIR)
    p.add_argument("--expert-tsv", default=DEFAULT_EXPERTS)
    p.add_argument("--fos-map", default=DEFAULT_FOS_MAP)
    p.add_argument("--fos-children", default=DEFAULT_FOS_CHILDREN)
    p.add_argument("--out-dir", default=DEFAULT_OUT)
    p.add_argument(
        "--max-profile-nodes",
        type=int,
        default=120,
        help="Keep top-N direct FoS rows per expert, sorted by profile weight",
    )
    p.add_argument(
        "--ancestor-depth",
        type=int,
        default=5,
        help="Maximum number of taxonomy parent hops used for propagation",
    )
    p.add_argument(
        "--decay",
        type=float,
        default=0.55,
        help="Multiplicative decay per parent hop during taxonomy propagation",
    )
    p.add_argument(
        "--missing-skill-weight",
        type=float,
        default=0.0,
        help="Weight used when a task skill has weight 0 or missing",
    )
    p.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Number of recommendations stored per task",
    )
    p.add_argument(
        "--blend-taxonomy-weight",
        type=float,
        default=0.25,
        help="Weight assigned to taxonomy score in the direct+taxonomy blend",
    )
    p.add_argument(
        "--case-index",
        type=int,
        default=0,
        help="0-based task row index to export as a readable taxonomy case",
    )
    return p.parse_args()


def norm_name(s: str) -> str:
    s = s.lower().replace("_", " ")
    s = s.replace("–", "-").replace("—", "-").replace("‑", "-")
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def safe_float(v: object, default: float = 0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def load_expert_names(path: Path) -> Dict[str, str]:
    names: Dict[str, str] = {}
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            expert_id = (row.get("expert_id") or "").strip()
            if expert_id:
                names[expert_id] = (row.get("name") or expert_id).strip()
    return names


def load_fos_map(path: Path) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, int]]:
    name_to_id: Dict[str, str] = {}
    id_to_name: Dict[str, str] = {}
    id_to_level: Dict[str, int] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 6:
                continue
            fos_id = parts[0].strip()
            norm = parts[2].strip()
            display = parts[3].strip()
            level = int(parts[5]) if parts[5].isdigit() else -1
            if not fos_id:
                continue
            id_to_name[fos_id] = display or norm or fos_id
            id_to_level[fos_id] = level
            for name in (norm, display):
                key = norm_name(name)
                if key:
                    name_to_id[key] = fos_id
    return name_to_id, id_to_name, id_to_level


def load_child_to_parents(path: Path) -> Dict[str, List[str]]:
    child_to_parents: Dict[str, List[str]] = defaultdict(list)
    pat = re.compile(
        r"<https://makg.org/entity/(\d+)>\s+"
        r"<https://makg.org/property/hasParent>\s+"
        r"<https://makg.org/entity/(\d+)>\s+\."
    )
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            m = pat.match(line.strip())
            if m:
                child_to_parents[m.group(1)].append(m.group(2))
    return dict(child_to_parents)


def ancestor_cache_builder(
    child_to_parents: Dict[str, List[str]], max_depth: int
):
    cache: Dict[str, List[Tuple[str, int]]] = {}

    def ancestors(seed: str) -> List[Tuple[str, int]]:
        if seed in cache:
            return cache[seed]
        out = [(seed, 0)]
        q = deque([(seed, 0)])
        seen = {seed}
        while q:
            node, dist = q.popleft()
            if dist >= max_depth:
                continue
            for parent in child_to_parents.get(node, []):
                if parent in seen:
                    continue
                seen.add(parent)
                out.append((parent, dist + 1))
                q.append((parent, dist + 1))
        cache[seed] = out
        return out

    return ancestors


def parse_task_members(members: str) -> List[str]:
    ids = []
    for member in (members or "").split("|"):
        member = member.strip()
        if not member:
            continue
        ids.append(member.split("_", 1)[0])
    return ids


def load_tasks(
    path: Path,
    name_to_id: Dict[str, str],
    id_to_name: Dict[str, str],
    missing_weight: float,
) -> List[dict]:
    tasks: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            skills = (row.get("skills") or "").split("|")
            weights = (row.get("skill_weights") or "").split("|")
            direct: List[Tuple[str, float, str]] = []
            missing: List[str] = []
            for idx, raw_skill in enumerate(skills):
                skill = raw_skill.strip()
                if not skill:
                    continue
                fos_id = name_to_id.get(norm_name(skill))
                if not fos_id:
                    missing.append(skill)
                    continue
                w = safe_float(weights[idx], 0.0) if idx < len(weights) else 0.0
                if w <= 0:
                    w = missing_weight
                direct.append((fos_id, w, id_to_name.get(fos_id, skill)))
            tasks.append(
                {
                    "paper_id": row.get("paper_id", ""),
                    "row_idx": row.get("row_idx", ""),
                    "year": row.get("year", ""),
                    "team_size": int(float(row.get("team_size") or 0)),
                    "members": parse_task_members(row.get("members", "")),
                    "member_labels": row.get("members", ""),
                    "direct": direct,
                    "missing_skills": missing,
                    "raw_skills": row.get("skills", ""),
                }
            )
    return tasks


def expand_vector(
    direct_items: Sequence[Tuple[str, float, str]],
    ancestors,
    decay: float,
    transform_weight: bool,
) -> Dict[str, float]:
    vec: Dict[str, float] = defaultdict(float)
    for fos_id, weight, _ in direct_items:
        base = math.log1p(weight) if transform_weight else weight
        for node, dist in ancestors(fos_id):
            vec[node] += base * (decay ** dist)
    return dict(vec)


def direct_vector(
    direct_items: Sequence[Tuple[str, float, str]],
    transform_weight: bool,
) -> Dict[str, float]:
    vec: Dict[str, float] = defaultdict(float)
    for fos_id, weight, _ in direct_items:
        vec[fos_id] += math.log1p(weight) if transform_weight else weight
    return dict(vec)


def read_profile_direct_items(path: Path, limit: int) -> List[Tuple[str, float, str]]:
    rows: List[Tuple[str, float, str]] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            fos_id = (row.get("fos_id") or "").strip()
            if not fos_id:
                continue
            w = safe_float(row.get("direct_weight_sum"), 0.0)
            if w <= 0:
                continue
            rows.append((fos_id, w, row.get("fos_name") or fos_id))
    total_weight = sum(w for _, w, _ in rows)
    rows.sort(key=lambda x: x[1], reverse=True)
    rows = rows[:limit] if limit > 0 else rows
    if total_weight > 0:
        rows = [(fos_id, w / total_weight, name) for fos_id, w, name in rows]
    return rows


def build_expert_vectors(
    profile_dir: Path,
    expert_names: Dict[str, str],
    ancestors,
    max_profile_nodes: int,
    decay: float,
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, float]]]:
    direct_vectors: Dict[str, Dict[str, float]] = {}
    tax_vectors: Dict[str, Dict[str, float]] = {}
    files = sorted(p for p in profile_dir.glob("*_direct_fos_nodes.tsv") if not p.name.startswith("_"))
    for idx, path in enumerate(files, start=1):
        expert_id = path.name.replace("_direct_fos_nodes.tsv", "")
        if expert_id not in expert_names:
            continue
        direct_items = read_profile_direct_items(path, max_profile_nodes)
        direct_vectors[expert_id] = direct_vector(direct_items, transform_weight=False)
        tax_vectors[expert_id] = expand_vector(
            direct_items, ancestors, decay, transform_weight=False
        )
        if idx % 2000 == 0:
            print(f"loaded expert profiles {idx}/{len(files)}")
    return direct_vectors, tax_vectors


def build_idf(expert_vectors: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    n = len(expert_vectors)
    df: Dict[str, int] = defaultdict(int)
    for vec in expert_vectors.values():
        for node in vec:
            df[node] += 1
    return {node: math.log((n + 1) / (cnt + 1)) + 1.0 for node, cnt in df.items()}


def build_index(
    expert_vectors: Dict[str, Dict[str, float]], idf: Dict[str, float]
) -> Tuple[Dict[str, List[Tuple[str, float]]], Dict[str, float]]:
    index: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
    norms: Dict[str, float] = {}
    for expert_id, vec in expert_vectors.items():
        norm_sq = 0.0
        for node, raw in vec.items():
            val = raw * idf.get(node, 1.0)
            if val <= 0:
                continue
            index[node].append((expert_id, val))
            norm_sq += val * val
        norms[expert_id] = math.sqrt(norm_sq)
    return dict(index), norms


def score_task(
    task_vec: Dict[str, float],
    index: Dict[str, List[Tuple[str, float]]],
    expert_norms: Dict[str, float],
    idf: Dict[str, float],
    top_k: int,
) -> List[Tuple[str, float]]:
    scores: Dict[str, float] = defaultdict(float)
    qnorm_sq = 0.0
    for node, raw in task_vec.items():
        qval = raw * idf.get(node, 1.0)
        if qval <= 0:
            continue
        qnorm_sq += qval * qval
        for expert_id, eval_ in index.get(node, []):
            scores[expert_id] += qval * eval_
    qnorm = math.sqrt(qnorm_sq)
    if qnorm <= 0:
        return []
    ranked = []
    for expert_id, dot in scores.items():
        denom = qnorm * expert_norms.get(expert_id, 0.0)
        if denom > 0:
            ranked.append((expert_id, dot / denom))
    ranked.sort(key=lambda x: x[1], reverse=True)
    return ranked[:top_k]


def rank_all_positive_positions(
    task_vec: Dict[str, float],
    index: Dict[str, List[Tuple[str, float]]],
    expert_norms: Dict[str, float],
    idf: Dict[str, float],
    positives: set,
) -> Tuple[List[Tuple[str, float]], Dict[str, int]]:
    ranked = score_task(task_vec, index, expert_norms, idf, top_k=len(expert_norms))
    positions = {expert_id: rank for rank, (expert_id, _) in enumerate(ranked, start=1)}
    return ranked, {p: positions[p] for p in positives if p in positions}


def metrics_at_ks(
    ranked_ids: Sequence[str], positives: set, ks: Sequence[int]
) -> Dict[str, float]:
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

        out[f"precision_at_{k}"] = precision
        out[f"recall_at_{k}"] = recall
        out[f"ndcg_at_{k}"] = ndcg
        out[f"map_at_{k}"] = map_k
    return out


def init_metric_lists(ks: Sequence[int]) -> Dict[str, List[float]]:
    metric_lists: Dict[str, List[float]] = {}
    for metric in ("precision", "recall", "ndcg", "map"):
        for k in ks:
            metric_lists[f"{metric}_at_{k}"] = []
    metric_lists["reciprocal_rank"] = []
    metric_lists["recall_at_team_size"] = []
    return metric_lists


def summarize_metrics(
    method: str, n_tasks: int, metric_lists: Dict[str, List[float]]
) -> dict:
    def mean(xs: Sequence[float]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    metrics = {"method": method, "tasks": n_tasks}
    metrics["mean_recall_at_team_size"] = mean(metric_lists["recall_at_team_size"])
    for metric in ("precision", "recall", "ndcg", "map"):
        for k in (2, 5, 10, 20):
            key = f"{metric}_at_{k}"
            metrics[f"mean_{key}"] = mean(metric_lists[key])
    if "recall_all_assigned_experts" in metric_lists:
        metrics["mean_recall_all_assigned_experts"] = mean(
            metric_lists["recall_all_assigned_experts"]
        )
    metrics["mrr_first_actual_member"] = mean(metric_lists["reciprocal_rank"])
    return metrics


def evaluate_method(
    method: str,
    tasks: List[dict],
    task_vectors: List[Dict[str, float]],
    index: Dict[str, List[Tuple[str, float]]],
    expert_norms: Dict[str, float],
    idf: Dict[str, float],
    expert_names: Dict[str, str],
    top_k: int,
) -> Tuple[dict, List[dict]]:
    rows: List[dict] = []
    ks = (2, 5, 10, 20)
    metric_lists = init_metric_lists(ks)

    for task, task_vec in zip(tasks, task_vectors):
        positives = {x for x in task["members"] if x in expert_norms}
        ranked, positions = rank_all_positive_positions(
            task_vec, index, expert_norms, idf, positives
        )
        top_ids = [expert_id for expert_id, _ in ranked[:top_k]]
        ranked_ids = [expert_id for expert_id, _ in ranked]
        team_k = max(1, task["team_size"])

        def rec_at(k: int) -> float:
            if not positives:
                return 0.0
            return len(positives.intersection(ranked_ids[:k])) / len(positives)

        metric_lists["recall_at_team_size"].append(rec_at(team_k))
        for key, value in metrics_at_ks(ranked_ids, positives, ks).items():
            metric_lists[key].append(value)
        best_rank = min(positions.values()) if positions else 0
        metric_lists["reciprocal_rank"].append(1.0 / best_rank if best_rank else 0.0)

        for rank, (expert_id, score) in enumerate(ranked[:top_k], start=1):
            rows.append(
                {
                    "method": method,
                    "paper_id": task["paper_id"],
                    "rank": rank,
                    "expert_id": expert_id,
                    "expert_name": expert_names.get(expert_id, expert_id),
                    "score": f"{score:.6f}",
                    "is_actual_member": "1" if expert_id in positives else "0",
                    "actual_member_rank": positions.get(expert_id, ""),
                }
            )

    return summarize_metrics(method, len(tasks), metric_lists), rows


def evaluate_blend(
    tasks: List[dict],
    direct_task_vectors: List[Dict[str, float]],
    tax_task_vectors: List[Dict[str, float]],
    direct_index: Dict[str, List[Tuple[str, float]]],
    tax_index: Dict[str, List[Tuple[str, float]]],
    direct_norms: Dict[str, float],
    tax_norms: Dict[str, float],
    direct_idf: Dict[str, float],
    tax_idf: Dict[str, float],
    expert_names: Dict[str, str],
    top_k: int,
    taxonomy_weight: float,
) -> Tuple[dict, List[dict]]:
    taxonomy_weight = min(max(taxonomy_weight, 0.0), 1.0)
    direct_weight = 1.0 - taxonomy_weight

    rows: List[dict] = []
    ks = (2, 5, 10, 20)
    metric_lists = init_metric_lists(ks)
    n_experts = max(len(direct_norms), len(tax_norms))

    for task, direct_vec, tax_vec in zip(tasks, direct_task_vectors, tax_task_vectors):
        positives = {x for x in task["members"] if x in direct_norms or x in tax_norms}
        direct_scores = dict(
            score_task(direct_vec, direct_index, direct_norms, direct_idf, n_experts)
        )
        tax_scores = dict(score_task(tax_vec, tax_index, tax_norms, tax_idf, n_experts))
        blended = []
        for expert_id in set(direct_scores) | set(tax_scores):
            score = direct_weight * direct_scores.get(expert_id, 0.0)
            score += taxonomy_weight * tax_scores.get(expert_id, 0.0)
            blended.append((expert_id, score))
        blended.sort(key=lambda x: x[1], reverse=True)
        positions = {
            expert_id: rank for rank, (expert_id, _) in enumerate(blended, start=1)
        }
        top_ids = [expert_id for expert_id, _ in blended[:top_k]]
        ranked_ids = [expert_id for expert_id, _ in blended]
        team_k = max(1, task["team_size"])

        def rec_at(k: int) -> float:
            if not positives:
                return 0.0
            return len(positives.intersection(ranked_ids[:k])) / len(positives)

        metric_lists["recall_at_team_size"].append(rec_at(team_k))
        for key, value in metrics_at_ks(ranked_ids, positives, ks).items():
            metric_lists[key].append(value)
        positive_positions = [positions[p] for p in positives if p in positions]
        best_rank = min(positive_positions) if positive_positions else 0
        metric_lists["reciprocal_rank"].append(1.0 / best_rank if best_rank else 0.0)

        for rank, (expert_id, score) in enumerate(blended[:top_k], start=1):
            rows.append(
                {
                    "method": "direct_taxonomy_blend",
                    "paper_id": task["paper_id"],
                    "rank": rank,
                    "expert_id": expert_id,
                    "expert_name": expert_names.get(expert_id, expert_id),
                    "score": f"{score:.6f}",
                    "is_actual_member": "1" if expert_id in positives else "0",
                    "actual_member_rank": positions.get(expert_id, "")
                    if expert_id in positives
                    else "",
                }
            )

    return summarize_metrics("direct_taxonomy_blend", len(tasks), metric_lists), rows


def build_task_subtree_vectors(task: dict, ancestors) -> Dict[str, Dict[str, float]]:
    """For each task taxonomy node, collect weighted target skills in its subtree."""
    subtree_vectors: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for leaf_id, weight, _ in task["direct"]:
        for node, _ in ancestors(leaf_id):
            subtree_vectors[node][leaf_id] += weight
    return {node: dict(vec) for node, vec in subtree_vectors.items()}


def evaluate_subtree_cover(
    tasks: List[dict],
    ancestors,
    direct_index: Dict[str, List[Tuple[str, float]]],
    direct_norms: Dict[str, float],
    direct_idf: Dict[str, float],
    expert_names: Dict[str, str],
    id_to_name: Dict[str, str],
    id_to_level: Dict[str, int],
    top_k: int,
) -> Tuple[dict, List[dict], List[dict]]:
    """Assign one best expert to every task taxonomy node using subtree score."""
    rows: List[dict] = []
    assignment_rows: List[dict] = []
    ks = (2, 5, 10, 20)
    metric_lists = init_metric_lists(ks)

    for task in tasks:
        positives = set(task["members"])
        node_predictions: List[Tuple[str, float]] = []
        subtree_vectors = build_task_subtree_vectors(task, ancestors)

        for node_id, subtree_vec in subtree_vectors.items():
            ranked = score_task(
                subtree_vec,
                direct_index,
                direct_norms,
                direct_idf,
                top_k=1,
            )
            if not ranked:
                continue
            expert_id, score = ranked[0]
            node_importance = sum(subtree_vec.values())
            weighted_score = score * node_importance
            node_predictions.append((expert_id, weighted_score))
            assignment_rows.append(
                {
                    "method": "subtree_cover",
                    "paper_id": task["paper_id"],
                    "node_id": node_id,
                    "node_name": id_to_name.get(node_id, node_id),
                    "node_level": id_to_level.get(node_id, ""),
                    "subtree_skill_count": len(subtree_vec),
                    "node_importance": f"{node_importance:.6f}",
                    "expert_id": expert_id,
                    "expert_name": expert_names.get(expert_id, expert_id),
                    "score": f"{score:.6f}",
                    "weighted_score": f"{weighted_score:.6f}",
                    "is_actual_member": "1" if expert_id in positives else "0",
                }
            )

        ranked_with_duplicates = sorted(node_predictions, key=lambda x: x[1], reverse=True)
        ranked = []
        seen_experts = set()
        for expert_id, score in ranked_with_duplicates:
            if expert_id in seen_experts:
                continue
            seen_experts.add(expert_id)
            ranked.append((expert_id, score))
        ranked_ids = [expert_id for expert_id, _ in ranked]
        metric_lists.setdefault("recall_all_assigned_experts", []).append(
            len(positives.intersection(ranked_ids)) / len(positives)
            if positives
            else 0.0
        )
        positions = {
            expert_id: rank for rank, expert_id in enumerate(ranked_ids, start=1)
        }
        team_k = max(1, task["team_size"])

        def rec_at(k: int) -> float:
            if not positives:
                return 0.0
            return len(positives.intersection(ranked_ids[:k])) / len(positives)

        metric_lists["recall_at_team_size"].append(rec_at(team_k))
        for key, value in metrics_at_ks(ranked_ids, positives, ks).items():
            metric_lists[key].append(value)
        positive_positions = [positions[p] for p in positives if p in positions]
        best_rank = min(positive_positions) if positive_positions else 0
        metric_lists["reciprocal_rank"].append(1.0 / best_rank if best_rank else 0.0)

        for rank, (expert_id, score) in enumerate(ranked[:top_k], start=1):
            rows.append(
                {
                    "method": "subtree_cover",
                    "paper_id": task["paper_id"],
                    "rank": rank,
                    "expert_id": expert_id,
                    "expert_name": expert_names.get(expert_id, expert_id),
                    "score": f"{score:.6f}",
                    "is_actual_member": "1" if expert_id in positives else "0",
                    "actual_member_rank": positions.get(expert_id, "")
                    if expert_id in positives
                    else "",
                }
            )

    return (
        summarize_metrics("subtree_cover", len(tasks), metric_lists),
        rows,
        assignment_rows,
    )


def top_nodes(vec: Dict[str, float], id_to_name: Dict[str, str], id_to_level: Dict[str, int], n: int) -> List[str]:
    rows = sorted(vec.items(), key=lambda x: x[1], reverse=True)[:n]
    return [
        f"{id_to_name.get(node, node)} (id={node}, level={id_to_level.get(node, '')}, weight={weight:.3f})"
        for node, weight in rows
    ]


def write_case(
    out_path: Path,
    task: dict,
    task_direct_vec: Dict[str, float],
    task_tax_vec: Dict[str, float],
    direct_ranked: List[Tuple[str, float]],
    tax_ranked: List[Tuple[str, float]],
    expert_names: Dict[str, str],
    id_to_name: Dict[str, str],
    id_to_level: Dict[str, int],
) -> None:
    positives = set(task["members"])
    with out_path.open("w", encoding="utf-8") as f:
        f.write(f"paper_id: {task['paper_id']}\n")
        f.write(f"year: {task['year']}\n")
        f.write(f"raw_task_skills: {task['raw_skills']}\n")
        f.write(f"actual_members: {task['member_labels']}\n")
        if task["missing_skills"]:
            f.write(f"unmapped_skills: {'|'.join(task['missing_skills'])}\n")
        f.write("\nTop direct task nodes:\n")
        for line in top_nodes(task_direct_vec, id_to_name, id_to_level, 15):
            f.write(f"- {line}\n")
        f.write("\nTop propagated taxonomy nodes:\n")
        for line in top_nodes(task_tax_vec, id_to_name, id_to_level, 25):
            f.write(f"- {line}\n")

        def write_ranked(title: str, ranked: List[Tuple[str, float]]) -> None:
            f.write(f"\n{title}:\n")
            for rank, (expert_id, score) in enumerate(ranked[:20], start=1):
                flag = " actual_member" if expert_id in positives else ""
                f.write(
                    f"{rank:02d}. {expert_id}\t{expert_names.get(expert_id, expert_id)}"
                    f"\tscore={score:.6f}{flag}\n"
                )

        write_ranked("Direct recommendations", direct_ranked)
        write_ranked("Taxonomy recommendations", tax_ranked)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("loading FoS map")
    name_to_id, id_to_name, id_to_level = load_fos_map(Path(args.fos_map))
    print("loading taxonomy edges")
    child_to_parents = load_child_to_parents(Path(args.fos_children))
    ancestors = ancestor_cache_builder(child_to_parents, args.ancestor_depth)

    print("loading experts and tasks")
    expert_names = load_expert_names(Path(args.expert_tsv))
    tasks = load_tasks(
        Path(args.tasks_csv), name_to_id, id_to_name, args.missing_skill_weight
    )
    tasks = [t for t in tasks if t["direct"]]

    print("building expert vectors")
    direct_expert_vecs, tax_expert_vecs = build_expert_vectors(
        Path(args.profile_dir),
        expert_names,
        ancestors,
        args.max_profile_nodes,
        args.decay,
    )

    print("building indexes")
    direct_idf = build_idf(direct_expert_vecs)
    tax_idf = build_idf(tax_expert_vecs)
    direct_index, direct_norms = build_index(direct_expert_vecs, direct_idf)
    tax_index, tax_norms = build_index(tax_expert_vecs, tax_idf)

    task_direct_vecs = [
        direct_vector(t["direct"], transform_weight=False) for t in tasks
    ]
    task_tax_vecs = [
        expand_vector(t["direct"], ancestors, args.decay, transform_weight=False)
        for t in tasks
    ]

    print("evaluating")
    direct_metrics, direct_rows = evaluate_method(
        "direct",
        tasks,
        task_direct_vecs,
        direct_index,
        direct_norms,
        direct_idf,
        expert_names,
        args.top_k,
    )
    tax_metrics, tax_rows = evaluate_method(
        "taxonomy",
        tasks,
        task_tax_vecs,
        tax_index,
        tax_norms,
        tax_idf,
        expert_names,
        args.top_k,
    )
    blend_metrics, blend_rows = evaluate_blend(
        tasks,
        task_direct_vecs,
        task_tax_vecs,
        direct_index,
        tax_index,
        direct_norms,
        tax_norms,
        direct_idf,
        tax_idf,
        expert_names,
        args.top_k,
        args.blend_taxonomy_weight,
    )
    subtree_metrics, subtree_rows, subtree_assignment_rows = evaluate_subtree_cover(
        tasks,
        ancestors,
        direct_index,
        direct_norms,
        direct_idf,
        expert_names,
        id_to_name,
        id_to_level,
        args.top_k,
    )
    metrics_path = out_dir / "metrics_summary.tsv"
    metric_rows = [direct_metrics, tax_metrics, blend_metrics, subtree_metrics]
    fieldnames = []
    for row in metric_rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with metrics_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(metric_rows)

    pred_path = out_dir / "predictions_topk.tsv"
    with pred_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "method",
            "paper_id",
            "rank",
            "expert_id",
            "expert_name",
            "score",
            "is_actual_member",
            "actual_member_rank",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(direct_rows)
        writer.writerows(tax_rows)
        writer.writerows(blend_rows)
        writer.writerows(subtree_rows)

    assignment_path = out_dir / "subtree_cover_assignments.tsv"
    with assignment_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "method",
            "paper_id",
            "node_id",
            "node_name",
            "node_level",
            "subtree_skill_count",
            "node_importance",
            "expert_id",
            "expert_name",
            "score",
            "weighted_score",
            "is_actual_member",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(subtree_assignment_rows)

    case_idx = min(max(args.case_index, 0), len(tasks) - 1)
    direct_ranked = score_task(
        task_direct_vecs[case_idx],
        direct_index,
        direct_norms,
        direct_idf,
        args.top_k,
    )
    tax_ranked = score_task(
        task_tax_vecs[case_idx],
        tax_index,
        tax_norms,
        tax_idf,
        args.top_k,
    )
    case_path = out_dir / f"case_task_{tasks[case_idx]['paper_id']}.txt"
    write_case(
        case_path,
        tasks[case_idx],
        task_direct_vecs[case_idx],
        task_tax_vecs[case_idx],
        direct_ranked,
        tax_ranked,
        expert_names,
        id_to_name,
        id_to_level,
    )

    print(f"tasks={len(tasks)}")
    print(f"experts={len(tax_expert_vecs)}")
    print(f"metrics={metrics_path}")
    print(f"predictions={pred_path}")
    print(f"subtree_assignments={assignment_path}")
    print(f"case={case_path}")


if __name__ == "__main__":
    main()
