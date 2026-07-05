#!/usr/bin/env python3
"""Evaluate untrained role-to-expert-node retrieval with taxonomy aggregation.

This script compares frozen role-description embeddings against expert-node
embeddings built as unweighted means of linked history-paper embeddings. In
``direct`` mode, a history paper only links to its direct FoS labels. In
``ancestor`` mode, each direct FoS also links upward to all taxonomy ancestors.
``minimal_tree`` mode links each historical paper only to the direct and
connector nodes needed for its compact paper-local taxonomy tree.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_stage1 import canonical_dev_paths, canonical_official_test_paths
from src.stage1_smoke_training import Stage1Paths, Stage1Task, load_fos_name_to_id, load_tasks, read_ids_tsv


ENTITY_RE = re.compile(r"/entity/(\d+)>")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="outputs/untrained_taxonomy_aggregated_expert_nodes")
    parser.add_argument("--link-mode", choices=("direct", "ancestor", "minimal_tree"), default="direct")
    parser.add_argument(
        "--node-weight-mode",
        choices=("unweighted", "weighted"),
        default="unweighted",
        help="How linked history papers are averaged for non-root expert-node vectors.",
    )
    parser.add_argument(
        "--include-root-node",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Add a virtual root row scored by paper-abstract query vs each expert's all-history mean embedding.",
    )
    parser.add_argument(
        "--dev-root-ids-tsv",
        default="outputs/paper_to_expert_mean_embedding/dev_paper_embedding_ids.tsv",
    )
    parser.add_argument(
        "--dev-root-embeddings",
        default="outputs/paper_to_expert_mean_embedding/dev_paper_embeddings.npy",
    )
    parser.add_argument(
        "--official-test-root-ids-tsv",
        default="outputs/paper_to_expert_mean_embedding/test_paper_embedding_ids.tsv",
    )
    parser.add_argument(
        "--official-test-root-embeddings",
        default="outputs/paper_to_expert_mean_embedding/test_paper_embeddings.npy",
    )
    parser.add_argument("--taxonomy-edges", default="../data/dblp/13.FieldOfStudyChildren.nt")
    parser.add_argument("--include-dev", action="store_true", default=True)
    parser.add_argument("--include-official-test", action="store_true", default=True)
    parser.add_argument("--skip-dev", action="store_true")
    parser.add_argument("--skip-official-test", action="store_true")
    parser.add_argument("--dev-sample-jsonl", default="")
    parser.add_argument("--dev-role-descriptions-jsonl", default="")
    parser.add_argument("--dev-role-ids-tsv", default="")
    parser.add_argument("--dev-role-embeddings", default="")
    parser.add_argument("--official-test-sample-jsonl", default="")
    parser.add_argument("--official-test-role-descriptions-jsonl", default="")
    parser.add_argument("--official-test-role-ids-tsv", default="")
    parser.add_argument("--official-test-role-embeddings", default="")
    parser.add_argument("--max-papers", type=int, default=0)
    return parser.parse_args()


def override_paths(
    base: Stage1Paths,
    sample_jsonl: str = "",
    role_descriptions_jsonl: str = "",
    role_ids_tsv: str = "",
    role_embeddings_npy: str = "",
) -> Stage1Paths:
    return Stage1Paths(
        sample_jsonl=Path(sample_jsonl) if sample_jsonl else base.sample_jsonl,
        role_descriptions_jsonl=Path(role_descriptions_jsonl)
        if role_descriptions_jsonl
        else base.role_descriptions_jsonl,
        role_ids_tsv=Path(role_ids_tsv) if role_ids_tsv else base.role_ids_tsv,
        role_embeddings_npy=Path(role_embeddings_npy) if role_embeddings_npy else base.role_embeddings_npy,
        history_author_papers_tsv=base.history_author_papers_tsv,
        history_ids_tsv=base.history_ids_tsv,
        history_embeddings_npy=base.history_embeddings_npy,
        history_paper_fos_weights_tsv=base.history_paper_fos_weights_tsv,
        expert_profile_dir=base.expert_profile_dir,
        fos_map_tsv=base.fos_map_tsv,
    )


def safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def normalize_rows(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float32, copy=False)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return values / norms


def normalize_vector(value: np.ndarray) -> np.ndarray:
    value = value.astype(np.float32, copy=False)
    norm = float(np.linalg.norm(value))
    if norm <= 1e-12:
        return value
    return value / norm


def load_child_to_parents(path: Path) -> dict[str, set[str]]:
    child_to_parents: dict[str, set[str]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            ids = ENTITY_RE.findall(line)
            if len(ids) < 2:
                continue
            child, parent = ids[0], ids[1]
            if child != parent:
                child_to_parents.setdefault(child, set()).add(parent)
    return child_to_parents


class AncestorResolver:
    def __init__(self, child_to_parents: Mapping[str, set[str]]) -> None:
        self.child_to_parents = child_to_parents
        self.cache: dict[str, set[str]] = {}

    def ancestors(self, node_id: str) -> set[str]:
        cached = self.cache.get(node_id)
        if cached is not None:
            return cached

        result: set[str] = set()
        frontier = {node_id}
        visited = {node_id}
        while frontier:
            next_frontier: set[str] = set()
            for child in frontier:
                for parent in self.child_to_parents.get(child, set()):
                    if parent in visited:
                        continue
                    visited.add(parent)
                    result.add(parent)
                    next_frontier.add(parent)
            if not next_frontier:
                break
            frontier = next_frontier
        self.cache[node_id] = result
        return result


def shortest_parent_paths(node_id: str, child_to_parents: Mapping[str, set[str]]) -> dict[str, list[str]]:
    paths: dict[str, list[str]] = {node_id: [node_id]}
    frontier = [node_id]
    for child in frontier:
        for parent in sorted(child_to_parents.get(child, set())):
            if parent in paths:
                continue
            paths[parent] = [*paths[child], parent]
            frontier.append(parent)
    return paths


def minimal_connector_paths(
    direct_fos_ids: Sequence[str],
    child_to_parents: Mapping[str, set[str]],
) -> dict[str, list[str]]:
    numeric_ids = [node_id for node_id in direct_fos_ids if not node_id.startswith("slug::")]
    if not numeric_ids:
        return {}
    path_maps = {node_id: shortest_parent_paths(node_id, child_to_parents) for node_id in numeric_ids}
    direct_sets = {node_id: set(paths) for node_id, paths in path_maps.items()}
    neighbors: dict[str, set[str]] = {node_id: set() for node_id in numeric_ids}
    for idx, left in enumerate(numeric_ids):
        for right in numeric_ids[idx + 1 :]:
            if direct_sets[left] & direct_sets[right]:
                neighbors[left].add(right)
                neighbors[right].add(left)

    components: list[list[str]] = []
    seen: set[str] = set()
    for node_id in numeric_ids:
        if node_id in seen:
            continue
        stack = [node_id]
        seen.add(node_id)
        component: list[str] = []
        while stack:
            current = stack.pop()
            component.append(current)
            for neighbor in sorted(neighbors[current]):
                if neighbor in seen:
                    continue
                seen.add(neighbor)
                stack.append(neighbor)
        components.append(sorted(component))

    selected: dict[str, list[str]] = {}

    def candidate_key(candidate: str, component: Sequence[str]) -> tuple[int, int, str]:
        selected_paths = [path_maps[node_id][candidate] for node_id in component]
        union_nodes = {path_node for path in selected_paths for path_node in path}
        total_hops = sum(len(path) - 1 for path in selected_paths)
        return (len(union_nodes), total_hops, candidate)

    for component in components:
        if len(component) == 1:
            selected[component[0]] = [component[0]]
            continue
        common_ancestors: set[str] | None = None
        for node_id in component:
            nodes = direct_sets[node_id]
            common_ancestors = nodes if common_ancestors is None else common_ancestors & nodes
        if common_ancestors:
            best_connector = min(common_ancestors, key=lambda candidate: candidate_key(candidate, component))
            for node_id in component:
                selected[node_id] = path_maps[node_id][best_connector]
            continue

        coverage: dict[str, set[str]] = {}
        component_set = set(component)
        for node_id in component:
            for candidate in path_maps[node_id]:
                coverage.setdefault(candidate, set()).add(node_id)
        shared_candidates = {
            candidate
            for candidate, covered in coverage.items()
            if len(covered & component_set) >= 2
        }
        for node_id in component:
            local_candidates = shared_candidates & set(path_maps[node_id])
            if not local_candidates:
                selected[node_id] = [node_id]
                continue
            best_connector = min(
                local_candidates,
                key=lambda candidate: (
                    len(path_maps[node_id][candidate]) - 1,
                    -len(coverage[candidate] & component_set),
                    candidate,
                ),
            )
            selected[node_id] = path_maps[node_id][best_connector]

    return selected


class MinimalTreeResolver:
    def __init__(self, child_to_parents: Mapping[str, set[str]]) -> None:
        self.child_to_parents = child_to_parents
        self.cache: dict[tuple[str, ...], dict[str, list[str]]] = {}

    def paths(self, direct_node_ids: Sequence[str]) -> dict[str, list[str]]:
        key = tuple(sorted(set(direct_node_ids)))
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        result = minimal_connector_paths(key, self.child_to_parents)
        self.cache[key] = result
        return result


def load_paper_direct_node_weights(path: Path, history_id_to_idx: Mapping[str, int]) -> dict[str, dict[str, float]]:
    paper_to_nodes: dict[str, dict[str, float]] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            paper_id = str(row.get("paper_id") or "").strip()
            node_id = str(row.get("fos_id") or row.get("node_id") or "").strip()
            if not paper_id or not node_id or paper_id not in history_id_to_idx:
                continue
            try:
                weight = float(row.get("weight", 0.0))
            except (TypeError, ValueError):
                weight = 0.0
            if weight > 0.0:
                paper_to_nodes.setdefault(paper_id, {})[node_id] = (
                    paper_to_nodes.setdefault(paper_id, {}).get(node_id, 0.0) + weight
                )
    return paper_to_nodes


def load_root_queries(ids_tsv: Path, embeddings_npy: Path) -> dict[str, np.ndarray]:
    ids = read_ids_tsv(ids_tsv)
    embeddings = normalize_rows(np.load(embeddings_npy))
    if len(ids) != int(embeddings.shape[0]):
        raise ValueError(f"root ids and embeddings row mismatch: {ids_tsv} {embeddings_npy}")
    return {paper_id: embeddings[idx] for idx, paper_id in enumerate(ids)}


def load_author_history_indices_for_candidates(
    author_papers_tsv: Path,
    history_id_to_idx: Mapping[str, int],
    candidate_author_ids: set[str],
) -> dict[str, set[int]]:
    author_indices: dict[str, set[int]] = {}
    with author_papers_tsv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            author_id = str(row.get("author_id") or "").strip()
            if author_id not in candidate_author_ids:
                continue
            paper_id = str(row.get("history_paper_id") or "").strip()
            history_idx = history_id_to_idx.get(paper_id)
            if history_idx is not None:
                author_indices.setdefault(author_id, set()).add(history_idx)
    return author_indices


def build_author_node_indices(
    paths: Stage1Paths,
    history_id_to_idx: Mapping[str, int],
    candidate_author_ids: set[str],
    needed_node_ids: set[str],
    link_mode: str,
    node_weight_mode: str,
    ancestor_resolver: AncestorResolver | MinimalTreeResolver | None,
) -> tuple[dict[str, dict[str, dict[int, float]]], dict[str, int]]:
    paper_direct_nodes = load_paper_direct_node_weights(paths.history_paper_fos_weights_tsv, history_id_to_idx)
    paper_expanded_cache: dict[str, dict[str, float]] = {}
    node_to_author_indices: dict[str, dict[str, dict[int, float]]] = {}
    stats = {
        "history_papers_with_direct_fos": len(paper_direct_nodes),
        "author_history_rows_seen": 0,
        "author_history_rows_used": 0,
        "paper_node_links_used": 0,
        "author_node_pairs": 0,
    }

    def expanded_nodes_for_paper(paper_id: str) -> dict[str, float]:
        cached = paper_expanded_cache.get(paper_id)
        if cached is not None:
            return cached
        direct_nodes = paper_direct_nodes.get(paper_id, {})
        expanded: dict[str, float] = {}
        if link_mode != "minimal_tree":
            for node_id, weight in direct_nodes.items():
                expanded[node_id] = expanded.get(node_id, 0.0) + weight
        if link_mode == "ancestor" and ancestor_resolver is not None:
            for node_id, weight in direct_nodes.items():
                if not isinstance(ancestor_resolver, AncestorResolver):
                    continue
                for ancestor_id in ancestor_resolver.ancestors(node_id):
                    expanded[ancestor_id] = expanded.get(ancestor_id, 0.0) + weight
        if link_mode == "minimal_tree" and ancestor_resolver is not None:
            if isinstance(ancestor_resolver, MinimalTreeResolver):
                for direct_node_id, path in ancestor_resolver.paths(list(direct_nodes)).items():
                    weight = direct_nodes.get(direct_node_id, 0.0)
                    for tree_node_id in path:
                        expanded[tree_node_id] = expanded.get(tree_node_id, 0.0) + weight
        relevant = {
            node_id: (1.0 if node_weight_mode == "unweighted" else weight)
            for node_id, weight in expanded.items()
            if node_id in needed_node_ids and weight > 0.0
        }
        paper_expanded_cache[paper_id] = relevant
        return relevant

    with paths.history_author_papers_tsv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            stats["author_history_rows_seen"] += 1
            author_id = str(row.get("author_id") or "").strip()
            if author_id not in candidate_author_ids:
                continue
            paper_id = str(row.get("history_paper_id") or "").strip()
            history_idx = history_id_to_idx.get(paper_id)
            if history_idx is None:
                continue
            node_weights = expanded_nodes_for_paper(paper_id)
            if not node_weights:
                continue
            stats["author_history_rows_used"] += 1
            stats["paper_node_links_used"] += len(node_weights)
            for node_id, weight in node_weights.items():
                index_weights = node_to_author_indices.setdefault(node_id, {}).setdefault(author_id, {})
                index_weights[history_idx] = index_weights.get(history_idx, 0.0) + weight

    stats["author_node_pairs"] = sum(len(author_map) for author_map in node_to_author_indices.values())
    return node_to_author_indices, stats


def write_tsv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


class UntrainedTaxonomyAssigner:
    def __init__(
        self,
        paths: Stage1Paths,
        tasks: Sequence[Stage1Task],
        link_mode: str,
        node_weight_mode: str,
        ancestor_resolver: AncestorResolver | None,
        include_root_node: bool,
        root_queries: dict[str, np.ndarray] | None,
    ) -> None:
        self.paths = paths
        self.tasks = list(tasks)
        self.author_ids = sorted({author_id for task in tasks for author_id in task.author_ids})
        self.author_to_col = {author_id: idx for idx, author_id in enumerate(self.author_ids)}
        self.role_ids = read_ids_tsv(paths.role_ids_tsv)
        self.role_id_to_idx = {role_id: idx for idx, role_id in enumerate(self.role_ids)}
        self.history_ids = read_ids_tsv(paths.history_ids_tsv)
        self.history_id_to_idx = {paper_id: idx for idx, paper_id in enumerate(self.history_ids)}
        self.role_embeddings = normalize_rows(np.load(paths.role_embeddings_npy))
        self.history_embeddings = np.load(paths.history_embeddings_npy).astype(np.float32)
        self.node_weight_mode = node_weight_mode
        self.include_root_node = include_root_node
        self.root_queries = root_queries or {}
        needed_node_ids = {node_id for task in tasks for node_id in task.node_ids}
        self.node_to_author_indices, self.build_stats = build_author_node_indices(
            paths=paths,
            history_id_to_idx=self.history_id_to_idx,
            candidate_author_ids=set(self.author_ids),
            needed_node_ids=needed_node_ids,
            link_mode=link_mode,
            node_weight_mode=node_weight_mode,
            ancestor_resolver=ancestor_resolver,
        )
        self.author_root_indices = load_author_history_indices_for_candidates(
            paths.history_author_papers_tsv,
            self.history_id_to_idx,
            set(self.author_ids),
        )
        self.root_matrix_cache: tuple[np.ndarray, np.ndarray] | None = None
        self.node_matrix_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._validate()

    @property
    def input_dim(self) -> int:
        return int(self.role_embeddings.shape[1])

    def _validate(self) -> None:
        if len(self.role_ids) != int(self.role_embeddings.shape[0]):
            raise ValueError("role ids and role embedding rows do not match")
        if len(self.history_ids) != int(self.history_embeddings.shape[0]):
            raise ValueError("history ids and history embedding rows do not match")
        missing_roles = [
            role_id
            for task in self.tasks
            for role_id in task.role_record_ids
            if role_id not in self.role_id_to_idx
        ]
        if missing_roles:
            raise ValueError(f"missing role embeddings for {len(missing_roles)} role ids")

    def role_query(self, role_record_id: str) -> np.ndarray:
        return self.role_embeddings[self.role_id_to_idx[role_record_id]]

    def root_query(self, paper_id: str) -> np.ndarray | None:
        return self.root_queries.get(paper_id)

    def root_author_matrix(self) -> tuple[np.ndarray, np.ndarray]:
        if self.root_matrix_cache is not None:
            return self.root_matrix_cache
        col_indices: list[int] = []
        vectors: list[np.ndarray] = []
        for author_id, index_set in self.author_root_indices.items():
            col_idx = self.author_to_col.get(author_id)
            if col_idx is None or not index_set:
                continue
            indices = sorted(index_set)
            mean_vector = self.history_embeddings[indices].mean(axis=0)
            vectors.append(normalize_vector(mean_vector))
            col_indices.append(col_idx)
        if vectors:
            cols = np.asarray(col_indices, dtype=np.int32)
            matrix = np.vstack(vectors).astype(np.float32, copy=False)
        else:
            cols = np.zeros((0,), dtype=np.int32)
            matrix = np.zeros((0, self.input_dim), dtype=np.float32)
        self.root_matrix_cache = (cols, matrix)
        return self.root_matrix_cache

    def node_author_matrix(self, node_id: str) -> tuple[np.ndarray, np.ndarray]:
        if node_id == "__root__":
            return self.root_author_matrix()
        cached = self.node_matrix_cache.get(node_id)
        if cached is not None:
            return cached
        author_map = self.node_to_author_indices.get(node_id, {})
        col_indices: list[int] = []
        vectors: list[np.ndarray] = []
        for author_id, index_weights in author_map.items():
            col_idx = self.author_to_col.get(author_id)
            if col_idx is None or not index_weights:
                continue
            indices = sorted(index_weights)
            weights = np.asarray([index_weights[idx] for idx in indices], dtype=np.float32)
            weight_sum = float(weights.sum())
            if weight_sum <= 0.0:
                continue
            mean_vector = (self.history_embeddings[indices] * weights[:, None]).sum(axis=0) / weight_sum
            vectors.append(normalize_vector(mean_vector))
            col_indices.append(col_idx)
        if vectors:
            cols = np.asarray(col_indices, dtype=np.int32)
            matrix = np.vstack(vectors).astype(np.float32, copy=False)
        else:
            cols = np.zeros((0,), dtype=np.int32)
            matrix = np.zeros((0, self.input_dim), dtype=np.float32)
        self.node_matrix_cache[node_id] = (cols, matrix)
        return cols, matrix

    def assign_task_hungarian(self, task: Stage1Task) -> list[dict[str, object]]:
        rows = list(zip(task.role_record_ids, task.node_ids, task.node_names))
        if self.include_root_node:
            rows = [("__root__", "__root__", "Root paper abstract")] + rows
        score_matrix = np.full((len(rows), len(self.author_ids)), -1.0e9, dtype=np.float32)
        valid_counts = np.zeros(len(rows), dtype=np.int32)

        for row_idx, (role_id, node_id, _) in enumerate(rows):
            q = self.root_query(task.paper_id) if node_id == "__root__" else self.role_query(role_id)
            if q is None:
                continue
            cols, expert_matrix = self.node_author_matrix(node_id)
            if len(cols) == 0:
                continue
            scores = expert_matrix @ q
            score_matrix[row_idx, cols] = scores
            valid_counts[row_idx] = len(cols)

        valid_rows = np.flatnonzero(valid_counts > 0)
        selected_by_row: dict[int, int] = {}
        if len(valid_rows) > 0:
            sub_scores = score_matrix[valid_rows]
            row_ind, col_ind = linear_sum_assignment(sub_scores, maximize=True)
            selected_by_row = {int(valid_rows[int(row)]): int(col) for row, col in zip(row_ind, col_ind)}

        gold = set(task.author_ids)
        assignments: list[dict[str, object]] = []
        for row_idx, (role_id, node_id, node_name) in enumerate(rows):
            col_idx = selected_by_row.get(row_idx)
            if col_idx is None or score_matrix[row_idx, col_idx] <= -1.0e8:
                assignments.append(
                    {
                        "paper_id": task.paper_id,
                        "role_record_id": role_id,
                        "node_id": node_id,
                        "node_name": node_name,
                        "pred_author_id": "",
                        "score": "",
                        "valid_candidates": int(valid_counts[row_idx]),
                        "is_gold": 0,
                    }
                )
                continue
            author_id = self.author_ids[col_idx]
            assignments.append(
                {
                    "paper_id": task.paper_id,
                    "role_record_id": role_id,
                    "node_id": node_id,
                    "node_name": node_name,
                    "pred_author_id": author_id,
                    "score": f"{float(score_matrix[row_idx, col_idx]):.8g}",
                    "valid_candidates": int(valid_counts[row_idx]),
                    "is_gold": int(author_id in gold),
                }
            )
        return assignments


def evaluate_split(
    split_name: str,
    paths: Stage1Paths,
    tasks: Sequence[Stage1Task],
    out_dir: Path,
    link_mode: str,
    node_weight_mode: str,
    ancestor_resolver: AncestorResolver | None,
    include_root_node: bool,
    root_queries: dict[str, np.ndarray] | None,
) -> dict[str, float | int | str]:
    split_dir = out_dir / split_name
    split_dir.mkdir(parents=True, exist_ok=True)
    assigner = UntrainedTaxonomyAssigner(
        paths,
        tasks,
        link_mode,
        node_weight_mode,
        ancestor_resolver,
        include_root_node,
        root_queries,
    )

    node_rows: list[dict[str, object]] = []
    task_rows: list[dict[str, object]] = []
    macro_precision_values: list[float] = []
    macro_recall_values: list[float] = []
    macro_f1_values: list[float] = []
    raw_precision_values: list[float] = []

    total_pred = 0
    total_gold = 0
    total_hits = 0
    raw_total = 0
    raw_hits = 0
    skipped_nodes = 0
    assigned_nodes = 0

    for task in tasks:
        gold = set(task.author_ids)
        predictions: list[str] = []
        raw_task_hits = 0
        task_assignments = assigner.assign_task_hungarian(task)

        for assignment_row in task_assignments:
            pred_author = str(assignment_row["pred_author_id"])
            if not pred_author:
                skipped_nodes += 1
                node_rows.append(assignment_row)
                continue
            is_gold = int(assignment_row["is_gold"])
            predictions.append(pred_author)
            raw_task_hits += is_gold
            raw_hits += is_gold
            raw_total += 1
            assigned_nodes += 1
            node_rows.append(assignment_row)

        pred_set = set(predictions)
        hits = len(pred_set & gold)
        precision = safe_div(hits, len(pred_set))
        recall = safe_div(hits, len(gold))
        f1 = safe_div(2.0 * precision * recall, precision + recall)
        raw_precision = safe_div(raw_task_hits, len(predictions))
        macro_precision_values.append(precision)
        macro_recall_values.append(recall)
        macro_f1_values.append(f1)
        raw_precision_values.append(raw_precision)
        total_pred += len(pred_set)
        total_gold += len(gold)
        total_hits += hits

        task_rows.append(
            {
                "paper_id": task.paper_id,
                "gold_team_size": len(gold),
                "assigned_nodes": len(predictions),
                "skipped_nodes": (len(task.role_record_ids) + int(include_root_node)) - len(predictions),
                "pred_team_size_dedup": len(pred_set),
                "hits_dedup": hits,
                "precision_dedup": f"{precision:.8g}",
                "recall_dedup": f"{recall:.8g}",
                "f1_dedup": f"{f1:.8g}",
                "raw_node_assignment_precision": f"{raw_precision:.8g}",
                "gold_author_ids": ",".join(task.author_ids),
                "pred_author_ids_raw": ",".join(predictions),
                "pred_author_ids_dedup": ",".join(sorted(pred_set)),
            }
        )

    write_tsv(split_dir / "node_assignments.tsv", node_rows)
    write_tsv(split_dir / "per_task_metrics.tsv", task_rows)
    micro_precision = safe_div(total_hits, total_pred)
    micro_recall = safe_div(total_hits, total_gold)
    micro_f1 = safe_div(2.0 * micro_precision * micro_recall, micro_precision + micro_recall)
    metrics: dict[str, float | int | str] = {
        "split": split_name,
        "method": "untrained_role_to_expert_node_mean",
        "link_mode": link_mode,
        "node_weight_mode": node_weight_mode,
        "ancestor_depth": (
            "full_to_root"
            if link_mode == "ancestor"
            else ("minimal_tree" if link_mode == "minimal_tree" else "direct_only")
        ),
        "assignment_mode": "hungarian",
        "include_root_node": int(include_root_node),
        "papers": len(tasks),
        "candidate_authors": len(assigner.author_ids),
        "task_nodes": sum(len(task.node_ids) for task in tasks),
        "assignment_rows": sum(len(task.node_ids) + int(include_root_node) for task in tasks),
        "queried_unique_nodes": len({node_id for task in tasks for node_id in task.node_ids}),
        "expert_nodes_with_vectors": int(assigner.build_stats["author_node_pairs"]),
        "history_papers_with_direct_fos": int(assigner.build_stats["history_papers_with_direct_fos"]),
        "author_history_rows_used": int(assigner.build_stats["author_history_rows_used"]),
        "paper_node_links_used": int(assigner.build_stats["paper_node_links_used"]),
        "assigned_nodes": assigned_nodes,
        "skipped_nodes_no_candidate": skipped_nodes,
        "raw_node_assignment_precision": safe_div(raw_hits, raw_total),
        "macro_precision_dedup": float(np.mean(macro_precision_values)) if macro_precision_values else 0.0,
        "macro_recall_dedup": float(np.mean(macro_recall_values)) if macro_recall_values else 0.0,
        "macro_f1_dedup": float(np.mean(macro_f1_values)) if macro_f1_values else 0.0,
        "micro_precision_dedup": micro_precision,
        "micro_recall_dedup": micro_recall,
        "micro_f1_dedup": micro_f1,
        "total_pred_dedup": total_pred,
        "total_gold": total_gold,
        "total_hits_dedup": total_hits,
    }
    write_tsv(split_dir / "metrics_summary.tsv", [metrics])
    return metrics


def main() -> None:
    args = parse_args()
    if args.link_mode == "direct":
        suffix = f"direct_{args.node_weight_mode}"
    elif args.link_mode == "ancestor":
        suffix = f"ancestor_{args.node_weight_mode}_full_to_root"
    else:
        suffix = f"minimal_tree_{args.node_weight_mode}"
    if args.include_root_node:
        suffix = f"{suffix}_with_root"
    out_dir = Path(args.out_dir) / suffix
    out_dir.mkdir(parents=True, exist_ok=True)

    split_paths: dict[str, Stage1Paths] = {}
    if args.include_dev and not args.skip_dev:
        split_paths["dev"] = override_paths(
            canonical_dev_paths(),
            sample_jsonl=args.dev_sample_jsonl,
            role_descriptions_jsonl=args.dev_role_descriptions_jsonl,
            role_ids_tsv=args.dev_role_ids_tsv,
            role_embeddings_npy=args.dev_role_embeddings,
        )
    if args.include_official_test and not args.skip_official_test:
        split_paths["official_test"] = override_paths(
            canonical_official_test_paths(),
            sample_jsonl=args.official_test_sample_jsonl,
            role_descriptions_jsonl=args.official_test_role_descriptions_jsonl,
            role_ids_tsv=args.official_test_role_ids_tsv,
            role_embeddings_npy=args.official_test_role_embeddings,
        )
    if not split_paths:
        raise SystemExit("No splits selected")

    child_to_parents: dict[str, set[str]] = {}
    ancestor_resolver: AncestorResolver | MinimalTreeResolver | None = None
    if args.link_mode in {"ancestor", "minimal_tree"}:
        child_to_parents = load_child_to_parents(Path(args.taxonomy_edges))
        ancestor_resolver = (
            AncestorResolver(child_to_parents)
            if args.link_mode == "ancestor"
            else MinimalTreeResolver(child_to_parents)
        )

    root_query_paths = {
        "dev": (Path(args.dev_root_ids_tsv), Path(args.dev_root_embeddings)),
        "official_test": (Path(args.official_test_root_ids_tsv), Path(args.official_test_root_embeddings)),
    }

    summaries: list[dict[str, float | int | str]] = []
    for split_name, paths in split_paths.items():
        print(f"loading_{split_name}=1", flush=True)
        name_to_fos_id = load_fos_name_to_id(paths.fos_map_tsv)
        tasks = load_tasks(
            paths.sample_jsonl,
            paths.role_descriptions_jsonl,
            name_to_fos_id,
            max_papers=args.max_papers,
        )
        print(f"evaluating_{split_name}=1 tasks={len(tasks)}", flush=True)
        root_queries = None
        if args.include_root_node:
            ids_tsv, embeddings_npy = root_query_paths[split_name]
            root_queries = load_root_queries(ids_tsv, embeddings_npy)
        metrics = evaluate_split(
            split_name=split_name,
            paths=paths,
            tasks=tasks,
            out_dir=out_dir,
            link_mode=args.link_mode,
            node_weight_mode=args.node_weight_mode,
            ancestor_resolver=ancestor_resolver,
            include_root_node=args.include_root_node,
            root_queries=root_queries,
        )
        summaries.append(metrics)
        print(json.dumps(metrics, ensure_ascii=False), flush=True)

    write_tsv(out_dir / "metrics_summary.tsv", summaries)
    (out_dir / "run_config.json").write_text(
        json.dumps(
            {
                "out_dir": str(out_dir),
                "link_mode": args.link_mode,
                "node_weight_mode": args.node_weight_mode,
                "ancestor_depth": (
                    "full_to_root"
                    if args.link_mode == "ancestor"
                    else ("minimal_tree" if args.link_mode == "minimal_tree" else "direct_only")
                ),
                "include_root_node": args.include_root_node,
                "taxonomy_edges": args.taxonomy_edges,
                "taxonomy_child_count": len(child_to_parents),
                "root_query_paths": {
                    name: {"ids_tsv": str(paths[0]), "embeddings_npy": str(paths[1])}
                    for name, paths in root_query_paths.items()
                },
                "max_papers": args.max_papers,
                "splits": {
                    name: {key: str(value) for key, value in asdict(paths).items()}
                    for name, paths in split_paths.items()
                },
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
