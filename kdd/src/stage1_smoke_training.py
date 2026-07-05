"""Minimal Stage-1 training components for the KDD smoke dataset.

The implementation follows the current Stage-1 objective:

- frozen role and history-paper embeddings are loaded from disk;
- alpha(d|i) is built from history-paper FoS weights, optionally expanded to
  the same minimal-tree connector nodes used by the untrained Top-M retrieval
  experiments, with irrelevant history papers hard-masked to zero;
- z_{e|i} is the node-conditioned expert-node embedding: the alpha-weighted
  mean of an author's history-paper embeddings for task node i;
- only the role-side and expert-side projection heads are trained;
- negative weighting supports V1-V4, with V3/V4 driven by the author citation
  graph when supplied.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import re
from collections import OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

ENTITY_RE = re.compile(r"/entity/(\d+)>")


@dataclass(frozen=True)
class Stage1Task:
    paper_id: str
    author_ids: List[str]
    role_record_ids: List[str]
    node_ids: List[str]
    node_names: List[str]


@dataclass(frozen=True)
class Stage1Paths:
    sample_jsonl: Path
    role_descriptions_jsonl: Path
    role_ids_tsv: Path
    role_embeddings_npy: Path
    history_author_papers_tsv: Path
    history_ids_tsv: Path
    history_embeddings_npy: Path
    history_paper_fos_weights_tsv: Path = Path("")
    expert_profile_dir: Path = Path("outputs/expert_profile_cutoffs/pre_2018_for_valid_2018")
    fos_map_tsv: Path = Path("../data/dblp/FieldsOfStudy.txt")


@dataclass(frozen=True)
class TrainConfig:
    projection_dim: int = 256
    tau: float = 0.07
    tau_m: float = 0.1
    alpha_top_k: int = 16
    alpha_eps: float = 1e-8
    node_link_mode: str = "minimal_tree"
    node_weight_mode: str = "unweighted"
    taxonomy_edges: str = "../data/dblp/13.FieldOfStudyChildren.nt"
    negative_mode: str = "v1"
    negative_pool_mode: str = "untrained_topm"
    untrained_negative_top_m: int = 20
    untrained_node_cache_size: int = 128
    global_negative_sample_size: int = 128
    author_node_cache_size: int = 100000
    pi0: float = 0.5
    w_near: float = 0.1
    w_far: float = 1.0
    prox_threshold: float | None = None
    prox_quantile: float = 0.75
    prox_beta: float = 20.0
    prox_bias: float = 1.0
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 8
    epochs: int = 10
    eval_frac: float = 0.1
    seed: int = 13
    device: str = "auto"
    max_papers: int = 0


def iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def read_ids_tsv(path: Path) -> List[str]:
    ids: List[str] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            ids.append(str(row["id"]))
    return ids


def load_fos_name_to_id(path: Path) -> Dict[str, str]:
    name_to_id: Dict[str, str] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                continue
            fos_id = parts[0].strip()
            norm_name = parts[2].strip()
            display_name = parts[3].strip()
            if not fos_id:
                continue
            if norm_name:
                name_to_id[norm_name.lower()] = fos_id
            if display_name:
                name_to_id[display_name.lower()] = fos_id
    return name_to_id


def load_child_to_parents(path: Path) -> Dict[str, set[str]]:
    child_to_parents: Dict[str, set[str]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            ids = ENTITY_RE.findall(line)
            if len(ids) < 2:
                continue
            child, parent = ids[0], ids[1]
            if child != parent:
                child_to_parents.setdefault(child, set()).add(parent)
    return child_to_parents


def shortest_parent_paths(node_id: str, child_to_parents: Mapping[str, set[str]]) -> Dict[str, List[str]]:
    paths: Dict[str, List[str]] = {node_id: [node_id]}
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
) -> Dict[str, List[str]]:
    numeric_ids = [node_id for node_id in direct_fos_ids if not str(node_id).startswith("slug::")]
    if not numeric_ids:
        return {}
    path_maps = {node_id: shortest_parent_paths(node_id, child_to_parents) for node_id in numeric_ids}
    direct_sets = {node_id: set(paths) for node_id, paths in path_maps.items()}
    neighbors: Dict[str, set[str]] = {node_id: set() for node_id in numeric_ids}
    for idx, left in enumerate(numeric_ids):
        for right in numeric_ids[idx + 1 :]:
            if direct_sets[left] & direct_sets[right]:
                neighbors[left].add(right)
                neighbors[right].add(left)

    components: List[List[str]] = []
    seen: set[str] = set()
    for node_id in numeric_ids:
        if node_id in seen:
            continue
        stack = [node_id]
        seen.add(node_id)
        component: List[str] = []
        while stack:
            current = stack.pop()
            component.append(current)
            for neighbor in sorted(neighbors[current]):
                if neighbor in seen:
                    continue
                seen.add(neighbor)
                stack.append(neighbor)
        components.append(sorted(component))

    selected: Dict[str, List[str]] = {}

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

        coverage: Dict[str, set[str]] = {}
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
        self.cache: Dict[tuple[str, ...], Dict[str, List[str]]] = {}

    def paths(self, direct_node_ids: Sequence[str]) -> Dict[str, List[str]]:
        key = tuple(sorted(set(direct_node_ids)))
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        result = minimal_connector_paths(key, self.child_to_parents)
        self.cache[key] = result
        return result


def extract_author_ids(task_row: dict) -> List[str]:
    stage = task_row.get("stage1_sample") or {}
    if isinstance(stage.get("author_ids"), list):
        return [str(x) for x in stage["author_ids"]]
    if isinstance(task_row.get("team_author_ids"), list):
        return [str(x) for x in task_row["team_author_ids"]]
    if isinstance(task_row.get("author_ids"), list):
        return [str(x) for x in task_row["author_ids"]]
    author_ids: List[str] = []
    for member in task_row.get("team_members") or []:
        if isinstance(member, dict) and member.get("author_id") is not None:
            author_ids.append(str(member["author_id"]))
    if author_ids:
        return author_ids
    for author in task_row.get("authors") or []:
        if isinstance(author, dict) and author.get("id") is not None:
            author_ids.append(str(author["id"]))
    return author_ids


def resolve_task_node_id(raw_node_id: object, node_name: object, name_to_fos_id: Mapping[str, str]) -> str:
    node_id = str(raw_node_id or "").strip()
    if node_id.isdigit():
        return node_id
    name = str(node_name or "").strip().lower()
    if name and name in name_to_fos_id:
        return name_to_fos_id[name]
    return node_id


def load_tasks(
    sample_jsonl: Path,
    role_descriptions_jsonl: Path,
    name_to_fos_id: Mapping[str, str],
    max_papers: int = 0,
) -> List[Stage1Task]:
    role_by_paper: Dict[str, List[dict]] = {}
    for row in iter_jsonl(role_descriptions_jsonl):
        paper_id = str(row.get("paper_id") or "")
        if paper_id:
            role_by_paper.setdefault(paper_id, []).append(row)

    tasks: List[Stage1Task] = []
    for row in iter_jsonl(sample_jsonl):
        paper_id = str(row.get("id") or row.get("paper_id") or "")
        if not paper_id or paper_id not in role_by_paper:
            continue
        author_ids = extract_author_ids(row)
        roles = role_by_paper[paper_id]
        if not author_ids or not roles:
            continue
        tasks.append(
            Stage1Task(
                paper_id=paper_id,
                author_ids=author_ids,
                role_record_ids=[str(role["id"]) for role in roles],
                node_ids=[
                    resolve_task_node_id(role.get("node_id"), role.get("node_name"), name_to_fos_id)
                    for role in roles
                ],
                node_names=[str(role.get("node_name") or role["node_id"]) for role in roles],
            )
        )
        if max_papers > 0 and len(tasks) >= max_papers:
            break
    return tasks


def load_author_history_indices(author_papers_tsv: Path, history_id_to_idx: Mapping[str, int]) -> Dict[str, List[int]]:
    author_to_indices: Dict[str, List[int]] = {}
    with author_papers_tsv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            author_id = str(row["author_id"])
            paper_id = str(row["history_paper_id"])
            idx = history_id_to_idx.get(paper_id)
            if idx is not None:
                author_to_indices.setdefault(author_id, []).append(idx)
    return {author_id: sorted(set(indices)) for author_id, indices in author_to_indices.items()}


def load_author_node_history_weights(
    expert_profile_dir: Path,
    author_ids: Iterable[str],
    history_id_to_idx: Mapping[str, int],
) -> Dict[str, Dict[str, List[tuple[int, float]]]]:
    author_node_weights: Dict[str, Dict[str, List[tuple[int, float]]]] = {}
    for author_id in sorted(set(str(author_id) for author_id in author_ids)):
        path = expert_profile_dir / f"{author_id}_direct_fos_nodes.tsv"
        if not path.exists():
            continue
        node_weights: Dict[str, List[tuple[int, float]]] = {}
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                node_id = str(row.get("fos_id") or "").strip()
                if not node_id:
                    continue
                try:
                    details = json.loads(row.get("paper_weight_details") or "[]")
                except json.JSONDecodeError:
                    continue
                weights: List[tuple[int, float]] = []
                for item in details:
                    if not isinstance(item, dict):
                        continue
                    paper_id = str(item.get("paper_id") or "")
                    idx = history_id_to_idx.get(paper_id)
                    if idx is None:
                        continue
                    try:
                        weight = float(item.get("weight", 0.0))
                    except (TypeError, ValueError):
                        weight = 0.0
                    if weight > 0.0:
                        weights.append((idx, weight))
                if weights:
                    node_weights[node_id] = weights
        if node_weights:
            author_node_weights[author_id] = node_weights
    return author_node_weights


def load_paper_fos_weights(path: Path, history_id_to_idx: Mapping[str, int]) -> Dict[str, List[tuple[str, float]]]:
    paper_weights: Dict[str, List[tuple[str, float]]] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            paper_id = str(row.get("paper_id") or "").strip()
            node_id = str(row.get("fos_id") or row.get("node_id") or "").strip()
            if not paper_id or paper_id not in history_id_to_idx or not node_id:
                continue
            try:
                weight = float(row.get("weight", 0.0))
            except (TypeError, ValueError):
                weight = 0.0
            if weight > 0.0:
                paper_weights.setdefault(paper_id, []).append((node_id, weight))
    return paper_weights


def load_author_node_history_weights_from_paper_fos(
    author_papers_tsv: Path,
    history_id_to_idx: Mapping[str, int],
    paper_fos_weights_tsv: Path,
    node_link_mode: str = "minimal_tree",
    node_weight_mode: str = "unweighted",
    taxonomy_edges: Path = Path("../data/dblp/13.FieldOfStudyChildren.nt"),
    needed_node_ids: set[str] | None = None,
) -> Dict[str, Dict[str, List[tuple[int, float]]]]:
    paper_weights = load_paper_fos_weights(paper_fos_weights_tsv, history_id_to_idx)
    node_link_mode = node_link_mode.lower()
    node_weight_mode = node_weight_mode.lower()
    if node_link_mode not in {"direct", "minimal_tree"}:
        raise ValueError(f"unsupported node_link_mode for Stage1: {node_link_mode}")
    if node_weight_mode not in {"unweighted", "weighted"}:
        raise ValueError(f"unsupported node_weight_mode for Stage1: {node_weight_mode}")
    resolver = (
        MinimalTreeResolver(load_child_to_parents(taxonomy_edges))
        if node_link_mode == "minimal_tree"
        else None
    )
    expanded_cache: Dict[str, List[tuple[str, float]]] = {}

    def expanded_weights_for_paper(paper_id: str) -> List[tuple[str, float]]:
        cached = expanded_cache.get(paper_id)
        if cached is not None:
            return cached
        direct_weights = paper_weights.get(paper_id, [])
        expanded: Dict[str, float] = {}
        if node_link_mode == "direct" or resolver is None:
            for node_id, weight in direct_weights:
                expanded[node_id] = expanded.get(node_id, 0.0) + weight
        else:
            direct_by_node: Dict[str, float] = {}
            for node_id, weight in direct_weights:
                direct_by_node[node_id] = direct_by_node.get(node_id, 0.0) + weight
            for direct_node_id, path in resolver.paths(list(direct_by_node)).items():
                direct_weight = direct_by_node.get(direct_node_id, 0.0)
                if direct_weight <= 0.0:
                    continue
                for tree_node_id in path:
                    expanded[tree_node_id] = expanded.get(tree_node_id, 0.0) + direct_weight

        rows = [
            (node_id, 1.0 if node_weight_mode == "unweighted" else weight)
            for node_id, weight in expanded.items()
            if weight > 0.0 and (needed_node_ids is None or node_id in needed_node_ids)
        ]
        expanded_cache[paper_id] = rows
        return rows

    author_node_weights: Dict[str, Dict[str, List[tuple[int, float]]]] = {}
    with author_papers_tsv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            author_id = str(row.get("author_id") or "").strip()
            paper_id = str(row.get("history_paper_id") or "").strip()
            idx = history_id_to_idx.get(paper_id)
            if not author_id or idx is None:
                continue
            for node_id, weight in expanded_weights_for_paper(paper_id):
                author_node_weights.setdefault(author_id, {}).setdefault(node_id, []).append((idx, weight))
    return author_node_weights


def split_tasks(tasks: Sequence[Stage1Task], eval_frac: float, seed: int) -> tuple[List[Stage1Task], List[Stage1Task]]:
    shuffled = list(tasks)
    rng = random.Random(seed)
    rng.shuffle(shuffled)
    if eval_frac <= 0.0 or len(shuffled) < 2:
        return shuffled, []
    eval_size = max(1, int(round(len(shuffled) * eval_frac)))
    return shuffled[eval_size:], shuffled[:eval_size]


def batched(items: Sequence[Stage1Task], batch_size: int) -> Iterable[List[Stage1Task]]:
    for start in range(0, len(items), batch_size):
        yield list(items[start : start + batch_size])


def choose_device(device: str) -> torch.device:
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class Stage1ProjectionModel(nn.Module):
    def __init__(self, input_dim: int, projection_dim: int) -> None:
        super().__init__()
        self.role_proj = nn.Linear(input_dim, projection_dim, bias=False)
        self.expert_proj = nn.Linear(input_dim, projection_dim, bias=False)

    def score(self, role_embeddings: Tensor, expert_embeddings: Tensor) -> Tensor:
        role = F.normalize(self.role_proj(role_embeddings), dim=-1)
        expert = F.normalize(self.expert_proj(expert_embeddings), dim=-1)
        return torch.einsum("mk,mak->ma", role, expert)


class CitationProximityWeights:
    def __init__(self, graph_dir: Path, relevant_author_ids: Iterable[str] | None = None) -> None:
        self.adjacency: Dict[str, Dict[str, float]] = {}
        self.degrees: Dict[str, float] = {}
        relevant = set(str(x) for x in relevant_author_ids) if relevant_author_ids is not None else None
        with (graph_dir / "edges_undirected.tsv").open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                a = str(row["author_id_a"])
                b = str(row["author_id_b"])
                weight = float(row["weight"])
                if weight <= 0.0:
                    continue
                if relevant is None:
                    self.degrees[a] = self.degrees.get(a, 0.0) + weight
                    self.degrees[b] = self.degrees.get(b, 0.0) + weight
                    self.adjacency.setdefault(a, {})[b] = self.adjacency.setdefault(a, {}).get(b, 0.0) + weight
                    self.adjacency.setdefault(b, {})[a] = self.adjacency.setdefault(b, {}).get(a, 0.0) + weight
                    continue
                if a in relevant:
                    self.degrees[a] = self.degrees.get(a, 0.0) + weight
                    if b in relevant:
                        self.adjacency.setdefault(a, {})[b] = self.adjacency.setdefault(a, {}).get(b, 0.0) + weight
                if b in relevant:
                    self.degrees[b] = self.degrees.get(b, 0.0) + weight
                    if a in relevant:
                        self.adjacency.setdefault(b, {})[a] = self.adjacency.setdefault(b, {}).get(a, 0.0) + weight

    def prox(self, author_id: str, team_author_ids: Sequence[str]) -> float:
        neighbors = self.adjacency.get(str(author_id), {})
        raw_value = 0.0
        for team_author_id in set(str(x) for x in team_author_ids):
            if team_author_id == str(author_id):
                continue
            raw_value += float(neighbors.get(team_author_id, 0.0))
        degree = float(self.degrees.get(str(author_id), 0.0))
        if degree <= 0.0:
            return 0.0
        return math.log1p(raw_value / degree)

    def estimate_threshold(
        self,
        tasks: Sequence[Stage1Task],
        candidate_author_ids: Sequence[str],
        quantile: float,
    ) -> float:
        values: List[float] = []
        candidate_set = list(dict.fromkeys(str(x) for x in candidate_author_ids))
        for task in tasks:
            gold = set(task.author_ids)
            for author_id in candidate_set:
                if author_id in gold:
                    continue
                values.append(self.prox(author_id, task.author_ids))
        if not values:
            return 0.0
        positive_values = [value for value in values if value > 0.0]
        if not positive_values:
            return float("inf")
        return float(np.quantile(np.asarray(positive_values, dtype=np.float64), quantile))


class Stage1SmokeDataset:
    def __init__(
        self,
        paths: Stage1Paths,
        max_papers: int = 0,
        node_link_mode: str = "minimal_tree",
        node_weight_mode: str = "unweighted",
        taxonomy_edges: Path = Path("../data/dblp/13.FieldOfStudyChildren.nt"),
    ) -> None:
        self.paths = paths
        self.node_link_mode = node_link_mode
        self.node_weight_mode = node_weight_mode
        self.taxonomy_edges = taxonomy_edges
        self.name_to_fos_id = load_fos_name_to_id(paths.fos_map_tsv)
        self.tasks = load_tasks(
            paths.sample_jsonl,
            paths.role_descriptions_jsonl,
            self.name_to_fos_id,
            max_papers=max_papers,
        )
        self.role_ids = read_ids_tsv(paths.role_ids_tsv)
        self.history_ids = read_ids_tsv(paths.history_ids_tsv)
        self.role_id_to_idx = {role_id: idx for idx, role_id in enumerate(self.role_ids)}
        self.history_id_to_idx = {paper_id: idx for idx, paper_id in enumerate(self.history_ids)}
        self.author_history_indices = load_author_history_indices(
            paths.history_author_papers_tsv,
            self.history_id_to_idx,
        )
        self.role_embeddings = np.load(paths.role_embeddings_npy).astype(np.float32)
        self.history_embeddings = np.load(paths.history_embeddings_npy).astype(np.float32)
        self.author_ids = sorted({author_id for task in self.tasks for author_id in task.author_ids})
        if paths.history_paper_fos_weights_tsv and paths.history_paper_fos_weights_tsv.exists():
            self.author_node_history_weights = load_author_node_history_weights_from_paper_fos(
                paths.history_author_papers_tsv,
                self.history_id_to_idx,
                paths.history_paper_fos_weights_tsv,
                node_link_mode=node_link_mode,
                node_weight_mode=node_weight_mode,
                taxonomy_edges=taxonomy_edges,
                needed_node_ids={node_id for task in self.tasks for node_id in task.node_ids},
            )
        else:
            self.author_node_history_weights = load_author_node_history_weights(
                paths.expert_profile_dir,
                self.author_ids,
                self.history_id_to_idx,
            )
        self._validate()

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
        missing_history_authors = [
            author_id
            for author_id in self.author_ids
            if author_id not in self.author_history_indices
        ]
        if missing_history_authors:
            raise ValueError(f"missing history embeddings for {len(missing_history_authors)} authors")
        missing_profile_authors = [
            author_id
            for author_id in self.author_ids
            if author_id not in self.author_node_history_weights
        ]
        if missing_profile_authors:
            raise ValueError(f"missing expert-node profiles for {len(missing_profile_authors)} authors")

    @property
    def input_dim(self) -> int:
        return int(self.role_embeddings.shape[1])


Stage1Dataset = Stage1SmokeDataset


@dataclass
class BatchMetrics:
    loss: float
    nodes_used: int
    nodes_skipped_no_positive: int
    top1_gold_hits: int
    top1_total: int
    mean_positive_score: float
    mean_negative_score: float


class Stage1LossComputer:
    def __init__(
        self,
        dataset: Stage1SmokeDataset,
        config: TrainConfig,
        device: torch.device,
        citation_weights: CitationProximityWeights | None = None,
    ) -> None:
        self.dataset = dataset
        self.config = config
        self.device = device
        self.citation_weights = citation_weights
        self.role_embeddings = torch.from_numpy(dataset.role_embeddings).to(device)
        self.history_embeddings_np = dataset.history_embeddings
        self.sample_round = 0
        self.untrained_topm_negative_cache: Dict[str, List[str]] = {}
        self.untrained_node_matrix_cache: OrderedDict[str, tuple[List[str], np.ndarray]] = OrderedDict()
        self.author_node_vector_cache: OrderedDict[tuple[str, str], np.ndarray | None] = OrderedDict()

    def role_indices_for_tasks(self, tasks: Sequence[Stage1Task]) -> tuple[List[int], List[str], List[int]]:
        role_indices: List[int] = []
        node_ids: List[str] = []
        task_offsets: List[int] = []
        for task in tasks:
            task_offsets.append(len(role_indices))
            role_indices.extend(self.dataset.role_id_to_idx[role_id] for role_id in task.role_record_ids)
            node_ids.extend(task.node_ids)
        return role_indices, node_ids, task_offsets

    def batch_author_ids(self, tasks: Sequence[Stage1Task]) -> List[str]:
        author_ids: List[str] = []
        seen: set[str] = set()
        for task in tasks:
            for author_id in task.author_ids:
                if author_id not in seen:
                    seen.add(author_id)
                    author_ids.append(author_id)
        return author_ids

    def _sample_seed(self, task: Stage1Task) -> int:
        key = f"{self.config.seed}\t{self.sample_round}\t{task.paper_id}".encode("utf-8")
        return int.from_bytes(hashlib.sha256(key).digest()[:8], byteorder="big", signed=False)

    def sample_global_negative_author_ids(self, task: Stage1Task) -> List[str]:
        gold = set(task.author_ids)
        pool = self.dataset.author_ids
        eligible_count = max(0, len(pool) - len(gold))
        if eligible_count == 0:
            return []
        sample_size = int(self.config.global_negative_sample_size)
        if sample_size <= 0 or sample_size >= eligible_count:
            return [author_id for author_id in pool if author_id not in gold]

        rng = random.Random(self._sample_seed(task))
        if sample_size * 2 >= eligible_count:
            eligible = [author_id for author_id in pool if author_id not in gold]
            rng.shuffle(eligible)
            return eligible[:sample_size]

        selected: List[str] = []
        seen: set[str] = set()
        while len(selected) < sample_size:
            author_id = pool[rng.randrange(len(pool))]
            if author_id in gold or author_id in seen:
                continue
            seen.add(author_id)
            selected.append(author_id)
        return selected

    def untrained_topm_negative_author_ids(self, task: Stage1Task) -> List[str]:
        cached = self.untrained_topm_negative_cache.get(task.paper_id)
        if cached is not None:
            return list(cached)

        top_m = int(self.config.untrained_negative_top_m)
        if top_m <= 0:
            self.untrained_topm_negative_cache[task.paper_id] = []
            return []

        gold = set(task.author_ids)
        selected: List[str] = []
        seen: set[str] = set()
        for role_id, node_id in zip(task.role_record_ids, task.node_ids):
            role_idx = self.dataset.role_id_to_idx[role_id]
            q = self.dataset.role_embeddings[role_idx].astype(np.float32, copy=False)
            q_norm = float(np.linalg.norm(q))
            if q_norm <= 1e-12:
                continue
            q = q / q_norm
            author_ids, expert_matrix = self.untrained_node_author_matrix(node_id)
            if len(author_ids) == 0:
                continue
            scores = expert_matrix @ q
            k = min(top_m, len(scores))
            if len(scores) <= k:
                top_cols = np.argsort(-scores)
            else:
                top_cols = np.argpartition(-scores, k - 1)[:k]
                top_cols = top_cols[np.argsort(-scores[top_cols])]
            for col_idx in top_cols:
                author_id = author_ids[int(col_idx)]
                if author_id in gold or author_id in seen:
                    continue
                seen.add(author_id)
                selected.append(author_id)

        self.untrained_topm_negative_cache[task.paper_id] = selected
        return list(selected)

    def precompute_untrained_topm_negatives(self, tasks: Sequence[Stage1Task]) -> None:
        mode = self.config.negative_pool_mode.lower()
        if mode not in {"untrained_topm", "mixed"}:
            return
        top_m = int(self.config.untrained_negative_top_m)
        if top_m <= 0:
            for task in tasks:
                self.untrained_topm_negative_cache.setdefault(task.paper_id, [])
            return

        pending = [task for task in tasks if task.paper_id not in self.untrained_topm_negative_cache]
        if not pending:
            return

        node_items: Dict[str, List[tuple[str, str, set[str]]]] = {}
        negatives_by_paper: Dict[str, List[str]] = {task.paper_id: [] for task in pending}
        seen_by_paper: Dict[str, set[str]] = {task.paper_id: set() for task in pending}
        for task in pending:
            gold = set(task.author_ids)
            for role_id, node_id in zip(task.role_record_ids, task.node_ids):
                node_items.setdefault(node_id, []).append((task.paper_id, role_id, gold))

        chunk_size = 256
        for node_id, items in node_items.items():
            author_ids, expert_matrix = self.untrained_node_author_matrix(node_id)
            if len(author_ids) == 0:
                continue
            for start in range(0, len(items), chunk_size):
                chunk = items[start : start + chunk_size]
                queries: List[np.ndarray] = []
                kept: List[tuple[str, set[str]]] = []
                for paper_id, role_id, gold in chunk:
                    role_idx = self.dataset.role_id_to_idx[role_id]
                    q = self.dataset.role_embeddings[role_idx].astype(np.float32, copy=False)
                    q_norm = float(np.linalg.norm(q))
                    if q_norm <= 1e-12:
                        continue
                    queries.append((q / q_norm).astype(np.float32, copy=False))
                    kept.append((paper_id, gold))
                if not queries:
                    continue
                query_matrix = np.vstack(queries).astype(np.float32, copy=False)
                scores = expert_matrix @ query_matrix.T
                k = min(top_m, scores.shape[0])
                for col_idx, (paper_id, gold) in enumerate(kept):
                    column = scores[:, col_idx]
                    if len(column) <= k:
                        top_cols = np.argsort(-column)
                    else:
                        top_cols = np.argpartition(-column, k - 1)[:k]
                        top_cols = top_cols[np.argsort(-column[top_cols])]
                    seen = seen_by_paper[paper_id]
                    out = negatives_by_paper[paper_id]
                    for author_col in top_cols:
                        author_id = author_ids[int(author_col)]
                        if author_id in gold or author_id in seen:
                            continue
                        seen.add(author_id)
                        out.append(author_id)

        for task in pending:
            self.untrained_topm_negative_cache[task.paper_id] = negatives_by_paper.get(task.paper_id, [])

    def untrained_node_author_matrix(self, node_id: str) -> tuple[List[str], np.ndarray]:
        cached = self.untrained_node_matrix_cache.get(node_id)
        if cached is not None:
            self.untrained_node_matrix_cache.move_to_end(node_id)
            return cached

        author_ids: List[str] = []
        vectors: List[np.ndarray] = []
        for author_id in self.dataset.author_ids:
            weighted_indices = self.dataset.author_node_history_weights.get(str(author_id), {}).get(str(node_id), [])
            if not weighted_indices:
                continue
            history_indices = [idx for idx, _ in weighted_indices]
            weights_np = np.asarray([weight for _, weight in weighted_indices], dtype=np.float32)
            alpha_sum = float(weights_np.sum())
            if alpha_sum <= self.config.alpha_eps:
                continue
            history_np = self.history_embeddings_np[history_indices]
            vector = (history_np * weights_np[:, None]).sum(axis=0) / alpha_sum
            norm = float(np.linalg.norm(vector))
            if norm <= 1e-12:
                continue
            vectors.append((vector / norm).astype(np.float32, copy=False))
            author_ids.append(author_id)

        if vectors:
            matrix = np.vstack(vectors).astype(np.float32, copy=False)
        else:
            matrix = np.zeros((0, self.dataset.input_dim), dtype=np.float32)
        result = (author_ids, matrix)
        self.untrained_node_matrix_cache[node_id] = result
        max_cache_size = int(self.config.untrained_node_cache_size)
        while max_cache_size > 0 and len(self.untrained_node_matrix_cache) > max_cache_size:
            self.untrained_node_matrix_cache.popitem(last=False)
        if max_cache_size <= 0:
            self.untrained_node_matrix_cache.clear()
        return result

    def candidate_author_ids_for_tasks(self, tasks: Sequence[Stage1Task]) -> tuple[List[str], List[List[str]]]:
        mode = self.config.negative_pool_mode.lower()
        if mode == "batch":
            author_ids = self.batch_author_ids(tasks)
            per_task_negatives = [
                [author_id for author_id in author_ids if author_id not in set(task.author_ids)]
                for task in tasks
            ]
            return author_ids, per_task_negatives
        if mode == "global":
            per_task_negatives = [self.sample_global_negative_author_ids(task) for task in tasks]
        elif mode in {"untrained_topm", "mixed"}:
            per_task_negatives = [self.untrained_topm_negative_author_ids(task) for task in tasks]
            if mode == "mixed":
                for idx, task in enumerate(tasks):
                    existing = set(per_task_negatives[idx]) | set(task.author_ids)
                    for author_id in self.sample_global_negative_author_ids(task):
                        if author_id not in existing:
                            existing.add(author_id)
                            per_task_negatives[idx].append(author_id)
        else:
            raise ValueError(f"unknown negative_pool_mode: {self.config.negative_pool_mode}")

        author_ids: List[str] = []
        seen: set[str] = set()
        for task, negatives in zip(tasks, per_task_negatives):
            for author_id in list(task.author_ids) + negatives:
                if author_id not in seen:
                    seen.add(author_id)
                    author_ids.append(author_id)
        return author_ids, per_task_negatives

    def alpha_weighted_expert_embeddings(self, node_ids: Sequence[str], author_ids: Sequence[str]) -> tuple[Tensor, Tensor]:
        per_author_z: List[Tensor] = []
        per_author_mask: List[Tensor] = []
        for author_id in author_ids:
            node_weights = self.dataset.author_node_history_weights.get(str(author_id), {})
            z = torch.zeros(
                (len(node_ids), self.dataset.input_dim),
                device=self.device,
                dtype=torch.float32,
            )
            mask_values: List[bool] = []
            for row_idx, node_id in enumerate(node_ids):
                cached = self.author_node_vector(author_id, node_id, node_weights)
                if cached is None:
                    mask_values.append(False)
                    continue
                z[row_idx] = torch.from_numpy(cached).to(self.device)
                mask_values.append(True)
            mask = torch.tensor(mask_values, device=self.device, dtype=torch.bool)
            per_author_z.append(z)
            per_author_mask.append(mask)
        return torch.stack(per_author_z, dim=1), torch.stack(per_author_mask, dim=1)

    def author_node_vector(
        self,
        author_id: str,
        node_id: str,
        node_weights: Mapping[str, List[tuple[int, float]]] | None = None,
    ) -> np.ndarray | None:
        key = (str(author_id), str(node_id))
        if key in self.author_node_vector_cache:
            self.author_node_vector_cache.move_to_end(key)
            return self.author_node_vector_cache[key]
        if node_weights is None:
            node_weights = self.dataset.author_node_history_weights.get(str(author_id), {})
        weighted_indices = node_weights.get(str(node_id), [])
        result: np.ndarray | None
        if not weighted_indices:
            result = None
        else:
            history_indices = [idx for idx, _ in weighted_indices]
            weights_np = np.asarray([weight for _, weight in weighted_indices], dtype=np.float32)
            alpha_sum = float(weights_np.sum())
            if alpha_sum <= self.config.alpha_eps:
                result = None
            else:
                history_np = self.history_embeddings_np[history_indices]
                result = ((history_np * weights_np[:, None]).sum(axis=0) / alpha_sum).astype(
                    np.float32,
                    copy=False,
                )
        self.author_node_vector_cache[key] = result
        max_cache_size = int(self.config.author_node_cache_size)
        while max_cache_size > 0 and len(self.author_node_vector_cache) > max_cache_size:
            self.author_node_vector_cache.popitem(last=False)
        if max_cache_size <= 0:
            self.author_node_vector_cache.clear()
        return result

    def negative_weights(
        self,
        task: Stage1Task,
        neg_author_ids: Sequence[str],
    ) -> Tensor:
        mode = self.config.negative_mode.lower()
        if mode == "v1":
            values = [1.0 for _ in neg_author_ids]
        elif mode == "v2":
            values = [self.config.pi0 for _ in neg_author_ids]
        elif mode in {"v3", "v4"}:
            if self.citation_weights is None:
                raise ValueError(f"negative_mode={mode} requires --citation-graph-dir")
            prox_values = [self.citation_weights.prox(author_id, task.author_ids) for author_id in neg_author_ids]
            if mode == "v3":
                if self.config.prox_threshold is None:
                    raise ValueError("negative_mode=v3 requires prox_threshold after estimation")
                values = [
                    self.config.w_near if prox >= self.config.prox_threshold else self.config.w_far
                    for prox in prox_values
                ]
            else:
                values = [
                    1.0 / (1.0 + math.exp(self.config.prox_beta * prox - self.config.prox_bias))
                    for prox in prox_values
                ]
        else:
            raise ValueError(f"unknown negative mode: {self.config.negative_mode}")
        return torch.tensor(values, device=self.device, dtype=torch.float32)

    def compute_batch_loss(
        self,
        model: Stage1ProjectionModel,
        tasks: Sequence[Stage1Task],
    ) -> tuple[Tensor, BatchMetrics]:
        _, per_task_negatives = self.candidate_author_ids_for_tasks(tasks)

        losses: List[Tensor] = []
        pos_scores_for_log: List[Tensor] = []
        neg_scores_for_log: List[Tensor] = []
        nodes_skipped = 0
        top1_hits = 0
        top1_total = 0
        zero = self.role_embeddings.sum() * 0.0

        for task_idx, task in enumerate(tasks):
            neg_author_ids = per_task_negatives[task_idx]
            author_ids: List[str] = []
            seen_authors: set[str] = set()
            for author_id in list(task.author_ids) + neg_author_ids:
                if author_id in seen_authors:
                    continue
                seen_authors.add(author_id)
                author_ids.append(author_id)
            if not author_ids:
                continue
            author_to_col = {author_id: idx for idx, author_id in enumerate(author_ids)}
            role_indices = [
                self.dataset.role_id_to_idx[role_id]
                for role_id in task.role_record_ids
            ]
            q_base = self.role_embeddings[torch.tensor(role_indices, device=self.device, dtype=torch.long)]
            z_base, valid_mask = self.alpha_weighted_expert_embeddings(task.node_ids, author_ids)
            scores = model.score(q_base, z_base)
            gold_cols = torch.tensor(
                [author_to_col[author_id] for author_id in task.author_ids if author_id in author_to_col],
                device=self.device,
                dtype=torch.long,
            )
            neg_cols = torch.tensor(
                [author_to_col[author_id] for author_id in neg_author_ids],
                device=self.device,
                dtype=torch.long,
            )
            neg_weights = self.negative_weights(task, neg_author_ids) if neg_author_ids else torch.empty(0, device=self.device)

            for row in range(len(task.node_ids)):
                pos_valid = valid_mask[row, gold_cols]
                if not bool(pos_valid.any()):
                    nodes_skipped += 1
                    continue
                pos_s = scores[row, gold_cols][pos_valid]
                s_plus = self.config.tau_m * torch.logsumexp(pos_s / self.config.tau_m, dim=0)

                if len(neg_cols) > 0:
                    neg_valid = valid_mask[row, neg_cols]
                    neg_s = scores[row, neg_cols][neg_valid]
                    weights = neg_weights[neg_valid]
                else:
                    neg_s = torch.empty(0, device=self.device)
                    weights = torch.empty(0, device=self.device)

                terms = [s_plus / self.config.tau]
                if neg_s.numel() > 0:
                    terms.append(neg_s / self.config.tau + torch.log(torch.clamp(weights, min=1e-8)))
                    neg_scores_for_log.append(neg_s.detach())
                denominator = torch.logsumexp(torch.cat([term.reshape(-1) for term in terms]), dim=0)
                losses.append(-(s_plus / self.config.tau - denominator))
                pos_scores_for_log.append(pos_s.detach())

                candidate_cols = torch.cat([gold_cols, neg_cols]) if len(neg_cols) > 0 else gold_cols
                candidate_valid = valid_mask[row, candidate_cols]
                if bool(candidate_valid.any()):
                    candidate_scores = scores[row, candidate_cols].masked_fill(~candidate_valid, -1e9)
                    top_col = int(candidate_cols[int(torch.argmax(candidate_scores).detach().cpu())].detach().cpu())
                    top1_hits += int(author_ids[top_col] in set(task.author_ids))
                    top1_total += 1

        if not losses:
            loss = zero
        else:
            loss = torch.stack(losses).mean()

        mean_pos = (
            float(torch.cat(pos_scores_for_log).mean().detach().cpu()) if pos_scores_for_log else float("nan")
        )
        mean_neg = (
            float(torch.cat(neg_scores_for_log).mean().detach().cpu()) if neg_scores_for_log else float("nan")
        )
        return loss, BatchMetrics(
            loss=float(loss.detach().cpu()),
            nodes_used=len(losses),
            nodes_skipped_no_positive=nodes_skipped,
            top1_gold_hits=top1_hits,
            top1_total=top1_total,
            mean_positive_score=mean_pos,
            mean_negative_score=mean_neg,
        )


def aggregate_metrics(metrics: Sequence[BatchMetrics]) -> dict:
    total_nodes = sum(m.nodes_used for m in metrics)
    total_skipped = sum(m.nodes_skipped_no_positive for m in metrics)
    top1_total = sum(m.top1_total for m in metrics)
    weighted_loss = sum(m.loss * m.nodes_used for m in metrics)
    pos_values = [m.mean_positive_score for m in metrics if not math.isnan(m.mean_positive_score)]
    neg_values = [m.mean_negative_score for m in metrics if not math.isnan(m.mean_negative_score)]
    return {
        "loss": weighted_loss / total_nodes if total_nodes else float("nan"),
        "nodes_used": total_nodes,
        "nodes_skipped_no_positive": total_skipped,
        "top1_gold_accuracy": sum(m.top1_gold_hits for m in metrics) / top1_total if top1_total else float("nan"),
        "top1_gold_hits": sum(m.top1_gold_hits for m in metrics),
        "top1_total": top1_total,
        "mean_positive_score": float(np.mean(pos_values)) if pos_values else float("nan"),
        "mean_negative_score": float(np.mean(neg_values)) if neg_values else float("nan"),
    }


def evaluate_stage1_dataset(
    model: Stage1ProjectionModel,
    dataset: Stage1SmokeDataset,
    config: TrainConfig,
    device: torch.device,
    citation_graph_dir: Path | None = None,
) -> dict:
    citation_weights = (
        CitationProximityWeights(citation_graph_dir, relevant_author_ids=dataset.author_ids)
        if citation_graph_dir
        else None
    )
    loss_computer = Stage1LossComputer(dataset, config, device, citation_weights)
    loss_computer.precompute_untrained_topm_negatives(dataset.tasks)
    model.eval()
    batch_metrics: List[BatchMetrics] = []
    with torch.no_grad():
        for batch in batched(dataset.tasks, config.batch_size):
            _, metrics = loss_computer.compute_batch_loss(model, batch)
            batch_metrics.append(metrics)
    return aggregate_metrics(batch_metrics)


def train_stage1_splits(
    train_dataset: Stage1SmokeDataset,
    eval_datasets: Mapping[str, Stage1SmokeDataset],
    config: TrainConfig,
    out_dir: Path,
    citation_graph_dir: Path | None = None,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    device = choose_device(config.device)
    if config.eval_frac != 0.0:
        config = TrainConfig(**{**asdict(config), "eval_frac": 0.0})

    for split_name, dataset in eval_datasets.items():
        if dataset.input_dim != train_dataset.input_dim:
            raise ValueError(
                f"{split_name} input_dim={dataset.input_dim} does not match "
                f"train input_dim={train_dataset.input_dim}"
            )

    train_tasks = list(train_dataset.tasks)
    citation_weights = (
        CitationProximityWeights(citation_graph_dir, relevant_author_ids=train_dataset.author_ids)
        if citation_graph_dir
        else None
    )
    if config.negative_mode.lower() == "v3" and config.prox_threshold is None:
        if citation_weights is None:
            raise ValueError("negative_mode=v3 requires --citation-graph-dir")
        threshold = citation_weights.estimate_threshold(
            train_tasks,
            train_dataset.author_ids,
            config.prox_quantile,
        )
        config = TrainConfig(**{**asdict(config), "prox_threshold": threshold})

    model = Stage1ProjectionModel(train_dataset.input_dim, config.projection_dim).to(device)
    loss_computer = Stage1LossComputer(train_dataset, config, device, citation_weights)
    loss_computer.precompute_untrained_topm_negatives(train_tasks)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    config_payload = {
        "config": asdict(config),
        "train_paths": {key: str(value) for key, value in asdict(train_dataset.paths).items()},
        "eval_paths": {
            name: {key: str(value) for key, value in asdict(dataset.paths).items()}
            for name, dataset in eval_datasets.items()
        },
        "num_train_tasks": len(train_dataset.tasks),
        "num_eval_tasks": {name: len(dataset.tasks) for name, dataset in eval_datasets.items()},
        "num_train_authors": len(train_dataset.author_ids),
        "num_eval_authors": {name: len(dataset.author_ids) for name, dataset in eval_datasets.items()},
        "input_dim": train_dataset.input_dim,
        "device": str(device),
        "citation_graph_dir": str(citation_graph_dir) if citation_graph_dir else "",
    }
    (out_dir / "config.json").write_text(
        json.dumps(config_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    log_path = out_dir / "train_log.tsv"
    final_metrics: dict[str, dict] = {}
    with log_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "epoch",
            "split",
            "loss",
            "nodes_used",
            "nodes_skipped_no_positive",
            "top1_gold_accuracy",
            "top1_gold_hits",
            "top1_total",
            "mean_positive_score",
            "mean_negative_score",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()

        for epoch in range(1, config.epochs + 1):
            random.Random(config.seed + epoch).shuffle(train_tasks)
            model.train()
            loss_computer.sample_round = epoch
            train_batch_metrics: List[BatchMetrics] = []
            for batch in batched(train_tasks, config.batch_size):
                optimizer.zero_grad(set_to_none=True)
                loss, metrics = loss_computer.compute_batch_loss(model, batch)
                loss.backward()
                optimizer.step()
                train_batch_metrics.append(metrics)
            train_metrics = aggregate_metrics(train_batch_metrics)
            final_metrics["train"] = train_metrics
            writer.writerow({"epoch": epoch, "split": "train", **train_metrics})

            print(
                "epoch={epoch} train_loss={loss:.6f} train_top1={top1:.4f} "
                "nodes={nodes} skipped={skipped}".format(
                    epoch=epoch,
                    loss=train_metrics["loss"],
                    top1=train_metrics["top1_gold_accuracy"],
                    nodes=train_metrics["nodes_used"],
                    skipped=train_metrics["nodes_skipped_no_positive"],
                ),
                flush=True,
            )

            for split_name, dataset in eval_datasets.items():
                metrics = evaluate_stage1_dataset(
                    model=model,
                    dataset=dataset,
                    config=config,
                    device=device,
                    citation_graph_dir=citation_graph_dir,
                )
                final_metrics[split_name] = metrics
                writer.writerow({"epoch": epoch, "split": split_name, **metrics})
                print(
                    "epoch={epoch} {split}_loss={loss:.6f} {split}_top1={top1:.4f} "
                    "nodes={nodes} skipped={skipped}".format(
                        epoch=epoch,
                        split=split_name,
                        loss=metrics["loss"],
                        top1=metrics["top1_gold_accuracy"],
                        nodes=metrics["nodes_used"],
                        skipped=metrics["nodes_skipped_no_positive"],
                    ),
                    flush=True,
                )
            f.flush()

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": asdict(config),
        "input_dim": train_dataset.input_dim,
        "projection_dim": config.projection_dim,
        "role_ids": train_dataset.role_ids,
    }
    torch.save(checkpoint, out_dir / "checkpoint_last.pt")
    summary = {
        "out_dir": str(out_dir),
        "config": asdict(config),
        "metrics": final_metrics,
        "checkpoint": str(out_dir / "checkpoint_last.pt"),
        "log": str(log_path),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def train_stage1_smoke(
    dataset: Stage1SmokeDataset,
    config: TrainConfig,
    out_dir: Path,
    citation_graph_dir: Path | None = None,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    device = choose_device(config.device)
    train_tasks, eval_tasks = split_tasks(dataset.tasks, config.eval_frac, config.seed)
    citation_weights = (
        CitationProximityWeights(citation_graph_dir, relevant_author_ids=dataset.author_ids)
        if citation_graph_dir
        else None
    )
    if config.negative_mode.lower() == "v3" and config.prox_threshold is None:
        if citation_weights is None:
            raise ValueError("negative_mode=v3 requires --citation-graph-dir")
        threshold = citation_weights.estimate_threshold(
            train_tasks,
            dataset.author_ids,
            config.prox_quantile,
        )
        config = TrainConfig(**{**asdict(config), "prox_threshold": threshold})

    model = Stage1ProjectionModel(dataset.input_dim, config.projection_dim).to(device)
    loss_computer = Stage1LossComputer(dataset, config, device, citation_weights)
    loss_computer.precompute_untrained_topm_negatives(train_tasks)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    config_payload = {
        "config": asdict(config),
        "paths": {key: str(value) for key, value in asdict(dataset.paths).items()},
        "num_tasks": len(dataset.tasks),
        "num_train_tasks": len(train_tasks),
        "num_eval_tasks": len(eval_tasks),
        "num_authors": len(dataset.author_ids),
        "input_dim": dataset.input_dim,
        "device": str(device),
        "citation_graph_dir": str(citation_graph_dir) if citation_graph_dir else "",
    }
    (out_dir / "config.json").write_text(
        json.dumps(config_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    log_path = out_dir / "train_log.tsv"
    with log_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "epoch",
            "split",
            "loss",
            "nodes_used",
            "nodes_skipped_no_positive",
            "top1_gold_accuracy",
            "top1_gold_hits",
            "top1_total",
            "mean_positive_score",
            "mean_negative_score",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()

        last_train_metrics: dict = {}
        last_eval_metrics: dict = {}
        for epoch in range(1, config.epochs + 1):
            random.Random(config.seed + epoch).shuffle(train_tasks)
            model.train()
            loss_computer.sample_round = epoch
            train_batch_metrics: List[BatchMetrics] = []
            for batch in batched(train_tasks, config.batch_size):
                optimizer.zero_grad(set_to_none=True)
                loss, metrics = loss_computer.compute_batch_loss(model, batch)
                loss.backward()
                optimizer.step()
                train_batch_metrics.append(metrics)
            last_train_metrics = aggregate_metrics(train_batch_metrics)
            writer.writerow({"epoch": epoch, "split": "train", **last_train_metrics})

            if eval_tasks:
                model.eval()
                loss_computer.sample_round = 0
                eval_batch_metrics: List[BatchMetrics] = []
                with torch.no_grad():
                    for batch in batched(eval_tasks, config.batch_size):
                        _, metrics = loss_computer.compute_batch_loss(model, batch)
                        eval_batch_metrics.append(metrics)
                last_eval_metrics = aggregate_metrics(eval_batch_metrics)
                writer.writerow({"epoch": epoch, "split": "eval", **last_eval_metrics})
            f.flush()

            print(
                "epoch={epoch} train_loss={loss:.6f} train_top1={top1:.4f} "
                "nodes={nodes} skipped={skipped}".format(
                    epoch=epoch,
                    loss=last_train_metrics["loss"],
                    top1=last_train_metrics["top1_gold_accuracy"],
                    nodes=last_train_metrics["nodes_used"],
                    skipped=last_train_metrics["nodes_skipped_no_positive"],
                ),
                flush=True,
            )
            if eval_tasks:
                print(
                    "epoch={epoch} eval_loss={loss:.6f} eval_top1={top1:.4f} "
                    "nodes={nodes} skipped={skipped}".format(
                        epoch=epoch,
                        loss=last_eval_metrics["loss"],
                        top1=last_eval_metrics["top1_gold_accuracy"],
                        nodes=last_eval_metrics["nodes_used"],
                        skipped=last_eval_metrics["nodes_skipped_no_positive"],
                    ),
                    flush=True,
                )

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": asdict(config),
        "input_dim": dataset.input_dim,
        "projection_dim": config.projection_dim,
        "role_ids": dataset.role_ids,
    }
    torch.save(checkpoint, out_dir / "checkpoint_last.pt")
    summary = {
        "out_dir": str(out_dir),
        "config": asdict(config),
        "train": last_train_metrics,
        "eval": last_eval_metrics,
        "checkpoint": str(out_dir / "checkpoint_last.pt"),
        "log": str(log_path),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary
