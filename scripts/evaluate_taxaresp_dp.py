#!/usr/bin/env python3
"""Evaluate the full TaxaResp-DP recurrence.

This implements the DP_u[x,k] method: the state keeps both the number of
responsibility regions and the owner of the open region containing the current
taxonomy node. Region owners are not constrained to be unique.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


VIRTUAL_ROOT = "__task_root__"
METHOD = "taxaresp_dp_full_with_virtual_root"
NEG_INF = -1.0e100


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--task-nodes-jsonl",
        default="output/hierec_embedding_server_inputs/task_nodes.jsonl",
    )
    p.add_argument(
        "--node-topm-candidates",
        default="output/embedding_taxonomy_region_cut_jsd_topm256_temp015_no_label/node_topm_candidates.tsv",
        help="Per-task-node Top-M same-node expert similarities.",
    )
    p.add_argument(
        "--virtual-root-matches",
        default="output/virtual_root_role_descriptions/virtual_root_expert_matches_llm_gptoss120b.tsv",
        help="Per-task virtual-root expert similarities.",
    )
    p.add_argument("--fos-children", default="data/dblp/13.FieldOfStudyChildren.nt")
    p.add_argument("--out-dir", default="output/taxaresp_dp_full_with_virtual_root_topm256_no_unique_owner")
    p.add_argument("--method-label", default=METHOD)
    p.add_argument(
        "--max-rank",
        type=int,
        default=256,
        help="Maximum rank to keep from the node Top-M candidate TSV.",
    )
    p.add_argument(
        "--missing-similarity",
        type=float,
        default=0.0,
        help=(
            "Similarity used for experts in X_q that were not retrieved for a node. "
            "This keeps X_q = union_v C_v usable with a truncated Top-M cache."
        ),
    )
    p.add_argument(
        "--max-root-rank",
        type=int,
        default=20,
        help="Maximum rank to keep from the virtual-root candidate TSV.",
    )
    p.add_argument(
        "--root-weight",
        choices=("all_skill_sum", "one", "none"),
        default="all_skill_sum",
        help="Scale virtual-root cosine similarity when the root is counted.",
    )
    p.add_argument(
        "--virtual-root-mode",
        choices=("counted", "connector"),
        default="counted",
        help="counted treats the virtual root as a responsibility node; connector preserves the previous zero-weight connector behavior.",
    )
    p.add_argument(
        "--limit-tasks",
        type=int,
        default=0,
        help="Optional smoke-test limit over task order; 0 means all tasks.",
    )
    p.add_argument(
        "--progress-every",
        type=int,
        default=25,
        help="Print progress every N tasks; 0 disables progress prints.",
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


def read_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def as_members(value) -> List[str]:
    if isinstance(value, list):
        return [str(x) for x in value]
    if isinstance(value, str):
        return [x for x in value.replace("|", ",").split(",") if x]
    return []


def dedupe(values: Iterable[str]) -> List[str]:
    seen = set()
    out = []
    for value in values:
        value = str(value)
        if value and value not in seen:
            seen.add(value)
            out.append(value)
    return out


def parse_subtree_skill_names(row: dict) -> set:
    names = set()
    for part in str(row.get("subtree_skills") or "").split(";"):
        name = part.rsplit(":", 1)[0].strip()
        if name:
            names.add(name)
    return names


def parse_weight_sum(text: str) -> float:
    total = 0.0
    for part in str(text or "").split(";"):
        part = part.strip()
        if not part or ":" not in part:
            continue
        _, raw = part.rsplit(":", 1)
        total += max(0.0, safe_float(raw, 0.0))
    return total


def root_weight_value(mode: str, all_skill_sum: float) -> float:
    if mode == "none":
        return 0.0
    if mode == "one":
        return 1.0
    return max(all_skill_sum, 1e-6)


def load_child_to_parents(path: Path) -> Dict[str, List[str]]:
    pat = re.compile(
        r"<https://makg.org/entity/(\d+)>\s+"
        r"<https://makg.org/property/hasParent>\s+"
        r"<https://makg.org/entity/(\d+)>\s+\."
    )
    out: Dict[str, List[str]] = defaultdict(list)
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            m = pat.match(line.strip())
            if m:
                out[m.group(1)].append(m.group(2))
    return dict(out)


def choose_task_parent(
    row: dict,
    node_ids: set,
    row_by_node: Dict[str, dict],
    child_to_parents: Dict[str, List[str]],
) -> str:
    node_id = str(row["node_id"])
    direct_parents = [p for p in child_to_parents.get(node_id, []) if p in node_ids]
    if not direct_parents:
        return VIRTUAL_ROOT
    child_skills = parse_subtree_skill_names(row)

    def key(parent_id: str) -> tuple:
        parent = row_by_node[parent_id]
        level_gap = safe_int(row.get("node_level"), 99) - safe_int(parent.get("node_level"), 99)
        parent_skills = parse_subtree_skill_names(parent)
        overlap = len(child_skills & parent_skills)
        return (level_gap, -overlap, str(parent_id))

    return sorted(direct_parents, key=key)[0]


def load_tasks(path: Path) -> Tuple[Dict[str, List[dict]], Dict[str, dict], List[str]]:
    rows_by_paper: Dict[str, List[dict]] = defaultdict(list)
    info = {}
    order = []
    for row in read_jsonl(path):
        paper_id = str(row["paper_id"])
        if paper_id not in info:
            order.append(paper_id)
        row = dict(row)
        row["paper_id"] = paper_id
        row["node_id"] = str(row["node_id"])
        row["task_node_id"] = f"{paper_id}::{row['node_id']}"
        row["node_importance"] = max(safe_float(row.get("node_importance"), 0.0), 0.0)
        rows_by_paper[paper_id].append(row)
        info[paper_id] = {
            "team_size": int(row.get("team_size") or len(as_members(row.get("members"))) or 1),
            "members": dedupe(as_members(row.get("members"))),
            "all_skill_sum": parse_weight_sum(row.get("all_task_skills")),
        }
    return rows_by_paper, info, order


def build_children(rows: List[dict], child_to_parents: Dict[str, List[str]]) -> Dict[str, List[str]]:
    row_by_node = {str(row["node_id"]): row for row in rows}
    node_ids = set(row_by_node)
    children: Dict[str, List[str]] = defaultdict(list)
    for row in rows:
        child_id = str(row["node_id"])
        parent_id = choose_task_parent(row, node_ids, row_by_node, child_to_parents)
        children[parent_id].append(child_id)
    for parent_id in list(children):
        children[parent_id].sort(
            key=lambda nid: (safe_int(row_by_node.get(nid, {}).get("node_level"), 99), str(nid))
        )
    return children


def load_node_rankings(
    path: Path,
    max_rank: int,
) -> Tuple[Dict[str, Dict[str, List[Tuple[str, float]]]], Dict[str, str]]:
    rankings: Dict[str, Dict[str, List[Tuple[str, float]]]] = defaultdict(lambda: defaultdict(list))
    expert_names: Dict[str, str] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            rank = safe_int(row.get("rank"), 0)
            if rank <= 0 or rank > max_rank:
                continue
            paper_id = str(row["paper_id"])
            node_id = str(row["node_id"])
            expert_id = str(row["expert_id"])
            expert_names.setdefault(expert_id, row.get("expert_name") or expert_id)
            rankings[paper_id][node_id].append((expert_id, safe_float(row.get("similarity"), 0.0)))
    for per_task in rankings.values():
        for node_id in list(per_task):
            per_task[node_id].sort(key=lambda item: (-item[1], item[0]))
    return {paper_id: dict(per_task) for paper_id, per_task in rankings.items()}, expert_names


def load_virtual_root_rankings(
    path: Path,
    max_rank: int,
) -> Tuple[Dict[str, List[Tuple[str, float]]], Dict[str, str]]:
    rankings: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
    expert_names: Dict[str, str] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            rank = safe_int(row.get("rank"), 0)
            if rank <= 0 or rank > max_rank:
                continue
            paper_id = str(row["paper_id"])
            expert_id = str(row["expert_id"])
            expert_names.setdefault(expert_id, expert_id)
            rankings[paper_id].append((expert_id, safe_float(row.get("cosine_similarity"), 0.0)))
    for paper_id in list(rankings):
        rankings[paper_id].sort(key=lambda item: (-item[1], item[0]))
    return dict(rankings), expert_names


def evaluate_prediction(preds: Sequence[str], golds: Sequence[str]) -> Tuple[int, float, float]:
    pred_unique = dedupe(preds)
    gold_unique = dedupe(golds)
    hits = len(set(pred_unique) & set(gold_unique))
    precision = hits / len(pred_unique) if pred_unique else 0.0
    recall = hits / len(gold_unique) if gold_unique else 0.0
    return hits, precision, recall


def raw_precision(preds: Sequence[str], golds: Sequence[str]) -> Tuple[int, float, float]:
    gold_unique = set(dedupe(golds))
    hits = len(set(dedupe(preds)) & gold_unique)
    precision = hits / len(preds) if preds else 0.0
    recall = hits / len(gold_unique) if gold_unique else 0.0
    return hits, precision, recall


class TaxaRespSolver:
    def __init__(
        self,
        paper_id: str,
        rows: List[dict],
        children: Dict[str, List[str]],
        rankings: Dict[str, List[Tuple[str, float]]],
        virtual_root_ranking: Sequence[Tuple[str, float]],
        expert_names: Dict[str, str],
        k: int,
        missing_similarity: float,
        root_weight: float,
        virtual_root_counted: bool,
    ) -> None:
        self.paper_id = paper_id
        self.rows = rows
        self.children = children
        self.row_by_node = {str(row["node_id"]): row for row in rows}
        self.virtual_root_counted = virtual_root_counted
        node_count = len(rows) + (1 if virtual_root_counted else 0)
        self.k = min(k, max(1, node_count))
        self.missing_similarity = missing_similarity

        expert_ids = sorted(
            {expert_id for pairs in rankings.values() for expert_id, _ in pairs}
            | {expert_id for expert_id, _ in virtual_root_ranking}
        )
        if not expert_ids:
            raise ValueError(f"paper_id={paper_id} has no node candidates")
        self.expert_ids = expert_ids
        self.expert_names = expert_names
        self.expert_to_idx = {expert_id: idx for idx, expert_id in enumerate(expert_ids)}
        self.l_experts = len(expert_ids)

        self.utility: Dict[str, List[float]] = {}
        for node_id, row in self.row_by_node.items():
            base = row["node_importance"] * self.missing_similarity
            values = [base] * self.l_experts
            for expert_id, sim in rankings.get(node_id, []):
                idx = self.expert_to_idx.get(expert_id)
                if idx is not None:
                    values[idx] = row["node_importance"] * sim
            self.utility[node_id] = values
        root_values = [0.0] * self.l_experts
        if virtual_root_counted:
            for expert_id, sim in virtual_root_ranking:
                idx = self.expert_to_idx.get(expert_id)
                if idx is not None:
                    root_values[idx] = root_weight * sim
        self.utility[VIRTUAL_ROOT] = root_values

        self.tables: Dict[str, List[List[float]]] = {}
        self.best_owner: Dict[str, List[Optional[int]]] = {}
        self.best_value: Dict[str, List[float]] = {}
        self.stage_backs: Dict[str, List[List[List[Optional[tuple]]]]] = {}
        self.virtual_stage_backs: List[List[List[List[Optional[tuple]]]]] = []

    def solve_node(self, node_id: str) -> List[List[float]]:
        if node_id in self.tables:
            return self.tables[node_id]

        table = [[NEG_INF] * (self.k + 1) for _ in range(self.l_experts)]
        util = self.utility[node_id]
        for x_idx in range(self.l_experts):
            table[x_idx][1] = util[x_idx]

        node_stage_backs: List[List[List[Optional[tuple]]]] = []
        for child_id in self.children.get(node_id, []):
            child_table = self.solve_node(child_id)
            child_best = self.best_value[child_id]
            child_best_owner = self.best_owner[child_id]
            next_table = [[NEG_INF] * (self.k + 1) for _ in range(self.l_experts)]
            stage_back = [[None] * (self.k + 1) for _ in range(self.l_experts)]

            for x_idx in range(self.l_experts):
                parent_vals = table[x_idx]
                child_vals_same_owner = child_table[x_idx]
                for p_regions in range(1, self.k + 1):
                    parent_score = parent_vals[p_regions]
                    if parent_score <= NEG_INF / 2:
                        continue

                    for child_regions in range(1, self.k + 1):
                        child_score = child_vals_same_owner[child_regions]
                        if child_score > NEG_INF / 2:
                            new_regions = p_regions + child_regions - 1
                            if new_regions <= self.k:
                                score = parent_score + child_score
                                if score > next_table[x_idx][new_regions]:
                                    next_table[x_idx][new_regions] = score
                                    stage_back[x_idx][new_regions] = (
                                        "keep",
                                        p_regions,
                                        child_regions,
                                        x_idx,
                                    )

                        owner_free_score = child_best[child_regions]
                        child_owner = child_best_owner[child_regions]
                        if child_owner is None or owner_free_score <= NEG_INF / 2:
                            continue
                        new_regions = p_regions + child_regions
                        if new_regions <= self.k:
                            score = parent_score + owner_free_score
                            if score > next_table[x_idx][new_regions]:
                                next_table[x_idx][new_regions] = score
                                stage_back[x_idx][new_regions] = (
                                    "cut",
                                    p_regions,
                                    child_regions,
                                    child_owner,
                                )

            table = next_table
            node_stage_backs.append(stage_back)

        best_vals = [NEG_INF] * (self.k + 1)
        best_owner = [None] * (self.k + 1)
        for regions in range(1, self.k + 1):
            best_idx = None
            best_score = NEG_INF
            for x_idx in range(self.l_experts):
                score = table[x_idx][regions]
                if score > best_score:
                    best_score = score
                    best_idx = x_idx
            if best_idx is not None and best_score > NEG_INF / 2:
                best_vals[regions] = best_score
                best_owner[regions] = best_idx

        self.tables[node_id] = table
        self.best_value[node_id] = best_vals
        self.best_owner[node_id] = best_owner
        self.stage_backs[node_id] = node_stage_backs
        return table

    def solve_virtual_root(self) -> List[List[List[float]]]:
        root_children = self.children.get(VIRTUAL_ROOT, [])
        table = [[[NEG_INF, NEG_INF] for _ in range(self.k + 1)] for _ in range(self.l_experts)]
        for x_idx in range(self.l_experts):
            table[x_idx][0][0] = 0.0

        backs: List[List[List[List[Optional[tuple]]]]] = []
        for child_id in root_children:
            child_table = self.solve_node(child_id)
            child_best = self.best_value[child_id]
            child_best_owner = self.best_owner[child_id]
            next_table = [[[NEG_INF, NEG_INF] for _ in range(self.k + 1)] for _ in range(self.l_experts)]
            stage_back = [
                [[None, None] for _ in range(self.k + 1)] for _ in range(self.l_experts)
            ]

            for x_idx in range(self.l_experts):
                for p_regions in range(0, self.k + 1):
                    for active in (0, 1):
                        parent_score = table[x_idx][p_regions][active]
                        if parent_score <= NEG_INF / 2:
                            continue

                        for child_regions in range(1, self.k + 1):
                            owner_free_score = child_best[child_regions]
                            child_owner = child_best_owner[child_regions]
                            if child_owner is not None and owner_free_score > NEG_INF / 2:
                                new_regions = p_regions + child_regions
                                if new_regions <= self.k:
                                    score = parent_score + owner_free_score
                                    if score > next_table[x_idx][new_regions][active]:
                                        next_table[x_idx][new_regions][active] = score
                                        stage_back[x_idx][new_regions][active] = (
                                            "cut",
                                            p_regions,
                                            child_regions,
                                            child_owner,
                                            active,
                                        )

                            child_score = child_table[x_idx][child_regions]
                            if child_score <= NEG_INF / 2:
                                continue
                            new_regions = p_regions + child_regions if active == 0 else p_regions + child_regions - 1
                            if new_regions <= self.k:
                                score = parent_score + child_score
                                if score > next_table[x_idx][new_regions][1]:
                                    next_table[x_idx][new_regions][1] = score
                                    stage_back[x_idx][new_regions][1] = (
                                        "keep",
                                        p_regions,
                                        child_regions,
                                        x_idx,
                                        active,
                                    )

            table = next_table
            backs.append(stage_back)

        self.virtual_stage_backs = backs
        return table

    def solve(self) -> Tuple[float, int, List[Tuple[str, str]], Dict[str, int]]:
        if self.virtual_root_counted:
            self.solve_node(VIRTUAL_ROOT)
            root_best = self.best_value[VIRTUAL_ROOT]
            root_owner = self.best_owner[VIRTUAL_ROOT]
            target_k = self.k
            if root_owner[target_k] is None:
                feasible = [regions for regions in range(1, self.k + 1) if root_owner[regions] is not None]
                if not feasible:
                    return 0.0, 0, [], {}
                target_k = max(feasible, key=lambda regions: (regions, root_best[regions]))
            cuts: List[Tuple[str, str]] = []
            owner_by_node: Dict[str, int] = {}
            self._trace_node(VIRTUAL_ROOT, root_owner[target_k], target_k, cuts, owner_by_node)
            return root_best[target_k], target_k, cuts, owner_by_node

        root_table = self.solve_virtual_root()
        target_k = self.k
        best_choice = self._best_virtual_choice(root_table, target_k)
        if best_choice is None:
            feasible = [
                regions
                for regions in range(1, self.k + 1)
                if self._best_virtual_choice(root_table, regions) is not None
            ]
            if not feasible:
                return 0.0, 0, [], {}
            target_k = max(
                feasible,
                key=lambda regions: (
                    regions,
                    self._best_virtual_choice(root_table, regions)[2],
                ),
            )
            best_choice = self._best_virtual_choice(root_table, target_k)

        cuts: List[Tuple[str, str]] = []
        owner_by_node: Dict[str, int] = {}
        root_owner_idx, root_active, root_score = best_choice
        self._trace_virtual_root(
            len(self.children.get(VIRTUAL_ROOT, [])),
            root_owner_idx,
            target_k,
            root_active,
            cuts,
            owner_by_node,
        )
        return root_score, target_k, cuts, owner_by_node

    def _best_virtual_choice(
        self,
        root_table: List[List[List[float]]],
        regions: int,
    ) -> Optional[Tuple[int, int, float]]:
        best = None
        for x_idx in range(self.l_experts):
            for active in (0, 1):
                score = root_table[x_idx][regions][active]
                if score <= NEG_INF / 2:
                    continue
                if best is None or score > best[2]:
                    best = (x_idx, active, score)
        return best

    def _trace_virtual_root(
        self,
        stage_idx: int,
        owner_idx: int,
        regions: int,
        active: int,
        cuts: List[Tuple[str, str]],
        owner_by_node: Dict[str, int],
    ) -> None:
        if stage_idx == 0:
            return
        children = self.children.get(VIRTUAL_ROOT, [])
        child_id = children[stage_idx - 1]
        choice = self.virtual_stage_backs[stage_idx - 1][owner_idx][regions][active]
        if choice is None:
            raise RuntimeError(
                f"missing virtual backpointer paper_id={self.paper_id} "
                f"stage={stage_idx} owner={owner_idx} regions={regions} active={active}"
            )
        decision, prev_regions, child_regions, child_owner, prev_active = choice
        self._trace_virtual_root(
            stage_idx - 1,
            owner_idx,
            prev_regions,
            prev_active,
            cuts,
            owner_by_node,
        )
        if decision == "keep":
            owner_by_node[VIRTUAL_ROOT] = owner_idx
            self._trace_node(child_id, owner_idx, child_regions, cuts, owner_by_node)
        elif decision == "cut":
            cuts.append((VIRTUAL_ROOT, child_id))
            self._trace_node(child_id, child_owner, child_regions, cuts, owner_by_node)
        else:
            raise RuntimeError(f"unknown virtual decision: {decision}")

    def _trace_node(
        self,
        node_id: str,
        owner_idx: int,
        regions: int,
        cuts: List[Tuple[str, str]],
        owner_by_node: Dict[str, int],
    ) -> None:
        owner_by_node[node_id] = owner_idx
        self._trace_stage(
            node_id,
            len(self.children.get(node_id, [])),
            owner_idx,
            regions,
            cuts,
            owner_by_node,
        )

    def _trace_stage(
        self,
        node_id: str,
        stage_idx: int,
        owner_idx: int,
        regions: int,
        cuts: List[Tuple[str, str]],
        owner_by_node: Dict[str, int],
    ) -> None:
        if stage_idx == 0:
            return
        children = self.children.get(node_id, [])
        child_id = children[stage_idx - 1]
        choice = self.stage_backs[node_id][stage_idx - 1][owner_idx][regions]
        if choice is None:
            raise RuntimeError(
                f"missing backpointer paper_id={self.paper_id} node={node_id} "
                f"stage={stage_idx} owner={owner_idx} regions={regions}"
            )
        decision, prev_regions, child_regions, child_owner = choice
        self._trace_stage(node_id, stage_idx - 1, owner_idx, prev_regions, cuts, owner_by_node)
        if decision == "keep":
            self._trace_node(child_id, owner_idx, child_regions, cuts, owner_by_node)
        elif decision == "cut":
            cuts.append((node_id, child_id))
            self._trace_node(child_id, child_owner, child_regions, cuts, owner_by_node)
        else:
            raise RuntimeError(f"unknown decision: {decision}")

    def connected_regions(
        self,
        cuts: Sequence[Tuple[str, str]],
        owner_by_node: Dict[str, int],
    ) -> List[dict]:
        cut_pairs = set(cuts)
        graph: Dict[str, List[str]] = defaultdict(list)
        all_nodes = {VIRTUAL_ROOT}
        all_nodes.update(self.row_by_node)
        for parent_id, child_ids in self.children.items():
            for child_id in child_ids:
                if (parent_id, child_id) in cut_pairs:
                    continue
                graph[parent_id].append(child_id)
                graph[child_id].append(parent_id)

        regions = []
        seen = set()
        for start in sorted(all_nodes):
            if start in seen:
                continue
            q = deque([start])
            seen.add(start)
            augmented_nodes = []
            real_nodes = []
            while q:
                node_id = q.popleft()
                augmented_nodes.append(node_id)
                if node_id != VIRTUAL_ROOT:
                    real_nodes.append(node_id)
                for nxt in graph.get(node_id, []):
                    if nxt not in seen:
                        seen.add(nxt)
                        q.append(nxt)
            if not real_nodes and not self.virtual_root_counted:
                continue
            owner_idx = owner_by_node[augmented_nodes[0]]
            score_node_ids = list(real_nodes)
            if self.virtual_root_counted and VIRTUAL_ROOT in augmented_nodes:
                score_node_ids.append(VIRTUAL_ROOT)
            score = sum(self.utility[node_id][owner_idx] for node_id in score_node_ids)
            output_node_ids = sorted(real_nodes)
            if self.virtual_root_counted and VIRTUAL_ROOT in augmented_nodes:
                output_node_ids = [VIRTUAL_ROOT] + output_node_ids
            regions.append(
                {
                    "augmented_nodes": sorted(augmented_nodes),
                    "node_ids": output_node_ids,
                    "owner_idx": owner_idx,
                    "score": score,
                    "includes_virtual_root": VIRTUAL_ROOT in augmented_nodes,
                }
            )

        regions.sort(
            key=lambda region: (
                -region["score"],
                region["node_ids"][0] if region["node_ids"] else "",
                self.expert_ids[region["owner_idx"]],
            )
        )
        return regions

    def region_label(self, node_ids: Sequence[str], max_names: int = 4) -> str:
        if list(node_ids) == [VIRTUAL_ROOT]:
            return "Task"
        rows = [self.row_by_node[node_id] for node_id in node_ids if node_id != VIRTUAL_ROOT]
        rows.sort(
            key=lambda row: (
                -row["node_importance"],
                safe_int(row.get("node_level"), 99),
                str(row.get("node_name") or row.get("node_id")),
            )
        )
        names = []
        if VIRTUAL_ROOT in node_ids:
            names.append("Task")
        names.extend(str(row.get("node_name") or row["node_id"]) for row in rows[:max_names])
        return " / ".join(names[:max_names])


def summarize(task_rows: List[dict], method_label: str) -> dict:
    micro_hits = sum(int(row["hits"]) for row in task_rows)
    micro_selected = sum(int(row["selected_regions"]) for row in task_rows)
    micro_unique_selected = sum(int(row["selected_unique_experts"]) for row in task_rows)
    micro_gold = sum(int(row["gold_count"]) for row in task_rows)
    return {
        "method": method_label,
        "tasks": len(task_rows),
        "macro_precision_percent": f"{100 * mean([float(row['precision']) for row in task_rows]):.6f}",
        "macro_recall_percent": f"{100 * mean([float(row['recall']) for row in task_rows]):.6f}",
        "dedup_macro_precision_percent": f"{100 * mean([float(row['dedup_precision']) for row in task_rows]):.6f}",
        "dedup_macro_recall_percent": f"{100 * mean([float(row['dedup_recall']) for row in task_rows]):.6f}",
        "micro_precision_percent": f"{100 * micro_hits / micro_selected if micro_selected else 0.0:.6f}",
        "micro_recall_percent": f"{100 * micro_hits / micro_gold if micro_gold else 0.0:.6f}",
        "dedup_micro_precision_percent": (
            f"{100 * micro_hits / micro_unique_selected if micro_unique_selected else 0.0:.6f}"
        ),
        "dedup_micro_recall_percent": f"{100 * micro_hits / micro_gold if micro_gold else 0.0:.6f}",
        "micro_hits": micro_hits,
        "micro_predicted": micro_selected,
        "dedup_micro_predicted": micro_unique_selected,
        "micro_gold": micro_gold,
        "avg_regions": f"{mean([int(row['selected_regions']) for row in task_rows]):.6f}",
        "avg_unique_experts": f"{mean([int(row['selected_unique_experts']) for row in task_rows]):.6f}",
        "duplicate_expert_assignments": sum(int(row["duplicates"]) for row in task_rows),
    }


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows_by_paper, task_info, paper_order = load_tasks(Path(args.task_nodes_jsonl))
    if args.limit_tasks > 0:
        paper_order = paper_order[: args.limit_tasks]
    child_to_parents = load_child_to_parents(Path(args.fos_children))
    rankings_by_paper, expert_names = load_node_rankings(Path(args.node_topm_candidates), args.max_rank)
    root_rankings_by_paper, root_expert_names = load_virtual_root_rankings(
        Path(args.virtual_root_matches), args.max_root_rank
    )
    expert_names.update(root_expert_names)

    task_rows = []
    region_rows = []
    prediction_rows = []
    cut_rows = []

    for idx, paper_id in enumerate(paper_order, start=1):
        if args.progress_every and (idx == 1 or idx % args.progress_every == 0 or idx == len(paper_order)):
            print(f"taxaresp_dp_progress task={idx}/{len(paper_order)} paper_id={paper_id}", flush=True)

        rows = rows_by_paper[paper_id]
        info = task_info[paper_id]
        k = max(1, int(info["team_size"]))
        children = build_children(rows, child_to_parents)
        root_weight = root_weight_value(args.root_weight, safe_float(info.get("all_skill_sum"), 0.0))
        solver = TaxaRespSolver(
            paper_id=paper_id,
            rows=rows,
            children=children,
            rankings=rankings_by_paper.get(paper_id, {}),
            virtual_root_ranking=root_rankings_by_paper.get(paper_id, []),
            expert_names=expert_names,
            k=k,
            missing_similarity=args.missing_similarity,
            root_weight=root_weight,
            virtual_root_counted=args.virtual_root_mode == "counted",
        )
        score, used_k, cuts, owner_by_node = solver.solve()
        regions = solver.connected_regions(cuts, owner_by_node)
        raw_experts = [solver.expert_ids[region["owner_idx"]] for region in regions]
        dedup_experts = dedupe(raw_experts)
        gold = info["members"]
        hits, precision, recall = raw_precision(raw_experts, gold)
        dedup_hits, dedup_precision, dedup_recall = evaluate_prediction(dedup_experts, gold)
        duplicates = len(raw_experts) - len(dedup_experts)

        task_rows.append(
            {
                "paper_id": paper_id,
                "team_size": k,
                "used_regions": used_k,
                "selected_regions": len(regions),
                "selected_unique_experts": len(dedup_experts),
                "candidate_experts": solver.l_experts,
                "task_nodes": len(rows) + (1 if args.virtual_root_mode == "counted" else 0),
                "real_task_nodes": len(rows),
                "virtual_root_mode": args.virtual_root_mode,
                "root_weight": f"{root_weight:.9f}",
                "cut_edges": len(cuts),
                "score": f"{score:.9f}",
                "hits": hits,
                "precision": f"{precision:.12f}",
                "recall": f"{recall:.12f}",
                "dedup_hits": dedup_hits,
                "dedup_precision": f"{dedup_precision:.12f}",
                "dedup_recall": f"{dedup_recall:.12f}",
                "duplicates": duplicates,
                "gold_count": len(gold),
                "gold": "|".join(gold),
            }
        )

        for parent_id, child_id in cuts:
            cut_rows.append(
                {
                    "method": args.method_label,
                    "paper_id": paper_id,
                    "parent_node_id": parent_id,
                    "child_node_id": child_id,
                    "parent_node_name": "Task" if parent_id == VIRTUAL_ROOT else solver.row_by_node[parent_id].get("node_name", parent_id),
                    "child_node_name": "Task" if child_id == VIRTUAL_ROOT else solver.row_by_node[child_id].get("node_name", child_id),
                }
            )

        for rank, region in enumerate(regions, start=1):
            expert_id = solver.expert_ids[region["owner_idx"]]
            expert_name = expert_names.get(expert_id, expert_id)
            is_gold = "1" if expert_id in set(gold) else "0"
            node_ids = region["node_ids"]
            node_names = [
                "Task" if node_id == VIRTUAL_ROOT else solver.row_by_node[node_id].get("node_name", node_id)
                for node_id in node_ids
            ]
            region_id = rank
            region_rows.append(
                {
                    "method": args.method_label,
                    "paper_id": paper_id,
                    "region_id": region_id,
                    "region_label": solver.region_label(node_ids),
                    "expert_id": expert_id,
                    "expert_name": expert_name,
                    "score": f"{region['score']:.9f}",
                    "node_count": len(node_ids),
                    "node_ids": "|".join(node_ids),
                    "node_names": "|".join(node_names),
                    "augmented_node_ids": "|".join(region["augmented_nodes"]),
                    "includes_virtual_root": "1" if region["includes_virtual_root"] else "0",
                    "is_actual_member": is_gold,
                }
            )
            prediction_rows.append(
                {
                    "method": args.method_label,
                    "paper_id": paper_id,
                    "rank": rank,
                    "expert_id": expert_id,
                    "expert_name": expert_name,
                    "score": f"{region['score']:.9f}",
                    "region_id": region_id,
                    "is_actual_member": is_gold,
                }
            )

    if not task_rows:
        raise ValueError("no tasks were evaluated")

    with (out_dir / "task_metrics.tsv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(task_rows[0].keys()), delimiter="\t")
        writer.writeheader()
        writer.writerows(task_rows)
    with (out_dir / "regions.tsv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(region_rows[0].keys()), delimiter="\t")
        writer.writeheader()
        writer.writerows(region_rows)
    with (out_dir / "predictions_team_size.tsv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(prediction_rows[0].keys()), delimiter="\t")
        writer.writeheader()
        writer.writerows(prediction_rows)
    with (out_dir / "cut_edges.tsv").open("w", encoding="utf-8", newline="") as f:
        cut_fieldnames = [
            "method",
            "paper_id",
            "parent_node_id",
            "child_node_id",
            "parent_node_name",
            "child_node_name",
        ]
        writer = csv.DictWriter(f, fieldnames=cut_fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(cut_rows)

    summary = summarize(task_rows, args.method_label)
    with (out_dir / "metrics_summary.tsv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()), delimiter="\t")
        writer.writeheader()
        writer.writerow(summary)

    print(
        f"{args.method_label} tasks={summary['tasks']} macro_p={summary['macro_precision_percent']} "
        f"macro_r={summary['macro_recall_percent']} micro_p={summary['micro_precision_percent']} "
        f"micro_r={summary['micro_recall_percent']} duplicates={summary['duplicate_expert_assignments']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
