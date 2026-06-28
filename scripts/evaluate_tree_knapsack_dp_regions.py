#!/usr/bin/env python3
"""Evaluate simplified DP_u[m] taxonomy pruning.

This is the tree-knapsack version of RR-DP: the DP state tracks only the number
of selected responsibility regions, not the selected expert set.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


VIRTUAL_ROOT = "__task_root__"
METHOD = "tree_knapsack_dp_regions_fixed_k"
ROOT_POLICIES = ("optional", "forced", "none")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task-nodes-jsonl", default="output/hierec_embedding_server_inputs/task_nodes.jsonl")
    p.add_argument(
        "--node-topm-candidates",
        default="output/embedding_taxonomy_region_cut_jsd_topm256_temp015_no_label/node_topm_candidates.tsv",
        help="Use rank=1 same-node expert as owner(node).",
    )
    p.add_argument(
        "--virtual-root-matches",
        default="output/virtual_root_role_descriptions/virtual_root_expert_matches_llm_gptoss120b.tsv",
        help="Use rank=1 as owner(virtual_root).",
    )
    p.add_argument("--fos-children", default="data/dblp/13.FieldOfStudyChildren.nt")
    p.add_argument("--out-dir", default="output/tree_knapsack_dp_regions_llm_root")
    p.add_argument(
        "--root-policy",
        choices=ROOT_POLICIES,
        default="optional",
        help=(
            "How to treat the virtual root: optional can select it if useful, "
            "forced matches the original behavior, none uses it only as a connector."
        ),
    )
    p.add_argument(
        "--root-weight",
        choices=("all_skill_sum", "one", "none"),
        default="all_skill_sum",
        help="Scale virtual-root cosine score.",
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


def parse_weight_sum(text: str) -> float:
    total = 0.0
    for part in str(text or "").split(";"):
        part = part.strip()
        if not part or ":" not in part:
            continue
        _, raw = part.rsplit(":", 1)
        total += max(0.0, safe_float(raw, 0.0))
    return total


def parse_subtree_skill_names(row: dict) -> set:
    names = set()
    for part in str(row.get("subtree_skills") or "").split(";"):
        name = part.rsplit(":", 1)[0].strip()
        if name:
            names.add(name)
    return names


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
        row["node_importance"] = safe_float(row.get("node_importance"), 0.0)
        rows_by_paper[paper_id].append(row)
        info[paper_id] = {
            "team_size": int(row.get("team_size") or len(as_members(row.get("members"))) or 1),
            "members": dedupe(as_members(row.get("members"))),
            "all_skill_sum": parse_weight_sum(row.get("all_task_skills")),
        }
    return rows_by_paper, info, order


def load_node_rank1(path: Path) -> Dict[Tuple[str, str], dict]:
    out = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if int(float(row.get("rank") or 0)) != 1:
                continue
            out[(str(row["paper_id"]), str(row["node_id"]))] = {
                "expert_id": str(row["expert_id"]),
                "expert_name": row.get("expert_name") or str(row["expert_id"]),
                "similarity": safe_float(row.get("similarity"), 0.0),
            }
    return out


def load_virtual_root_rank1(path: Path) -> Dict[str, dict]:
    out = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if int(float(row.get("rank") or 0)) != 1:
                continue
            out[str(row["paper_id"])] = {
                "expert_id": str(row["expert_id"]),
                "expert_name": str(row["expert_id"]),
                "similarity": safe_float(row.get("cosine_similarity"), 0.0),
            }
    return out


def root_weight_value(mode: str, all_skill_sum: float) -> float:
    if mode == "none":
        return 0.0
    if mode == "one":
        return 1.0
    return max(all_skill_sum, 1e-6)


def build_children(rows: List[dict], child_to_parents: Dict[str, List[str]]) -> Dict[str, List[str]]:
    row_by_node = {str(row["node_id"]): row for row in rows}
    node_ids = set(row_by_node)
    children: Dict[str, List[str]] = defaultdict(list)
    for row in rows:
        child_id = str(row["node_id"])
        parent_id = choose_task_parent(row, node_ids, row_by_node, child_to_parents)
        children[parent_id].append(child_id)
    for parent in list(children):
        children[parent].sort(key=lambda nid: (safe_int(row_by_node.get(nid, {}).get("node_level"), 99), str(nid)))
    return children


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


def solve_task_dp(
    paper_id: str,
    rows: List[dict],
    info: dict,
    children: Dict[str, List[str]],
    node_rank1: Dict[Tuple[str, str], dict],
    root_rank1: Dict[str, dict],
    root_policy: str,
    root_weight_mode: str,
) -> Dict[int, Tuple[float, List[dict]]]:
    row_by_node = {str(row["node_id"]): row for row in rows}
    k = max(1, int(info["team_size"]))

    def node_owner_score(node_id: str) -> Tuple[str, str, float, float, float]:
        if node_id == VIRTUAL_ROOT:
            rec = root_rank1.get(paper_id)
            if not rec:
                return "", "", -math.inf, 0.0, 0.0
            weight = root_weight_value(root_weight_mode, safe_float(info.get("all_skill_sum"), 0.0))
            score = weight * rec["similarity"]
            return rec["expert_id"], rec["expert_name"], score, rec["similarity"], weight
        row = row_by_node[node_id]
        rec = node_rank1.get((paper_id, node_id))
        if not rec:
            return "", "", -math.inf, 0.0, row["node_importance"]
        score = row["node_importance"] * rec["similarity"]
        return rec["expert_id"], rec["expert_name"], score, rec["similarity"], row["node_importance"]

    memo: Dict[str, Dict[int, Tuple[float, List[dict]]]] = {}

    def dp(node_id: str) -> Dict[int, Tuple[float, List[dict]]]:
        if node_id in memo:
            return memo[node_id]
        states: Dict[int, Tuple[float, List[dict]]] = {}
        if node_id == VIRTUAL_ROOT and root_policy in ("optional", "none"):
            states[0] = (0.0, [])

        expert_id, expert_name, score, similarity, weight = node_owner_score(node_id)
        if root_policy != "none" or node_id != VIRTUAL_ROOT:
            if math.isfinite(score) and expert_id:
                label = "Task" if node_id == VIRTUAL_ROOT else row_by_node[node_id].get("node_name", node_id)
                states[1] = (
                    score,
                    [
                        {
                            "region_root_node_id": node_id,
                            "region_root_node_name": label,
                            "expert_id": expert_id,
                            "expert_name": expert_name,
                            "score": score,
                            "similarity": similarity,
                            "node_weight": weight,
                        }
                    ],
                )

        for child in children.get(node_id, []):
            child_states = dp(child)
            next_states = {}
            for m, (base_score, base_roots) in states.items():
                current_root_selected = node_id != VIRTUAL_ROOT or any(
                    root["region_root_node_id"] == VIRTUAL_ROOT for root in base_roots
                )
                # keep child: unchanged, child subtree is covered by node_id.
                if current_root_selected:
                    kept = (base_score, base_roots)
                    if m not in next_states or kept[0] > next_states[m][0]:
                        next_states[m] = kept
                for mv, (child_score, child_roots) in child_states.items():
                    new_m = m + mv
                    if new_m > k:
                        continue
                    cand = (base_score + child_score, base_roots + child_roots)
                    if new_m not in next_states or cand[0] > next_states[new_m][0]:
                        next_states[new_m] = cand
            states = next_states
        memo[node_id] = states
        return states

    return dp(VIRTUAL_ROOT)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows_by_paper, task_info, paper_order = load_tasks(Path(args.task_nodes_jsonl))
    child_to_parents = load_child_to_parents(Path(args.fos_children))
    node_rank1 = load_node_rank1(Path(args.node_topm_candidates))
    root_rank1 = load_virtual_root_rank1(Path(args.virtual_root_matches))

    prediction_rows = []
    region_rows = []
    task_rows = []

    for paper_id in paper_order:
        rows = rows_by_paper[paper_id]
        info = task_info[paper_id]
        k = max(1, int(info["team_size"]))
        children = build_children(rows, child_to_parents)
        states = solve_task_dp(
            paper_id, rows, info, children, node_rank1, root_rank1, args.root_policy, args.root_weight
        )
        score, roots = states.get(k, max(states.values(), key=lambda x: (len(x[1]), x[0])) if states else (0.0, []))
        raw_experts = [root["expert_id"] for root in roots]
        dedup_experts = dedupe(raw_experts)
        raw_hits, raw_p, raw_r = raw_precision(raw_experts, info["members"])
        dedup_hits, dedup_p, dedup_r = evaluate_prediction(dedup_experts, info["members"])
        duplicates = len(raw_experts) - len(dedup_experts)
        task_rows.append(
            {
                "paper_id": paper_id,
                "team_size": k,
                "selected_regions": len(roots),
                "selected_unique_experts": len(dedup_experts),
                "duplicates": duplicates,
                "score": f"{score:.9f}",
                "raw_hits": raw_hits,
                "raw_precision": raw_p,
                "raw_recall": raw_r,
                "dedup_hits": dedup_hits,
                "dedup_precision": dedup_p,
                "dedup_recall": dedup_r,
                "gold": "|".join(info["members"]),
            }
        )
        for rank, root in enumerate(roots, start=1):
            is_gold = "1" if root["expert_id"] in set(info["members"]) else "0"
            region_rows.append(
                {
                    "method": METHOD,
                    "paper_id": paper_id,
                    "rank": rank,
                    **root,
                    "is_actual_member": is_gold,
                }
            )
            prediction_rows.append(
                {
                    "method": METHOD,
                    "paper_id": paper_id,
                    "rank": rank,
                    "expert_id": root["expert_id"],
                    "expert_name": root["expert_name"],
                    "score": f"{root['score']:.9f}",
                    "is_actual_member": is_gold,
                }
            )

    def summarize(kind: str) -> dict:
        if kind.startswith("raw"):
            hits = sum(int(row["raw_hits"]) for row in task_rows)
            predicted = sum(int(row["selected_regions"]) for row in task_rows)
            precision_values = [float(row["raw_precision"]) for row in task_rows]
            recall_values = [float(row["raw_recall"]) for row in task_rows]
        else:
            hits = sum(int(row["dedup_hits"]) for row in task_rows)
            predicted = sum(int(row["selected_unique_experts"]) for row in task_rows)
            precision_values = [float(row["dedup_precision"]) for row in task_rows]
            recall_values = [float(row["dedup_recall"]) for row in task_rows]
        gold = sum(len(task_info[paper_id]["members"]) for paper_id in paper_order)
        return {
            "method": METHOD,
            "prediction_kind": kind,
            "tasks": len(task_rows),
            "macro_precision_percent": f"{100 * mean(precision_values):.6f}",
            "macro_recall_percent": f"{100 * mean(recall_values):.6f}",
            "micro_precision_percent": f"{100 * hits / predicted if predicted else 0.0:.6f}",
            "micro_recall_percent": f"{100 * hits / gold if gold else 0.0:.6f}",
            "micro_hits": hits,
            "micro_predicted": predicted,
            "micro_gold": gold,
            "avg_predicted": f"{predicted / len(task_rows):.6f}",
            "avg_gold": f"{gold / len(task_rows):.6f}",
                "duplicates": sum(int(row["duplicates"]) for row in task_rows),
                "root_policy": args.root_policy,
                "root_weight": args.root_weight,
            }

    with (out_dir / "predictions_team_size.tsv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(prediction_rows[0].keys()), delimiter="\t")
        writer.writeheader()
        writer.writerows(prediction_rows)
    with (out_dir / "selected_region_roots.tsv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(region_rows[0].keys()), delimiter="\t")
        writer.writeheader()
        writer.writerows(region_rows)
    with (out_dir / "task_metrics.tsv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(task_rows[0].keys()), delimiter="\t")
        writer.writeheader()
        writer.writerows(task_rows)
    summaries = [summarize("raw_regions"), summarize("dedup_experts")]
    with (out_dir / "metrics_summary.tsv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0].keys()), delimiter="\t")
        writer.writeheader()
        writer.writerows(summaries)
    for row in summaries:
        print(
            f"{row['prediction_kind']} macro_p={row['macro_precision_percent']} "
            f"macro_r={row['macro_recall_percent']} micro_p={row['micro_precision_percent']} "
            f"micro_r={row['micro_recall_percent']} duplicates={row['duplicates']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
