#!/usr/bin/env python3
"""Compute weighted specialization scores for teams.

For expert e and task node v:

    p_e(v) = w(v) * sim_emb(e, v) / sum_v' w(v') * sim_emb(e, v')

The specialization score for a team S is the mean pairwise JSD between member
distributions. Similarities are computed between the task-node embedding
``paper_id::node_id`` and the same expert-node embedding ``expert_id::node_id``.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import random
import pickle
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute weighted team SpecScore")
    p.add_argument("--task-nodes-jsonl", required=True)
    p.add_argument("--task-node-ids", required=True)
    p.add_argument("--task-node-embeddings", required=True)
    p.add_argument("--expert-node-ids", required=True)
    p.add_argument("--expert-node-embeddings", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument(
        "--predictions-tsv",
        action="append",
        default=[],
        help="Prediction TSV with paper_id, rank, expert_id columns. Repeatable.",
    )
    p.add_argument(
        "--prediction-label",
        action="append",
        default=[],
        help="Label for the corresponding --predictions-tsv. Repeatable.",
    )
    p.add_argument(
        "--opentf-token-pred-csv",
        action="append",
        default=[],
        help=(
            "Headerless OpenTF seq2seq prediction CSV where each row contains "
            "space-separated m{candidate_index} tokens. Repeatable."
        ),
    )
    p.add_argument(
        "--opentf-token-label",
        action="append",
        default=[],
        help="Label for the corresponding --opentf-token-pred-csv. Repeatable.",
    )
    p.add_argument(
        "--indexes-pkl",
        default="",
        help="OpenTF indexes.pkl containing i2c, required for token predictions.",
    )
    p.add_argument(
        "--negative-sim",
        choices=("zero", "raw"),
        default="zero",
        help="Use zero to clamp negative similarities before normalization.",
    )
    p.add_argument(
        "--random-baseline",
        action="store_true",
        help=(
            "Add a random baseline sampled per task from experts with at least one "
            "same-node expert embedding for that task."
        ),
    )
    p.add_argument("--random-seed", type=int, default=13)
    return p.parse_args()


def mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def read_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def as_members(value) -> List[str]:
    if isinstance(value, list):
        return [str(x) for x in value if str(x)]
    if isinstance(value, str):
        return [x for x in value.replace("|", ",").split(",") if x]
    return []


def load_task_rows(path: Path) -> Tuple[Dict[str, List[dict]], Dict[str, dict]]:
    rows_by_paper: Dict[str, List[dict]] = defaultdict(list)
    info: Dict[str, dict] = {}
    for row in read_jsonl(path):
        paper_id = str(row["paper_id"])
        node_id = str(row["node_id"])
        row = dict(row)
        row["task_node_id"] = f"{paper_id}::{node_id}"
        row["node_id"] = node_id
        row["node_importance"] = float(row.get("node_importance") or 0.0)
        rows_by_paper[paper_id].append(row)
        info[paper_id] = {
            "team_size": int(row["team_size"]),
            "members": as_members(row.get("members")),
        }
    return rows_by_paper, info


def load_id_to_row(path: Path) -> Dict[str, int]:
    out: Dict[str, int] = {}
    with path.open("r", encoding="utf-8") as f:
        for idx, row in enumerate(csv.DictReader(f, delimiter="\t")):
            out[str(row["id"])] = idx
    return out


def load_needed_expert_rows(
    path: Path, needed_experts: set[str]
) -> Tuple[Dict[Tuple[str, str], int], Dict[str, str]]:
    index: Dict[Tuple[str, str], int] = {}
    names: Dict[str, str] = {}
    with path.open("r", encoding="utf-8") as f:
        for idx, row in enumerate(csv.DictReader(f, delimiter="\t")):
            expert_id = str(row["expert_id"])
            if expert_id not in needed_experts:
                continue
            node_id = str(row["node_id"])
            index[(expert_id, node_id)] = idx
            names.setdefault(expert_id, row.get("expert_name") or expert_id)
    return index, names


def load_experts_by_task_nodes(
    path: Path, needed_nodes: set[str]
) -> Dict[str, List[str]]:
    by_node: Dict[str, set[str]] = defaultdict(set)
    with path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            node_id = str(row["node_id"])
            if node_id not in needed_nodes:
                continue
            by_node[node_id].add(str(row["expert_id"]))
    return {node_id: sorted(expert_ids) for node_id, expert_ids in by_node.items()}


def load_groundtruth_teams(task_info: Dict[str, dict]) -> Dict[str, List[str]]:
    return {paper_id: list(info["members"]) for paper_id, info in task_info.items()}


def load_random_teams(
    rows_by_paper: Dict[str, List[dict]],
    task_info: Dict[str, dict],
    experts_by_node: Dict[str, List[str]],
    seed: int,
) -> Dict[str, List[str]]:
    rng = random.Random(seed)
    teams = {}
    for paper_id, rows in sorted(rows_by_paper.items()):
        pool = sorted(
            {
                expert_id
                for row in rows
                for expert_id in experts_by_node.get(row["node_id"], [])
            }
        )
        team_size = max(1, int(task_info[paper_id]["team_size"]))
        if len(pool) <= team_size:
            teams[paper_id] = pool
        else:
            teams[paper_id] = rng.sample(pool, team_size)
    return teams


def load_prediction_teams(path: Path, task_info: Dict[str, dict]) -> Dict[str, List[str]]:
    grouped: Dict[str, List[Tuple[int, str]]] = defaultdict(list)
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        required = {"paper_id", "expert_id"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"{path} must contain columns: {sorted(required)}")
        for row_idx, row in enumerate(reader, start=1):
            paper_id = str(row.get("paper_id") or "")
            expert_id = str(row.get("expert_id") or "")
            if not paper_id or not expert_id:
                continue
            try:
                rank = int(float(row.get("rank") or row_idx))
            except ValueError:
                rank = row_idx
            grouped[paper_id].append((rank, expert_id))

    teams: Dict[str, List[str]] = {}
    for paper_id, ranked in grouped.items():
        if paper_id not in task_info:
            continue
        team_size = max(1, int(task_info[paper_id]["team_size"]))
        seen = set()
        selected = []
        for _, expert_id in sorted(ranked, key=lambda item: item[0]):
            if expert_id in seen:
                continue
            seen.add(expert_id)
            selected.append(expert_id)
            if len(selected) >= team_size:
                break
        teams[paper_id] = selected
    return teams


def load_opentf_token_teams(
    path: Path,
    indexes_pkl: Path,
    paper_order: Sequence[str],
) -> Dict[str, List[str]]:
    with indexes_pkl.open("rb") as f:
        indexes = pickle.load(f)
    i2c = indexes.get("i2c")
    if not isinstance(i2c, dict):
        raise ValueError(f"{indexes_pkl} does not contain an i2c dictionary")

    teams: Dict[str, List[str]] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        for row_idx, row in enumerate(reader):
            if row_idx >= len(paper_order):
                break
            text = row[0] if row else ""
            selected = []
            seen = set()
            for raw_idx in re.findall(r"m(\d+)", text):
                idname = i2c.get(int(raw_idx))
                if not idname:
                    continue
                expert_id = str(idname).split("_", 1)[0]
                if expert_id in seen:
                    continue
                seen.add(expert_id)
                selected.append(expert_id)
            teams[paper_order[row_idx]] = selected
    return teams


def js_divergence(p: Dict[str, float], q: Dict[str, float]) -> float:
    keys = set(p) | set(q)
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


def expert_distribution(
    expert_id: str,
    rows: List[dict],
    task_id_to_row: Dict[str, int],
    task_arr: np.ndarray,
    expert_index: Dict[Tuple[str, str], int],
    expert_arr: np.ndarray,
    clamp_negative: bool,
) -> Tuple[Dict[str, float], int, int]:
    weighted: Dict[str, float] = {}
    matched = 0
    negative = 0
    for row in rows:
        task_row = task_id_to_row.get(row["task_node_id"])
        expert_row = expert_index.get((expert_id, row["node_id"]))
        if task_row is None or expert_row is None:
            continue
        matched += 1
        sim = float(np.dot(task_arr[task_row], expert_arr[expert_row]))
        if sim < 0:
            negative += 1
            if clamp_negative:
                sim = 0.0
        value = row["node_importance"] * sim
        if value > 0:
            weighted[row["node_id"]] = value
    total = sum(weighted.values())
    if total <= 0:
        return {}, matched, negative
    return {node_id: value / total for node_id, value in weighted.items()}, matched, negative


def specscore_for_team(
    paper_id: str,
    expert_ids: Sequence[str],
    rows_by_paper: Dict[str, List[dict]],
    task_id_to_row: Dict[str, int],
    task_arr: np.ndarray,
    expert_index: Dict[Tuple[str, str], int],
    expert_arr: np.ndarray,
    clamp_negative: bool,
) -> dict:
    rows = rows_by_paper.get(paper_id, [])
    distributions = {}
    matched_nodes = {}
    negative_sims = 0
    for expert_id in expert_ids:
        dist, matched, negative = expert_distribution(
            expert_id,
            rows,
            task_id_to_row,
            task_arr,
            expert_index,
            expert_arr,
            clamp_negative,
        )
        if dist:
            distributions[expert_id] = dist
        matched_nodes[expert_id] = matched
        negative_sims += negative

    pair_scores = []
    for a, b in itertools.combinations(distributions, 2):
        pair_scores.append(js_divergence(distributions[a], distributions[b]))

    return {
        "paper_id": paper_id,
        "team_size": len(expert_ids),
        "experts_with_distribution": len(distributions),
        "valid_pairs": len(pair_scores),
        "possible_pairs": math.comb(len(expert_ids), 2) if len(expert_ids) >= 2 else 0,
        "specscore": mean(pair_scores),
        "avg_matched_nodes_per_expert": mean(list(matched_nodes.values())),
        "negative_similarities": negative_sims,
    }


def summarize(label: str, rows: List[dict]) -> dict:
    total_possible = sum(int(r["possible_pairs"]) for r in rows)
    total_valid = sum(int(r["valid_pairs"]) for r in rows)
    usable = [r for r in rows if int(r["valid_pairs"]) > 0]
    weighted_num = sum(float(r["specscore"]) * int(r["valid_pairs"]) for r in usable)
    return {
        "label": label,
        "tasks": len(rows),
        "tasks_with_valid_pairs": len(usable),
        "mean_specscore_by_task": f"{mean([float(r['specscore']) for r in usable]):.12f}",
        "pair_weighted_specscore": f"{(weighted_num / total_valid) if total_valid else 0.0:.12f}",
        "valid_pairs": total_valid,
        "possible_pairs": total_possible,
        "pair_coverage": f"{(total_valid / total_possible) if total_possible else 0.0:.12f}",
        "avg_experts_with_distribution": f"{mean([int(r['experts_with_distribution']) for r in rows]):.6f}",
        "avg_matched_nodes_per_expert": f"{mean([float(r['avg_matched_nodes_per_expert']) for r in rows]):.6f}",
        "negative_similarities": sum(int(r["negative_similarities"]) for r in rows),
    }


def write_tsv(path: Path, rows: List[dict], fieldnames: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if args.prediction_label and len(args.prediction_label) != len(args.predictions_tsv):
        raise ValueError("--prediction-label count must match --predictions-tsv count")
    if args.opentf_token_label and len(args.opentf_token_label) != len(
        args.opentf_token_pred_csv
    ):
        raise ValueError("--opentf-token-label count must match --opentf-token-pred-csv count")
    if args.opentf_token_pred_csv and not args.indexes_pkl:
        raise ValueError("--indexes-pkl is required with --opentf-token-pred-csv")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("loading task rows", flush=True)
    rows_by_paper, task_info = load_task_rows(Path(args.task_nodes_jsonl))
    paper_order = list(rows_by_paper)
    teams_by_label = {"groundtruth": load_groundtruth_teams(task_info)}

    if args.random_baseline:
        needed_nodes = {row["node_id"] for rows in rows_by_paper.values() for row in rows}
        print(
            f"loading random eligible pools for {len(needed_nodes):,} task nodes",
            flush=True,
        )
        experts_by_node = load_experts_by_task_nodes(Path(args.expert_node_ids), needed_nodes)
        teams_by_label["random"] = load_random_teams(
            rows_by_paper, task_info, experts_by_node, args.random_seed
        )

    for idx, pred_path in enumerate(args.predictions_tsv):
        label = (
            args.prediction_label[idx]
            if args.prediction_label
            else Path(pred_path).parent.name or Path(pred_path).stem
        )
        teams_by_label[label] = load_prediction_teams(Path(pred_path), task_info)

    for idx, pred_path in enumerate(args.opentf_token_pred_csv):
        label = (
            args.opentf_token_label[idx]
            if args.opentf_token_label
            else Path(pred_path).stem
        )
        teams_by_label[label] = load_opentf_token_teams(
            Path(pred_path), Path(args.indexes_pkl), paper_order
        )

    needed_experts = {
        expert_id
        for teams in teams_by_label.values()
        for team in teams.values()
        for expert_id in team
    }

    print(f"loading task embedding index for {len(rows_by_paper):,} tasks", flush=True)
    task_id_to_row = load_id_to_row(Path(args.task_node_ids))
    task_arr = np.load(args.task_node_embeddings, mmap_mode="r")

    print(f"loading expert index for {len(needed_experts):,} selected experts", flush=True)
    expert_index, _ = load_needed_expert_rows(Path(args.expert_node_ids), needed_experts)
    expert_arr = np.load(args.expert_node_embeddings, mmap_mode="r")

    clamp_negative = args.negative_sim == "zero"
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
    for label, teams in teams_by_label.items():
        print(f"computing label={label} tasks={len(teams):,}", flush=True)
        detail_rows = [
            specscore_for_team(
                paper_id,
                team,
                rows_by_paper,
                task_id_to_row,
                task_arr,
                expert_index,
                expert_arr,
                clamp_negative,
            )
            for paper_id, team in sorted(teams.items())
        ]
        for row in detail_rows:
            row["specscore"] = f"{float(row['specscore']):.12f}"
            row["avg_matched_nodes_per_expert"] = (
                f"{float(row['avg_matched_nodes_per_expert']):.6f}"
            )
        write_tsv(out_dir / f"{label}.task_specscore.tsv", detail_rows, detail_fields)
        summary_rows.append(summarize(label, detail_rows))

    summary_fields = list(summary_rows[0])
    write_tsv(out_dir / "summary.tsv", summary_rows, summary_fields)
    print(f"summary={out_dir / 'summary.tsv'}", flush=True)


if __name__ == "__main__":
    main()
