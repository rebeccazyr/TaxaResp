#!/usr/bin/env python3
"""Diagnose whether root-role queries align with expert mean-paper embeddings."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task-nodes-jsonl", default="output/hierec_embedding_server_inputs/task_nodes.jsonl")
    p.add_argument(
        "--expert-ids",
        default="output/virtual_root_role_descriptions/expert_mean_paper_embedding_ids.tsv",
    )
    p.add_argument(
        "--expert-embeddings",
        default="output/virtual_root_role_descriptions/expert_mean_paper_embeddings.npy",
    )
    p.add_argument(
        "--paper-ids",
        default="output/all_expert_paper_embeddings/paper_embedding_ids.tsv",
    )
    p.add_argument(
        "--paper-embeddings",
        default="output/all_expert_paper_embeddings/paper_embeddings.npy",
    )
    p.add_argument(
        "--query",
        action="append",
        default=[],
        metavar="LABEL:IDS_TSV:EMB_NPY",
        help="Additional query embedding table keyed by paper_id.",
    )
    p.add_argument(
        "--out-tsv",
        default="output/virtual_root_role_descriptions/root_query_expert_mean_diagnostics.tsv",
    )
    p.add_argument(
        "--summary-tsv",
        default="output/virtual_root_role_descriptions/root_query_expert_mean_diagnostics_summary.tsv",
    )
    return p.parse_args()


def read_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def read_tasks(path: Path) -> tuple[List[str], Dict[str, List[str]]]:
    order: List[str] = []
    members: Dict[str, List[str]] = {}
    seen = set()
    for row in read_jsonl(path):
        paper_id = str(row["paper_id"])
        if paper_id in seen:
            continue
        seen.add(paper_id)
        order.append(paper_id)
        members[paper_id] = [str(x) for x in row.get("members") or []]
    return order, members


def read_ids(path: Path) -> List[str]:
    ids: List[str] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            ids.append(str(row["id"]))
    return ids


def normalize_rows(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.maximum(norms, 1e-12)


def parse_query_specs(items: Sequence[str]) -> List[tuple[str, Path, Path]]:
    out = []
    for item in items:
        parts = item.split(":", 2)
        if len(parts) != 3:
            raise SystemExit(f"--query must be LABEL:IDS_TSV:EMB_NPY: {item}")
        out.append((parts[0], Path(parts[1]), Path(parts[2])))
    return out


def percentile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.array(values, dtype=np.float64), q))


def main() -> None:
    args = parse_args()
    paper_order, members = read_tasks(Path(args.task_nodes_jsonl))
    expert_ids = read_ids(Path(args.expert_ids))
    expert_to_idx = {expert_id: idx for idx, expert_id in enumerate(expert_ids)}
    expert_arr = normalize_rows(np.load(args.expert_embeddings, mmap_mode="r"))

    query_specs = parse_query_specs(args.query)
    query_specs.append(("paper_abstract", Path(args.paper_ids), Path(args.paper_embeddings)))

    out_path = Path(args.out_tsv)
    summary_path = Path(args.summary_tsv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    summary_rows = []

    for label, ids_path, emb_path in query_specs:
        query_ids = read_ids(ids_path)
        query_to_idx = {query_id: idx for idx, query_id in enumerate(query_ids)}
        query_arr = normalize_rows(np.load(emb_path, mmap_mode="r"))

        best_ranks: List[int] = []
        best_gold_scores: List[float] = []
        top1_scores: List[float] = []
        top1_hits = 0
        top5_any = 0
        top20_any = 0
        missing_query = 0
        missing_gold = 0

        for paper_id in paper_order:
            qidx = query_to_idx.get(paper_id)
            golds = [g for g in dict.fromkeys(members.get(paper_id, [])) if g in expert_to_idx]
            if qidx is None:
                missing_query += 1
                continue
            if not golds:
                missing_gold += 1
                continue

            qvec = query_arr[qidx]
            scores = np.asarray(expert_arr @ qvec, dtype=np.float32)
            order_desc = np.argsort(-scores)
            ranks_by_expert_idx = np.empty_like(order_desc)
            ranks_by_expert_idx[order_desc] = np.arange(1, len(order_desc) + 1)
            gold_indices = [expert_to_idx[g] for g in golds]
            gold_ranks = [int(ranks_by_expert_idx[idx]) for idx in gold_indices]
            gold_scores = [float(scores[idx]) for idx in gold_indices]
            best_rank = min(gold_ranks)
            best_score = max(gold_scores)
            top1_idx = int(order_desc[0])
            top1_expert = expert_ids[top1_idx]
            top1_score = float(scores[top1_idx])
            top20 = {expert_ids[int(i)] for i in order_desc[:20]}
            top5 = {expert_ids[int(i)] for i in order_desc[:5]}

            hit1 = top1_expert in golds
            hit5 = bool(top5.intersection(golds))
            hit20 = bool(top20.intersection(golds))
            top1_hits += int(hit1)
            top5_any += int(hit5)
            top20_any += int(hit20)
            best_ranks.append(best_rank)
            best_gold_scores.append(best_score)
            top1_scores.append(top1_score)
            rows.append(
                {
                    "query_label": label,
                    "paper_id": paper_id,
                    "team_size": len(golds),
                    "top1_expert": top1_expert,
                    "top1_score": f"{top1_score:.9f}",
                    "top1_is_gold": int(hit1),
                    "top5_has_gold": int(hit5),
                    "top20_has_gold": int(hit20),
                    "best_gold_rank": best_rank,
                    "best_gold_score": f"{best_score:.9f}",
                    "score_gap_top1_minus_best_gold": f"{top1_score - best_score:.9f}",
                }
            )

        n = len(best_ranks)
        summary_rows.append(
            {
                "query_label": label,
                "tasks_evaluated": n,
                "missing_query": missing_query,
                "missing_gold": missing_gold,
                "top1_tasks_any_gold": top1_hits,
                "top5_tasks_any_gold": top5_any,
                "top20_tasks_any_gold": top20_any,
                "top1_task_rate": f"{top1_hits / n * 100 if n else 0:.3f}",
                "top5_task_rate": f"{top5_any / n * 100 if n else 0:.3f}",
                "top20_task_rate": f"{top20_any / n * 100 if n else 0:.3f}",
                "median_best_gold_rank": f"{percentile(best_ranks, 50):.3f}",
                "p75_best_gold_rank": f"{percentile(best_ranks, 75):.3f}",
                "p90_best_gold_rank": f"{percentile(best_ranks, 90):.3f}",
                "mean_best_gold_score": f"{np.mean(best_gold_scores) if best_gold_scores else 0:.6f}",
                "mean_top1_score": f"{np.mean(top1_scores) if top1_scores else 0:.6f}",
                "mean_top1_minus_best_gold": f"{np.mean(np.array(top1_scores) - np.array(best_gold_scores)) if best_gold_scores else 0:.6f}",
            }
        )

    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    with summary_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()), delimiter="\t")
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"wrote={out_path} rows={len(rows)}")
    print(f"wrote={summary_path} rows={len(summary_rows)}")


if __name__ == "__main__":
    main()
