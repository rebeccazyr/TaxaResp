#!/usr/bin/env python3
"""Evaluate test-paper abstract search against mean expert paper embeddings.

For each test paper, this script ranks experts by cosine similarity between the
test paper abstract embedding and each expert's normalized mean embedding over
all historical paper abstracts. It retrieves top-k experts where k is the
groundtruth team size for that test paper, then reports exact-member P/R/F1.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


METHOD = "test_abstract_to_expert_all_paper_mean_embedding_top_gold_team_size"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task-nodes-jsonl", default="output/hierec_embedding_server_inputs/task_nodes.jsonl")
    p.add_argument(
        "--query-ids-tsv",
        default="output/virtual_root_role_descriptions/task_paper_text_embedding_ids.tsv",
        help="TSV whose id column is aligned to query-embeddings rows.",
    )
    p.add_argument(
        "--query-embeddings",
        default="output/virtual_root_role_descriptions/task_paper_text_embeddings.npy",
        help="Test paper title/abstract embedding matrix.",
    )
    p.add_argument(
        "--expert-ids-tsv",
        default="output/virtual_root_role_descriptions/expert_mean_paper_embedding_ids.tsv",
        help="TSV whose id column is aligned to expert-embeddings rows.",
    )
    p.add_argument(
        "--expert-embeddings",
        default="output/virtual_root_role_descriptions/expert_mean_paper_embeddings.npy",
        help="Expert all-paper mean abstract embedding matrix.",
    )
    p.add_argument("--out-dir", default="output/abstract_expert_mean_embedding_search")
    p.add_argument("--score-chunk-size", type=int, default=64)
    return p.parse_args()


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def f1(precision: float, recall: float) -> float:
    denom = precision + recall
    return 2.0 * precision * recall / denom if denom else 0.0


def dedupe(values: Iterable[str]) -> List[str]:
    seen = set()
    out = []
    for value in values:
        value = str(value)
        if value and value not in seen:
            seen.add(value)
            out.append(value)
    return out


def as_members(value) -> List[str]:
    if isinstance(value, list):
        return [str(x) for x in value if str(x)]
    if isinstance(value, str):
        return [x for x in value.replace("|", ",").split(",") if x]
    return []


def read_tasks(path: Path) -> Tuple[List[str], Dict[str, List[str]], Dict[str, int]]:
    paper_order: List[str] = []
    members_by_paper: Dict[str, List[str]] = {}
    team_size_by_paper: Dict[str, int] = {}
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
            members = dedupe(as_members(row.get("members")))
            paper_order.append(paper_id)
            members_by_paper[paper_id] = members
            team_size_by_paper[paper_id] = int(row.get("team_size") or len(members) or 1)
    return paper_order, members_by_paper, team_size_by_paper


def read_embedding_id_rows(path: Path) -> Tuple[Dict[str, int], Dict[str, dict]]:
    id_to_row: Dict[str, int] = {}
    metadata_by_id: Dict[str, dict] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for idx, row in enumerate(reader):
            item_id = str(row.get("id") or row.get("paper_id") or row.get("expert_id") or "")
            if item_id:
                id_to_row[item_id] = idx
                metadata_by_id[item_id] = dict(row)
            if (idx + 1) % 250000 == 0:
                print(f"embedding_id_progress rows={idx + 1:,}", flush=True)
    return id_to_row, metadata_by_id


def normalize_rows(arr: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.maximum(norms, 1e-12)


def summarize(task_rows: Sequence[dict]) -> dict:
    micro_hits = sum(row["hits"] for row in task_rows)
    micro_predicted = sum(row["predicted"] for row in task_rows)
    micro_gold = sum(row["gold"] for row in task_rows)
    micro_precision = micro_hits / micro_predicted if micro_predicted else 0.0
    micro_recall = micro_hits / micro_gold if micro_gold else 0.0
    micro_f1 = f1(micro_precision, micro_recall)
    return {
        "method": METHOD,
        "tasks": len(task_rows),
        "macro_precision": f"{mean([row['precision'] for row in task_rows]):.12f}",
        "macro_recall": f"{mean([row['recall'] for row in task_rows]):.12f}",
        "macro_f1": f"{mean([row['f1'] for row in task_rows]):.12f}",
        "percent_precision": f"{100 * mean([row['precision'] for row in task_rows]):.6f}",
        "percent_recall": f"{100 * mean([row['recall'] for row in task_rows]):.6f}",
        "percent_f1": f"{100 * mean([row['f1'] for row in task_rows]):.6f}",
        "micro_precision": f"{micro_precision:.12f}",
        "micro_recall": f"{micro_recall:.12f}",
        "micro_f1": f"{micro_f1:.12f}",
        "micro_hits": micro_hits,
        "micro_predicted": micro_predicted,
        "micro_gold": micro_gold,
        "avg_predicted": f"{mean([row['predicted'] for row in task_rows]):.6f}",
        "avg_gold": f"{mean([row['gold'] for row in task_rows]):.6f}",
    }


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    paper_order, members_by_paper, team_size_by_paper = read_tasks(Path(args.task_nodes_jsonl))

    print("indexing query embedding ids", flush=True)
    query_to_row, _ = read_embedding_id_rows(Path(args.query_ids_tsv))
    missing_tasks = [paper_id for paper_id in paper_order if paper_id not in query_to_row]
    if missing_tasks:
        raise SystemExit(f"Missing task paper embeddings for {len(missing_tasks)} papers, e.g. {missing_tasks[:5]}")

    print("indexing expert embedding ids", flush=True)
    expert_to_row, expert_meta = read_embedding_id_rows(Path(args.expert_ids_tsv))

    print("opening query and expert embeddings", flush=True)
    query_arr = np.load(args.query_embeddings, mmap_mode="r")
    expert_arr = np.load(args.expert_embeddings, mmap_mode="r")
    if len(query_to_row) != query_arr.shape[0]:
        raise SystemExit(
            f"query id/embedding row mismatch: {len(query_to_row)} ids vs {query_arr.shape[0]} embeddings"
        )
    if len(expert_to_row) != expert_arr.shape[0]:
        raise SystemExit(
            f"expert id/embedding row mismatch: {len(expert_to_row)} ids vs {expert_arr.shape[0]} embeddings"
        )
    if query_arr.shape[1] != expert_arr.shape[1]:
        raise SystemExit(f"Embedding dimension mismatch: query={query_arr.shape[1]} expert={expert_arr.shape[1]}")

    task_embedding_rows = [query_to_row[paper_id] for paper_id in paper_order]
    task_norms = np.linalg.norm(np.asarray(query_arr[task_embedding_rows], dtype=np.float32), axis=1)
    zero_task_count = int((task_norms <= 0.0).sum())
    if zero_task_count:
        raise SystemExit(f"Invalid query embeddings: {zero_task_count}/{len(task_embedding_rows)} task vectors are zero.")

    expert_ids = sorted(expert_to_row, key=lambda expert_id: expert_to_row[expert_id])
    expert_mat = normalize_rows(np.asarray(expert_arr[[expert_to_row[expert_id] for expert_id in expert_ids]], dtype=np.float32))
    expert_paper_counts = {
        expert_id: int(expert_meta.get(expert_id, {}).get("matched_paper_count") or expert_meta.get(expert_id, {}).get("paper_count") or 0)
        for expert_id in expert_ids
    }
    zero_expert_count = int((np.linalg.norm(expert_mat, axis=1) <= 0.0).sum())
    if zero_expert_count:
        raise SystemExit(f"Invalid expert embeddings: {zero_expert_count}/{len(expert_ids)} expert vectors are zero.")

    task_mat = normalize_rows(np.asarray(query_arr[task_embedding_rows], dtype=np.float32))
    prediction_rows = []
    per_task_rows = []

    print("ranking experts", flush=True)
    for start in range(0, len(paper_order), args.score_chunk_size):
        end = min(start + args.score_chunk_size, len(paper_order))
        scores = task_mat[start:end] @ expert_mat.T
        for local_idx, paper_id in enumerate(paper_order[start:end]):
            k = min(max(1, int(team_size_by_paper.get(paper_id, 1))), len(expert_ids))
            row_scores = scores[local_idx]
            if k < len(expert_ids):
                top_pos = np.argpartition(-row_scores, k - 1)[:k]
                top_pos = top_pos[np.argsort(-row_scores[top_pos])]
            else:
                top_pos = np.argsort(-row_scores)

            selected = [expert_ids[pos] for pos in top_pos]
            golds = set(dedupe(members_by_paper.get(paper_id, [])))
            hits = len(golds.intersection(selected))
            precision = hits / len(selected) if selected else 0.0
            recall = hits / len(golds) if golds else 0.0
            row_f1 = f1(precision, recall)
            per_task_rows.append(
                {
                    "method": METHOD,
                    "paper_id": paper_id,
                    "team_size": team_size_by_paper.get(paper_id, len(golds)),
                    "hits": hits,
                    "predicted": len(selected),
                    "gold": len(golds),
                    "precision": f"{precision:.12f}",
                    "recall": f"{recall:.12f}",
                    "f1": f"{row_f1:.12f}",
                }
            )

            for rank, pos in enumerate(top_pos, start=1):
                expert_id = expert_ids[pos]
                prediction_rows.append(
                    {
                        "method": METHOD,
                        "paper_id": paper_id,
                        "rank": rank,
                        "expert_id": expert_id,
                        "score": f"{float(row_scores[pos]):.9f}",
                        "expert_all_paper_count": expert_paper_counts.get(expert_id, 0),
                        "is_actual_member": "1" if expert_id in golds else "0",
                    }
                )

    metric_row = summarize(
        [
            {
                **row,
                "hits": int(row["hits"]),
                "predicted": int(row["predicted"]),
                "gold": int(row["gold"]),
                "precision": float(row["precision"]),
                "recall": float(row["recall"]),
                "f1": float(row["f1"]),
            }
            for row in per_task_rows
        ]
    )

    with (out_dir / "predictions_team_size.tsv").open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "method",
            "paper_id",
            "rank",
            "expert_id",
            "score",
            "expert_all_paper_count",
            "is_actual_member",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(prediction_rows)

    with (out_dir / "per_task_metrics.tsv").open("w", encoding="utf-8", newline="") as f:
        fieldnames = ["method", "paper_id", "team_size", "hits", "predicted", "gold", "precision", "recall", "f1"]
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(per_task_rows)

    with (out_dir / "metrics_summary.tsv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(metric_row), delimiter="\t")
        writer.writeheader()
        writer.writerow(metric_row)

    with (out_dir / "run_summary.tsv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "value"], delimiter="\t")
        writer.writeheader()
        writer.writerows(
            [
                {"metric": "tasks", "value": len(paper_order)},
                {"metric": "embedding_dim", "value": query_arr.shape[1]},
                {"metric": "query_embedding_rows", "value": query_arr.shape[0]},
                {"metric": "expert_embedding_rows", "value": expert_arr.shape[0]},
                {"metric": "experts_with_mean_embeddings", "value": len(expert_ids)},
                {"metric": "expert_paper_pairs", "value": sum(expert_paper_counts.values())},
                {"metric": "query_embedding_scope", "value": str(args.query_embeddings)},
                {"metric": "expert_embedding_scope", "value": str(args.expert_embeddings)},
                {"metric": "quantity_matching", "value": "top-k equals groundtruth team size per paper"},
            ]
        )

    print(
        f"{METHOD} p={float(metric_row['percent_precision']):.4f}% "
        f"r={float(metric_row['percent_recall']):.4f}% "
        f"f1={float(metric_row['percent_f1']):.4f}% "
        f"hits={metric_row['micro_hits']}/{metric_row['micro_gold']}",
        flush=True,
    )
    print(f"out_dir={out_dir}", flush=True)


if __name__ == "__main__":
    main()
