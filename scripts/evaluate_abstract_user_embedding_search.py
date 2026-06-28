#!/usr/bin/env python3
"""Evaluate direct test-abstract to aggregated-user embedding search.

This script uses the embedding-server cache where paper_embeddings.npy rows are
aligned to paper_texts.jsonl line order. For each expert, it averages unique
historical evidence-paper embeddings from expert_node_evidence.jsonl, normalizes
the mean, then ranks experts by cosine similarity to each test paper abstract.

The number of retrieved experts per task is matched to a reference prediction
file, typically the BFS unique-assignment predictions.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


METHOD_DIRECT = "abstract_user_mean_embedding_search_qwen2560_evidence_papers"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task-nodes-jsonl", default="output/hierec_embedding_server_inputs/task_nodes.jsonl")
    p.add_argument("--paper-texts-jsonl", default="output/hierec_embedding_server_inputs/paper_texts.jsonl")
    p.add_argument("--paper-embeddings", default="output/hierec_embedding_server_inputs/paper_embeddings.npy")
    p.add_argument("--expert-node-evidence-jsonl", default="output/hierec_embedding_server_inputs/expert_node_evidence.jsonl")
    p.add_argument("--reference-predictions", default="output/embedding_bfs_unique_assignment_no_label/predictions_team_size.tsv")
    p.add_argument(
        "--reference-method",
        default="embedding_bfs_unique_assign_each_node_then_top_team_size_by_weighted_score",
    )
    p.add_argument("--out-dir", default="output/abstract_user_embedding_search_qwen2560")
    p.add_argument("--score-chunk-size", type=int, default=64)
    return p.parse_args()


def mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


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
        return [str(x) for x in value]
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
            paper_order.append(paper_id)
            members_by_paper[paper_id] = as_members(row.get("members"))
            team_size_by_paper[paper_id] = int(row.get("team_size") or len(members_by_paper[paper_id]) or 1)
    return paper_order, members_by_paper, team_size_by_paper


def read_paper_text_rows(path: Path) -> Dict[str, int]:
    out: Dict[str, int] = {}
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if not line.strip():
                continue
            obj = json.loads(line)
            paper_id = str(obj.get("paper_id") or obj.get("id") or "")
            if paper_id:
                out[paper_id] = idx
            if (idx + 1) % 250000 == 0:
                print(f"paper_text_index_progress rows={idx + 1:,}", flush=True)
    return out


def load_reference_predictions(path: Path, method: str) -> Dict[str, List[str]]:
    out: Dict[str, List[Tuple[int, str]]] = defaultdict(list)
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if (row.get("method") or "") != method:
                continue
            try:
                rank = int(float(row.get("rank") or 0))
            except ValueError:
                rank = 0
            out[str(row["paper_id"])].append((rank, str(row["expert_id"])))
    return {paper_id: dedupe(expert for _, expert in sorted(rows)) for paper_id, rows in out.items()}


def collect_expert_papers(path: Path, paper_to_row: Dict[str, int]) -> Dict[str, set]:
    expert_to_papers: Dict[str, set] = defaultdict(set)
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            expert_id = str(row["expert_id"])
            for paper in row.get("papers") or []:
                paper_id = str(paper.get("paper_id") or "")
                if paper_id in paper_to_row:
                    expert_to_papers[expert_id].add(paper_id)
            if idx % 250000 == 0:
                print(f"expert_evidence_progress rows={idx:,} experts={len(expert_to_papers):,}", flush=True)
    return expert_to_papers


def normalize_rows(arr: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.maximum(norms, 1e-12)


def build_user_embedding_matrix(
    paper_arr: np.ndarray,
    paper_to_row: Dict[str, int],
    expert_to_papers: Dict[str, set],
) -> Tuple[List[str], np.ndarray, Dict[str, int]]:
    expert_ids: List[str] = []
    vectors: List[np.ndarray] = []
    paper_counts: Dict[str, int] = {}
    for idx, (expert_id, papers) in enumerate(sorted(expert_to_papers.items()), start=1):
        rows = [paper_to_row[paper_id] for paper_id in papers if paper_id in paper_to_row]
        if not rows:
            continue
        vec = np.asarray(paper_arr[rows], dtype=np.float32).mean(axis=0)
        norm = float(np.linalg.norm(vec))
        if norm <= 0.0:
            continue
        expert_ids.append(expert_id)
        vectors.append((vec / norm).astype(np.float32))
        paper_counts[expert_id] = len(rows)
        if idx % 1000 == 0:
            print(f"user_embedding_progress experts={idx:,}/{len(expert_to_papers):,}", flush=True)
    return expert_ids, np.vstack(vectors).astype(np.float32), paper_counts


def summarize(
    method: str,
    paper_order: Sequence[str],
    members_by_paper: Dict[str, List[str]],
    predictions_by_paper: Dict[str, List[str]],
) -> dict:
    task_rows = []
    for paper_id in paper_order:
        golds = set(dedupe(members_by_paper.get(paper_id, [])))
        preds = dedupe(predictions_by_paper.get(paper_id, []))
        hits = len(golds.intersection(preds))
        task_rows.append(
            {
                "hits": hits,
                "predicted": len(preds),
                "gold": len(golds),
                "precision": hits / len(preds) if preds else 0.0,
                "recall": hits / len(golds) if golds else 0.0,
            }
        )
    micro_hits = sum(row["hits"] for row in task_rows)
    micro_pred = sum(row["predicted"] for row in task_rows)
    micro_gold = sum(row["gold"] for row in task_rows)
    return {
        "method": method,
        "tasks": len(task_rows),
        "macro_precision": f"{mean([row['precision'] for row in task_rows]):.12f}",
        "macro_recall": f"{mean([row['recall'] for row in task_rows]):.12f}",
        "percent_precision": f"{100 * mean([row['precision'] for row in task_rows]):.6f}",
        "percent_recall": f"{100 * mean([row['recall'] for row in task_rows]):.6f}",
        "micro_precision": f"{(micro_hits / micro_pred) if micro_pred else 0.0:.12f}",
        "micro_recall": f"{(micro_hits / micro_gold) if micro_gold else 0.0:.12f}",
        "micro_hits": micro_hits,
        "micro_predicted": micro_pred,
        "micro_gold": micro_gold,
        "avg_predicted": f"{mean([row['predicted'] for row in task_rows]):.6f}",
        "avg_gold": f"{mean([row['gold'] for row in task_rows]):.6f}",
    }


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    paper_order, members_by_paper, team_size_by_paper = read_tasks(Path(args.task_nodes_jsonl))
    reference_predictions = load_reference_predictions(Path(args.reference_predictions), args.reference_method)
    reference_counts = {
        paper_id: len(reference_predictions.get(paper_id, [])) or max(1, int(team_size_by_paper.get(paper_id, 1)))
        for paper_id in paper_order
    }

    print("indexing paper_texts jsonl", flush=True)
    paper_to_row = read_paper_text_rows(Path(args.paper_texts_jsonl))
    missing_tasks = [paper_id for paper_id in paper_order if paper_id not in paper_to_row]
    if missing_tasks:
        raise SystemExit(f"Missing task paper embeddings for {len(missing_tasks)} papers, e.g. {missing_tasks[:5]}")

    print("opening paper embeddings", flush=True)
    paper_arr = np.load(args.paper_embeddings, mmap_mode="r")
    if len(paper_to_row) != paper_arr.shape[0]:
        raise SystemExit(
            f"paper_texts/embedding row mismatch: {len(paper_to_row)} text rows vs {paper_arr.shape[0]} embeddings"
        )

    task_rows = [paper_to_row[paper_id] for paper_id in paper_order]
    task_norms = np.linalg.norm(np.asarray(paper_arr[task_rows], dtype=np.float32), axis=1)
    zero_task_count = int((task_norms <= 0.0).sum())
    if zero_task_count:
        raise SystemExit(
            "Invalid paper embedding cache: "
            f"{zero_task_count}/{len(task_rows)} task abstract embeddings are zero vectors. "
            "Regenerate paper_embeddings.npy before evaluating direct abstract search."
        )

    print("collecting expert evidence papers", flush=True)
    expert_to_papers = collect_expert_papers(Path(args.expert_node_evidence_jsonl), paper_to_row)
    print("building user embedding matrix", flush=True)
    expert_ids, user_mat, paper_counts = build_user_embedding_matrix(paper_arr, paper_to_row, expert_to_papers)
    if len(expert_ids) < len(expert_to_papers):
        missing = len(expert_to_papers) - len(expert_ids)
        raise SystemExit(
            "Invalid paper embedding cache: "
            f"{missing}/{len(expert_to_papers)} experts have only zero-vector evidence embeddings. "
            "Regenerate paper_embeddings.npy before evaluating direct abstract search."
        )

    task_mat = normalize_rows(np.asarray(paper_arr[task_rows], dtype=np.float32))

    predictions_by_paper: Dict[str, List[str]] = {}
    prediction_rows = []
    print("ranking experts", flush=True)
    for start in range(0, len(paper_order), args.score_chunk_size):
        end = min(start + args.score_chunk_size, len(paper_order))
        scores = task_mat[start:end] @ user_mat.T
        for local_idx, paper_id in enumerate(paper_order[start:end]):
            k = min(reference_counts[paper_id], len(expert_ids))
            row_scores = scores[local_idx]
            if k < len(expert_ids):
                top_pos = np.argpartition(-row_scores, k - 1)[:k]
                top_pos = top_pos[np.argsort(-row_scores[top_pos])]
            else:
                top_pos = np.argsort(-row_scores)
            selected = [expert_ids[pos] for pos in top_pos]
            predictions_by_paper[paper_id] = selected
            golds = set(members_by_paper.get(paper_id, []))
            for rank, pos in enumerate(top_pos, start=1):
                expert_id = expert_ids[pos]
                prediction_rows.append(
                    {
                        "method": METHOD_DIRECT,
                        "paper_id": paper_id,
                        "rank": rank,
                        "expert_id": expert_id,
                        "score": f"{float(row_scores[pos]):.9f}",
                        "user_evidence_papers": paper_counts.get(expert_id, 0),
                        "is_actual_member": "1" if expert_id in golds else "0",
                    }
                )

    metric_rows = [
        summarize(METHOD_DIRECT, paper_order, members_by_paper, predictions_by_paper),
        summarize(args.reference_method, paper_order, members_by_paper, reference_predictions),
    ]

    with (out_dir / "predictions_team_size.tsv").open("w", encoding="utf-8", newline="") as f:
        fieldnames = ["method", "paper_id", "rank", "expert_id", "score", "user_evidence_papers", "is_actual_member"]
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(prediction_rows)

    with (out_dir / "metrics_summary.tsv").open("w", encoding="utf-8", newline="") as f:
        fieldnames = list(metric_rows[0])
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(metric_rows)

    with (out_dir / "run_summary.tsv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "value"], delimiter="\t")
        writer.writeheader()
        writer.writerows(
            [
                {"metric": "tasks", "value": len(paper_order)},
                {"metric": "paper_embedding_dim", "value": paper_arr.shape[1]},
                {"metric": "paper_text_rows", "value": len(paper_to_row)},
                {"metric": "experts_with_user_embeddings", "value": len(expert_ids)},
                {"metric": "expert_evidence_scope", "value": "unique papers from expert_node_evidence.jsonl"},
                {"metric": "quantity_matching", "value": f"per-paper count from {args.reference_method}"},
            ]
        )

    for row in metric_rows:
        print(
            f"{row['method']} p={float(row['percent_precision']):.4f}% "
            f"r={float(row['percent_recall']):.4f}% "
            f"hits={row['micro_hits']}/{row['micro_gold']}",
            flush=True,
        )
    print(f"out_dir={out_dir}", flush=True)


if __name__ == "__main__":
    main()
