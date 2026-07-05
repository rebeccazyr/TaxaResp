#!/usr/bin/env python3
"""Evaluate trained Stage-1 projections with whole-paper to whole-expert search."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate_stage1_full_node_assignment import load_model, safe_div
from scripts.train_stage1 import canonical_dev_paths, canonical_official_test_paths
from src.stage1_smoke_training import Stage1Dataset, Stage1Task


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        default="outputs/stage1_training/task_expert_node_dev_and_test_v1/checkpoint_last.pt",
    )
    parser.add_argument("--out-dir", default="outputs/stage1_projected_whole_mean_search/task_expert_node_v1")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-papers", type=int, default=0)
    parser.add_argument("--skip-dev", action="store_true")
    parser.add_argument("--skip-official-test", action="store_true")
    parser.add_argument(
        "--query-mode",
        choices=("mean_nodes",),
        default="mean_nodes",
        help="mean_nodes averages all raw node role embeddings for each paper before role projection.",
    )
    return parser.parse_args()


def write_tsv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def projected_expert_mean_matrix(
    dataset: Stage1Dataset,
    model,
    device: torch.device,
) -> tuple[list[str], np.ndarray]:
    author_ids: list[str] = []
    raw_means: list[np.ndarray] = []
    for author_id in dataset.author_ids:
        indices = dataset.author_history_indices.get(author_id, [])
        if not indices:
            continue
        vec = np.asarray(dataset.history_embeddings[indices], dtype=np.float32).mean(axis=0)
        if float(np.linalg.norm(vec)) <= 1e-12:
            continue
        author_ids.append(author_id)
        raw_means.append(vec)
    if not raw_means:
        raise SystemExit("No expert history means available")
    with torch.no_grad():
        raw = torch.from_numpy(np.vstack(raw_means).astype(np.float32)).to(device)
        projected = F.normalize(model.expert_proj(raw), dim=-1).cpu().numpy().astype(np.float32)
    return author_ids, projected


def projected_task_query(
    task: Stage1Task,
    dataset: Stage1Dataset,
    model,
    device: torch.device,
) -> np.ndarray | None:
    role_indices = [dataset.role_id_to_idx[role_id] for role_id in task.role_record_ids]
    if not role_indices:
        return None
    raw_query = np.asarray(dataset.role_embeddings[role_indices], dtype=np.float32).mean(axis=0)
    if float(np.linalg.norm(raw_query)) <= 1e-12:
        return None
    with torch.no_grad():
        query = torch.from_numpy(raw_query.astype(np.float32)).to(device).unsqueeze(0)
        projected = F.normalize(model.role_proj(query), dim=-1).squeeze(0).cpu().numpy()
    return projected.astype(np.float32)


def evaluate_split(
    split_name: str,
    dataset: Stage1Dataset,
    model,
    device: torch.device,
    out_dir: Path,
    query_mode: str,
) -> dict[str, object]:
    split_dir = out_dir / split_name
    split_dir.mkdir(parents=True, exist_ok=True)
    expert_ids, expert_matrix = projected_expert_mean_matrix(dataset, model, device)

    pred_rows: list[dict[str, object]] = []
    task_rows: list[dict[str, object]] = []
    macro_p: list[float] = []
    macro_r: list[float] = []
    macro_f1: list[float] = []
    total_pred = 0
    total_gold = 0
    total_hits = 0
    missing_query = 0

    for task in dataset.tasks:
        query = projected_task_query(task, dataset, model, device)
        if query is None:
            missing_query += 1
            continue
        top_k = max(1, len(task.role_record_ids))
        scores = expert_matrix @ query
        k = min(top_k, len(expert_ids))
        if len(scores) <= k:
            top_indices = np.argsort(-scores)
        else:
            top_indices = np.argpartition(-scores, k - 1)[:k]
            top_indices = top_indices[np.argsort(-scores[top_indices])]
        preds = [expert_ids[int(idx)] for idx in top_indices]
        pred_set = set(preds)
        gold = set(task.author_ids)
        hits = len(pred_set & gold)
        precision = safe_div(hits, len(pred_set))
        recall = safe_div(hits, len(gold))
        f1 = safe_div(2.0 * precision * recall, precision + recall)
        macro_p.append(precision)
        macro_r.append(recall)
        macro_f1.append(f1)
        total_pred += len(pred_set)
        total_gold += len(gold)
        total_hits += hits

        task_rows.append(
            {
                "paper_id": task.paper_id,
                "query_mode": query_mode,
                "top_k_node_count": top_k,
                "gold_team_size": len(gold),
                "pred_team_size": len(pred_set),
                "hits": hits,
                "precision": f"{precision:.8g}",
                "recall": f"{recall:.8g}",
                "f1": f"{f1:.8g}",
                "gold_author_ids": ",".join(task.author_ids),
                "pred_author_ids": ",".join(preds),
            }
        )
        for rank, idx in enumerate(top_indices, start=1):
            author_id = expert_ids[int(idx)]
            pred_rows.append(
                {
                    "paper_id": task.paper_id,
                    "rank": rank,
                    "author_id": author_id,
                    "score": f"{float(scores[int(idx)]):.8g}",
                    "is_gold": int(author_id in gold),
                }
            )

    micro_p = safe_div(total_hits, total_pred)
    micro_r = safe_div(total_hits, total_gold)
    micro_f1 = safe_div(2.0 * micro_p * micro_r, micro_p + micro_r)
    summary: dict[str, object] = {
        "split": split_name,
        "query_mode": query_mode,
        "expert_mode": "all_history_mean_then_expert_proj",
        "papers": len(dataset.tasks),
        "evaluated_papers": len(task_rows),
        "missing_query_papers": missing_query,
        "candidate_authors": len(dataset.author_ids),
        "expert_embeddings_built": len(expert_ids),
        "mean_top_k_node_count": float(np.mean([int(r["top_k_node_count"]) for r in task_rows])) if task_rows else 0.0,
        "macro_precision": float(np.mean(macro_p)) if macro_p else 0.0,
        "macro_recall": float(np.mean(macro_r)) if macro_r else 0.0,
        "macro_f1": float(np.mean(macro_f1)) if macro_f1 else 0.0,
        "micro_precision": micro_p,
        "micro_recall": micro_r,
        "micro_f1": micro_f1,
        "total_pred": total_pred,
        "total_gold": total_gold,
        "total_hits": total_hits,
    }
    write_tsv(split_dir / "per_task_metrics.tsv", task_rows)
    write_tsv(split_dir / "predictions_top_node_count.tsv", pred_rows)
    write_tsv(split_dir / "metrics_summary.tsv", [summary])
    return summary


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    split_paths = {}
    if not args.skip_dev:
        split_paths["dev"] = canonical_dev_paths()
    if not args.skip_official_test:
        split_paths["official_test"] = canonical_official_test_paths()
    if not split_paths:
        raise SystemExit("No split selected")

    datasets: dict[str, Stage1Dataset] = {}
    for split_name, paths in split_paths.items():
        print(f"loading_{split_name}=1", flush=True)
        datasets[split_name] = Stage1Dataset(paths, max_papers=args.max_papers)

    first_dataset = next(iter(datasets.values()))
    model = load_model(Path(args.checkpoint), first_dataset.input_dim, device)

    summaries = []
    for split_name, dataset in datasets.items():
        print(f"evaluating_{split_name}=1", flush=True)
        summary = evaluate_split(split_name, dataset, model, device, out_dir, args.query_mode)
        summaries.append(summary)
        print(json.dumps(summary, ensure_ascii=False), flush=True)

    write_tsv(out_dir / "metrics_summary.tsv", summaries)
    (out_dir / "run_config.json").write_text(
        json.dumps(
            {
                "checkpoint": args.checkpoint,
                "out_dir": args.out_dir,
                "device": args.device,
                "max_papers": args.max_papers,
                "query_mode": args.query_mode,
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
