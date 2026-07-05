#!/usr/bin/env python3
"""Search whole-paper embeddings against whole-expert mean history embeddings."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sample-jsonl", required=True)
    p.add_argument("--role-descriptions-jsonl", required=True)
    p.add_argument("--paper-ids-tsv", required=True)
    p.add_argument("--paper-embeddings", required=True)
    p.add_argument("--history-author-papers-tsv", required=True)
    p.add_argument("--history-ids-tsv", required=True)
    p.add_argument("--history-embeddings", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument(
        "--k-grid",
        default="",
        help="Optional comma-separated K values. If set, evaluate whole-paper top-K for each K instead of top_k=node_count.",
    )
    p.add_argument("--method-label", default="paper_mean")
    return p.parse_args()


def read_ids(path: Path) -> list[str]:
    ids: list[str] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            ids.append(str(row["id"]))
    return ids


def load_tasks(path: Path) -> dict[str, list[str]]:
    tasks: dict[str, list[str]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            paper_id = str(row.get("paper_id") or row.get("id") or "")
            authors = [str(x) for x in row.get("team_author_ids") or []]
            if paper_id and authors:
                tasks[paper_id] = authors
    return tasks


def load_node_counts(path: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            paper_id = str(row.get("paper_id") or "")
            if paper_id:
                counts[paper_id] = counts.get(paper_id, 0) + 1
    return counts


def build_expert_means(
    author_papers_tsv: Path,
    history_ids_tsv: Path,
    history_embeddings_npy: Path,
    candidate_authors: Sequence[str],
) -> tuple[list[str], np.ndarray]:
    history_ids = read_ids(history_ids_tsv)
    history_id_to_idx = {paper_id: idx for idx, paper_id in enumerate(history_ids)}
    history_embeddings = np.load(history_embeddings_npy, mmap_mode="r")
    wanted = set(candidate_authors)
    author_to_indices: dict[str, list[int]] = {}
    with author_papers_tsv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            author_id = str(row.get("author_id") or "")
            if author_id not in wanted:
                continue
            idx = history_id_to_idx.get(str(row.get("history_paper_id") or ""))
            if idx is not None:
                author_to_indices.setdefault(author_id, []).append(idx)

    expert_ids: list[str] = []
    expert_vectors: list[np.ndarray] = []
    for author_id in candidate_authors:
        indices = sorted(set(author_to_indices.get(author_id, [])))
        if not indices:
            continue
        vec = np.asarray(history_embeddings[indices], dtype=np.float32).mean(axis=0)
        norm = float(np.linalg.norm(vec))
        if norm <= 1e-12:
            continue
        expert_ids.append(author_id)
        expert_vectors.append(vec / norm)
    if not expert_vectors:
        raise SystemExit("No expert mean embeddings built")
    return expert_ids, np.vstack(expert_vectors).astype(np.float32)


def safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def write_tsv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    k_grid = [int(x) for x in str(args.k_grid).split(",") if x.strip()]

    tasks = load_tasks(Path(args.sample_jsonl))
    node_counts = load_node_counts(Path(args.role_descriptions_jsonl))
    candidate_authors = sorted({author for authors in tasks.values() for author in authors})
    expert_ids, expert_embeddings = build_expert_means(
        Path(args.history_author_papers_tsv),
        Path(args.history_ids_tsv),
        Path(args.history_embeddings),
        candidate_authors,
    )

    paper_ids = read_ids(Path(args.paper_ids_tsv))
    paper_id_to_idx = {paper_id: idx for idx, paper_id in enumerate(paper_ids)}
    paper_embeddings = np.load(args.paper_embeddings, mmap_mode="r")

    pred_rows: list[dict[str, object]] = []
    task_rows: list[dict[str, object]] = []
    grid_task_rows: list[dict[str, object]] = []
    macro_p: list[float] = []
    macro_r: list[float] = []
    macro_f1: list[float] = []
    total_hits = 0
    total_pred = 0
    total_gold = 0
    missing_query = 0
    grid_total_pred: dict[int, int] = {k: 0 for k in k_grid}
    grid_total_hits: dict[int, int] = {k: 0 for k in k_grid}
    grid_total_gold = 0
    grid_macro_p: dict[int, list[float]] = {k: [] for k in k_grid}
    grid_macro_r: dict[int, list[float]] = {k: [] for k in k_grid}
    grid_macro_f1: dict[int, list[float]] = {k: [] for k in k_grid}

    for paper_id, gold_list in tasks.items():
        q_idx = paper_id_to_idx.get(paper_id)
        if q_idx is None:
            missing_query += 1
            continue
        q = np.asarray(paper_embeddings[q_idx], dtype=np.float32)
        q_norm = float(np.linalg.norm(q))
        if q_norm <= 1e-12:
            missing_query += 1
            continue
        q = q / q_norm
        top_k = max(1, int(node_counts.get(paper_id, 0)))
        scores = expert_embeddings @ q
        max_needed_k = max(k_grid) if k_grid else top_k
        k = min(max_needed_k, len(expert_ids))
        top_indices = np.argpartition(-scores, kth=k - 1)[:k]
        top_indices = top_indices[np.argsort(-scores[top_indices])]
        ranked_authors = [expert_ids[int(idx)] for idx in top_indices]
        gold = set(gold_list)

        if k_grid:
            grid_total_gold += len(gold)
            for grid_k in k_grid:
                preds = set(ranked_authors[: min(grid_k, len(ranked_authors))])
                hits = len(preds & gold)
                precision = safe_div(hits, len(preds))
                recall = safe_div(hits, len(gold))
                f1 = safe_div(2.0 * precision * recall, precision + recall)
                grid_total_pred[grid_k] += len(preds)
                grid_total_hits[grid_k] += hits
                grid_macro_p[grid_k].append(precision)
                grid_macro_r[grid_k].append(recall)
                grid_macro_f1[grid_k].append(f1)
                grid_task_rows.append(
                    {
                        "k": grid_k,
                        "paper_id": paper_id,
                        "gold_team_size": len(gold),
                        "pred_team_size": len(preds),
                        "hits": hits,
                        "precision": f"{precision:.8g}",
                        "recall": f"{recall:.8g}",
                        "f1": f"{f1:.8g}",
                    }
                )
            continue

        preds = ranked_authors
        pred_set = set(preds)
        hits = len(pred_set & gold)
        precision = safe_div(hits, len(pred_set))
        recall = safe_div(hits, len(gold))
        f1 = safe_div(2.0 * precision * recall, precision + recall)
        macro_p.append(precision)
        macro_r.append(recall)
        macro_f1.append(f1)
        total_hits += hits
        total_pred += len(pred_set)
        total_gold += len(gold)
        task_rows.append(
            {
                "paper_id": paper_id,
                "top_k_node_count": top_k,
                "gold_team_size": len(gold),
                "pred_team_size": len(pred_set),
                "hits": hits,
                "precision": f"{precision:.8g}",
                "recall": f"{recall:.8g}",
                "f1": f"{f1:.8g}",
                "gold_author_ids": ",".join(gold_list),
                "pred_author_ids": ",".join(preds),
            }
        )
        for rank, idx in enumerate(top_indices, start=1):
            author_id = expert_ids[int(idx)]
            pred_rows.append(
                {
                    "paper_id": paper_id,
                    "rank": rank,
                    "author_id": author_id,
                    "score": f"{float(scores[int(idx)]):.8g}",
                    "is_gold": int(author_id in gold),
                }
            )

    micro_p = safe_div(total_hits, total_pred)
    micro_r = safe_div(total_hits, total_gold)
    micro_f1 = safe_div(2.0 * micro_p * micro_r, micro_p + micro_r)
    if k_grid:
        summary_rows: list[dict[str, object]] = []
        for grid_k in k_grid:
            micro_p = safe_div(grid_total_hits[grid_k], grid_total_pred[grid_k])
            micro_r = safe_div(grid_total_hits[grid_k], grid_total_gold)
            micro_f1 = safe_div(2.0 * micro_p * micro_r, micro_p + micro_r)
            row: dict[str, object] = {
                "method_label": args.method_label,
                "split": "official_test",
                "k": grid_k,
                "papers": len(tasks),
                "evaluated_papers": len(grid_macro_r[grid_k]),
                "missing_query_papers": missing_query,
                "candidate_authors": len(candidate_authors),
                "expert_embeddings_built": len(expert_ids),
                "micro_precision": micro_p,
                "micro_recall": micro_r,
                "micro_f1": micro_f1,
                "macro_precision": float(np.mean(grid_macro_p[grid_k])) if grid_macro_p[grid_k] else 0.0,
                "macro_recall": float(np.mean(grid_macro_r[grid_k])) if grid_macro_r[grid_k] else 0.0,
                "macro_f1": float(np.mean(grid_macro_f1[grid_k])) if grid_macro_f1[grid_k] else 0.0,
                "avg_pred_size": safe_div(grid_total_pred[grid_k], len(grid_macro_r[grid_k])),
                "total_pred": grid_total_pred[grid_k],
                "total_gold": grid_total_gold,
                "total_hits": grid_total_hits[grid_k],
            }
            summary_rows.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
        write_tsv(out_dir / "paper_mean_topk_summary.tsv", summary_rows)
        write_tsv(out_dir / "paper_mean_topk_per_task.tsv", grid_task_rows)
        return

    summary = {
        "papers": len(tasks),
        "evaluated_papers": len(task_rows),
        "missing_query_papers": missing_query,
        "candidate_authors": len(candidate_authors),
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
    write_tsv(out_dir / "per_task_metrics.tsv", task_rows)
    write_tsv(out_dir / "predictions_top_node_count.tsv", pred_rows)
    write_tsv(out_dir / "metrics_summary.tsv", [summary])
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
