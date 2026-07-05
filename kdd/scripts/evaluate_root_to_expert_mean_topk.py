#!/usr/bin/env python3
"""Evaluate root-role query embeddings against expert mean-history embeddings."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-jsonl", required=True)
    parser.add_argument("--role-descriptions-jsonl", required=True)
    parser.add_argument("--role-ids-tsv", required=True)
    parser.add_argument("--role-embeddings", required=True)
    parser.add_argument("--history-author-papers-tsv", required=True)
    parser.add_argument("--history-ids-tsv", required=True)
    parser.add_argument("--history-embeddings", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--k-grid", default="1,3,5,10,20,50")
    parser.add_argument("--method-label", default="root_mean")
    return parser.parse_args()


def read_ids(path: Path) -> list[str]:
    ids: list[str] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            ids.append(str(row["id"]))
    return ids


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def load_tasks(path: Path) -> dict[str, list[str]]:
    tasks: dict[str, list[str]] = {}
    for row in iter_jsonl(path):
        paper_id = str(row.get("paper_id") or row.get("id") or "").strip()
        authors = [str(x) for x in row.get("team_author_ids") or []]
        if paper_id and authors:
            tasks[paper_id] = authors
    return tasks


def load_root_role_ids(path: Path) -> dict[str, str]:
    root_ids: dict[str, str] = {}
    for row in iter_jsonl(path):
        if str(row.get("node_id") or "") != "__root__":
            continue
        paper_id = str(row.get("paper_id") or "").strip()
        role_id = str(row.get("id") or "").strip()
        if paper_id and role_id:
            root_ids[paper_id] = role_id
    return root_ids


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


def top_indices(scores: np.ndarray, k: int) -> np.ndarray:
    if len(scores) <= k:
        return np.argsort(-scores)
    idx = np.argpartition(-scores, k - 1)[:k]
    return idx[np.argsort(-scores[idx])]


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
    candidate_authors = sorted({author for authors in tasks.values() for author in authors})
    expert_ids, expert_embeddings = build_expert_means(
        Path(args.history_author_papers_tsv),
        Path(args.history_ids_tsv),
        Path(args.history_embeddings),
        candidate_authors,
    )

    root_role_ids = load_root_role_ids(Path(args.role_descriptions_jsonl))
    role_ids = read_ids(Path(args.role_ids_tsv))
    role_id_to_idx = {role_id: idx for idx, role_id in enumerate(role_ids)}
    role_embeddings = np.load(args.role_embeddings, mmap_mode="r")
    if role_embeddings.shape[0] != len(role_ids):
        raise SystemExit("role id and embedding row counts differ")

    per_k_pred: dict[int, int] = {k: 0 for k in k_grid}
    per_k_hits: dict[int, int] = {k: 0 for k in k_grid}
    total_gold = 0
    macro_p: dict[int, list[float]] = {k: [] for k in k_grid}
    macro_r: dict[int, list[float]] = {k: [] for k in k_grid}
    macro_f1: dict[int, list[float]] = {k: [] for k in k_grid}
    per_task_rows: list[dict[str, object]] = []
    missing_root_query = 0

    for paper_id, gold_list in tasks.items():
        role_id = root_role_ids.get(paper_id)
        q_idx = role_id_to_idx.get(role_id or "")
        if q_idx is None:
            missing_root_query += 1
            continue
        query = np.asarray(role_embeddings[q_idx], dtype=np.float32)
        q_norm = float(np.linalg.norm(query))
        if q_norm <= 1e-12:
            missing_root_query += 1
            continue
        query = query / q_norm
        scores = expert_embeddings @ query
        max_k = min(max(k_grid), len(expert_ids))
        ranked = [expert_ids[int(idx)] for idx in top_indices(scores, max_k)]
        gold = set(gold_list)
        total_gold += len(gold)
        for k in k_grid:
            preds = set(ranked[: min(k, len(ranked))])
            hits = len(preds & gold)
            precision = safe_div(hits, len(preds))
            recall = safe_div(hits, len(gold))
            f1 = safe_div(2.0 * precision * recall, precision + recall)
            per_k_pred[k] += len(preds)
            per_k_hits[k] += hits
            macro_p[k].append(precision)
            macro_r[k].append(recall)
            macro_f1[k].append(f1)
            per_task_rows.append(
                {
                    "k": k,
                    "paper_id": paper_id,
                    "pred_size": len(preds),
                    "gold_size": len(gold),
                    "hits": hits,
                    "precision": f"{precision:.8g}",
                    "recall": f"{recall:.8g}",
                    "f1": f"{f1:.8g}",
                }
            )

    summary_rows: list[dict[str, object]] = []
    for k in k_grid:
        micro_p = safe_div(per_k_hits[k], per_k_pred[k])
        micro_r = safe_div(per_k_hits[k], total_gold)
        micro_f1 = safe_div(2.0 * micro_p * micro_r, micro_p + micro_r)
        row: dict[str, object] = {
            "method_label": args.method_label,
            "split": "official_test",
            "k": k,
            "papers": len(tasks),
            "evaluated_papers": len(macro_r[k]),
            "missing_root_query_papers": missing_root_query,
            "candidate_authors": len(candidate_authors),
            "expert_embeddings_built": len(expert_ids),
            "micro_precision": micro_p,
            "micro_recall": micro_r,
            "micro_f1": micro_f1,
            "macro_precision": float(np.mean(macro_p[k])) if macro_p[k] else 0.0,
            "macro_recall": float(np.mean(macro_r[k])) if macro_r[k] else 0.0,
            "macro_f1": float(np.mean(macro_f1[k])) if macro_f1[k] else 0.0,
            "avg_pred_size": safe_div(per_k_pred[k], len(macro_r[k])),
            "total_pred": per_k_pred[k],
            "total_gold": total_gold,
            "total_hits": per_k_hits[k],
        }
        summary_rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)

    write_tsv(out_dir / "root_mean_topk_summary.tsv", summary_rows)
    write_tsv(out_dir / "root_mean_topk_per_task.tsv", per_task_rows)
    (out_dir / "run_config.json").write_text(
        json.dumps(
            {
                "sample_jsonl": args.sample_jsonl,
                "role_descriptions_jsonl": args.role_descriptions_jsonl,
                "role_ids_tsv": args.role_ids_tsv,
                "role_embeddings": args.role_embeddings,
                "history_author_papers_tsv": args.history_author_papers_tsv,
                "history_ids_tsv": args.history_ids_tsv,
                "history_embeddings": args.history_embeddings,
                "out_dir": args.out_dir,
                "k_grid": k_grid,
                "method_label": args.method_label,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
