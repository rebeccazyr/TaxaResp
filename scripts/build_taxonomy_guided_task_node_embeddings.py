#!/usr/bin/env python3
"""Build task-node embeddings by projecting task abstracts onto taxonomy nodes.

For each task paper and taxonomy node, this script attends from the taxonomy
node embedding to sentence-like abstract chunks from the same paper:

    z_{t,v} = normalize(sum_i softmax(cos(h_v, e_i) / tau) e_i)

The output id/embedding table is compatible with the existing hard-match
evaluator in data_preprocess/evaluate_hierec_embedding_team_size.py.
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

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
UTILS_DIR = ROOT / "data_preprocess"
import sys

if str(UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(UTILS_DIR))

from embedding_pipeline_utils import l2, read_jsonl, save_embedding_table  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task-nodes-jsonl", required=True)
    p.add_argument("--node-ids", required=True)
    p.add_argument("--node-embeddings", required=True)
    p.add_argument("--chunk-jsonl", required=True)
    p.add_argument("--chunk-ids", required=True)
    p.add_argument("--chunk-embeddings", required=True)
    p.add_argument("--ids-out", required=True)
    p.add_argument("--embeddings-out", required=True)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--max-chars", type=int, default=360)
    p.add_argument("--min-chars", type=int, default=40)
    p.add_argument(
        "--write-chunks-only",
        action="store_true",
        help="Only write abstract chunks for an external embedding step.",
    )
    return p.parse_args()


def read_id_rows(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def load_embedding_table_mmap(ids_path: Path, npy_path: Path) -> Tuple[List[str], np.ndarray]:
    ids = [row["id"] for row in read_id_rows(ids_path)]
    arr = np.load(npy_path, mmap_mode="r")
    if len(ids) != arr.shape[0]:
        raise ValueError(f"ids/embedding row mismatch for {ids_path}: {len(ids)} vs {arr.shape[0]}")
    return ids, arr


def sentence_split(text: str) -> List[str]:
    text = " ".join(str(text or "").split())
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text)
    return [part.strip() for part in parts if part.strip()]


def chunk_text(text: str, max_chars: int, min_chars: int) -> List[str]:
    sentences = sentence_split(text)
    chunks: List[str] = []
    current: List[str] = []
    current_len = 0
    for sent in sentences:
        sent_len = len(sent)
        if current and current_len + 1 + sent_len > max_chars:
            chunks.append(" ".join(current).strip())
            current = [sent]
            current_len = sent_len
        else:
            current.append(sent)
            current_len += sent_len + (1 if current_len else 0)
    if current:
        chunks.append(" ".join(current).strip())

    merged: List[str] = []
    for chunk in chunks:
        if merged and len(chunk) < min_chars:
            merged[-1] = f"{merged[-1]} {chunk}".strip()
        else:
            merged.append(chunk)
    return merged or ([text] if text else [])


def unique_task_papers(task_nodes_jsonl: Path) -> Dict[str, dict]:
    papers: Dict[str, dict] = {}
    for row in read_jsonl(task_nodes_jsonl):
        paper_id = str(row["paper_id"])
        if paper_id not in papers:
            papers[paper_id] = {
                "paper_id": paper_id,
                "text": str(row.get("task_paper_text") or ""),
                "members": row.get("members", []),
                "team_size": row.get("team_size", ""),
            }
    return papers


def write_chunks(path: Path, papers: Dict[str, dict], max_chars: int, min_chars: int) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with path.open("w", encoding="utf-8") as f:
        for paper_id, paper in sorted(papers.items()):
            chunks = chunk_text(paper["text"], max_chars=max_chars, min_chars=min_chars)
            for idx, text in enumerate(chunks):
                obj = {
                    "id": f"{paper_id}::chunk{idx:03d}",
                    "paper_id": paper_id,
                    "chunk_index": idx,
                    "text": text,
                }
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")
                written += 1
    return written


def softmax(values: np.ndarray, temperature: float) -> np.ndarray:
    if values.size == 0:
        return values
    scale = max(float(temperature), 1e-12)
    x = values.astype(np.float64) / scale
    x -= float(np.max(x))
    weights = np.exp(x)
    denom = float(np.sum(weights))
    return (weights / denom).astype(np.float32) if denom > 0 else np.full_like(values, 1.0 / len(values))


def parse_chunk_id(chunk_id: str) -> Tuple[str, int]:
    paper_id, _, chunk_part = chunk_id.partition("::chunk")
    try:
        idx = int(chunk_part)
    except ValueError:
        idx = 0
    return paper_id, idx


def build_chunk_index(chunk_ids: Sequence[str]) -> Dict[str, List[int]]:
    by_paper: Dict[str, List[int]] = defaultdict(list)
    for row_idx, chunk_id in enumerate(chunk_ids):
        paper_id, _ = parse_chunk_id(chunk_id)
        by_paper[paper_id].append(row_idx)
    for paper_id, indexes in by_paper.items():
        indexes.sort(key=lambda i: parse_chunk_id(chunk_ids[i])[1])
    return dict(by_paper)


def main() -> None:
    args = parse_args()
    task_nodes_path = Path(args.task_nodes_jsonl)
    papers = unique_task_papers(task_nodes_path)
    chunk_count = write_chunks(Path(args.chunk_jsonl), papers, args.max_chars, args.min_chars)
    print(f"task_papers={len(papers)}")
    print(f"abstract_chunks={chunk_count}")
    print(f"chunk_jsonl={args.chunk_jsonl}")
    if args.write_chunks_only:
        return

    node_ids, node_arr = load_embedding_table_mmap(Path(args.node_ids), Path(args.node_embeddings))
    chunk_ids, chunk_arr = load_embedding_table_mmap(Path(args.chunk_ids), Path(args.chunk_embeddings))
    if node_arr.shape[1] != chunk_arr.shape[1]:
        raise SystemExit(f"embedding dim mismatch: node={node_arr.shape[1]} chunk={chunk_arr.shape[1]}")

    node_index = {node_id: idx for idx, node_id in enumerate(node_ids)}
    chunks_by_paper = build_chunk_index(chunk_ids)

    ids: List[str] = []
    vectors: List[np.ndarray] = []
    extra: Dict[str, dict] = {}
    missing_nodes = 0
    missing_chunks = 0
    for row in read_jsonl(task_nodes_path):
        paper_id = str(row["paper_id"])
        node_id = str(row["node_id"])
        node_idx = node_index.get(node_id)
        chunk_indexes = chunks_by_paper.get(paper_id, [])
        if node_idx is None:
            missing_nodes += 1
            continue
        if not chunk_indexes:
            missing_chunks += 1
            continue

        node_vec = np.asarray(node_arr[node_idx], dtype=np.float32)
        node_vec = l2(node_vec)
        chunk_mat = np.asarray(chunk_arr[chunk_indexes], dtype=np.float32)
        chunk_norms = np.linalg.norm(chunk_mat, axis=1, keepdims=True)
        chunk_mat = chunk_mat / np.maximum(chunk_norms, 1e-12)
        scores = chunk_mat @ node_vec
        weights = softmax(scores, args.temperature)
        vec = l2(weights @ chunk_mat)

        task_node_id = f"{paper_id}::{node_id}"
        ids.append(task_node_id)
        vectors.append(vec.astype(np.float32))
        top_pos = int(np.argmax(weights))
        extra[task_node_id] = {
            "paper_id": paper_id,
            "node_id": node_id,
            "node_name": row.get("node_name", node_id),
            "node_level": row.get("node_level", ""),
            "team_size": row.get("team_size", ""),
            "node_importance": row.get("node_importance", ""),
            "subtree_skill_count": row.get("subtree_skill_count", ""),
            "abstract_chunk_count": len(chunk_indexes),
            "top_chunk_id": chunk_ids[chunk_indexes[top_pos]],
            "top_chunk_attention": f"{float(weights[top_pos]):.9f}",
            "top_chunk_cosine": f"{float(scores[top_pos]):.9f}",
        }

    if not vectors:
        raise SystemExit("No task-node projection embeddings built")
    arr = np.vstack(vectors).astype(np.float32)
    save_embedding_table(Path(args.ids_out), Path(args.embeddings_out), ids, arr, extra)
    print(f"task_node_embeddings={len(ids)}")
    print(f"missing_nodes={missing_nodes}")
    print(f"missing_chunks={missing_chunks}")
    print(f"dim={arr.shape[1]}")
    print(f"temperature={args.temperature:g}")
    print(f"ids_out={args.ids_out}")
    print(f"embeddings_out={args.embeddings_out}")


if __name__ == "__main__":
    main()
