#!/usr/bin/env python3
"""Retrieve global experts for virtual-root role embeddings."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--root-role-ids",
        default="output/virtual_root_role_descriptions/root_role_embedding_ids.tsv",
    )
    p.add_argument(
        "--root-role-embeddings",
        default="output/virtual_root_role_descriptions/root_role_embeddings.npy",
    )
    p.add_argument(
        "--expert-ids",
        default="output/virtual_root_role_descriptions/expert_mean_paper_embedding_ids.tsv",
    )
    p.add_argument(
        "--expert-embeddings",
        default="output/virtual_root_role_descriptions/expert_mean_paper_embeddings.npy",
    )
    p.add_argument(
        "--out-tsv",
        default="output/virtual_root_role_descriptions/virtual_root_expert_matches.tsv",
    )
    p.add_argument(
        "--top-k",
        type=int,
        default=20,
    )
    p.add_argument(
        "--predictions-top1-tsv",
        default="output/virtual_root_role_descriptions/virtual_root_top1_predictions.tsv",
    )
    return p.parse_args()


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


def main() -> None:
    args = parse_args()
    root_ids = read_ids(Path(args.root_role_ids))
    expert_ids = read_ids(Path(args.expert_ids))
    root_arr = normalize_rows(np.load(args.root_role_embeddings, mmap_mode="r"))
    expert_arr = normalize_rows(np.load(args.expert_embeddings, mmap_mode="r"))
    if root_arr.shape[0] != len(root_ids):
        raise SystemExit("root id count does not match root embedding rows")
    if expert_arr.shape[0] != len(expert_ids):
        raise SystemExit("expert id count does not match expert embedding rows")
    if root_arr.shape[1] != expert_arr.shape[1]:
        raise SystemExit(f"embedding dim mismatch: root={root_arr.shape[1]} expert={expert_arr.shape[1]}")

    top_k = max(1, min(args.top_k, len(expert_ids)))
    out_path = Path(args.out_tsv)
    pred_path = Path(args.predictions_top1_tsv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pred_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8", newline="") as out_f, pred_path.open(
        "w", encoding="utf-8", newline=""
    ) as pred_f:
        writer = csv.writer(out_f, delimiter="\t")
        writer.writerow(["paper_id", "rank", "expert_id", "cosine_similarity", "cosine_distance"])
        pred_writer = csv.writer(pred_f, delimiter="\t")
        pred_writer.writerow(["method", "paper_id", "rank", "expert_id", "score"])

        for start in range(0, len(root_ids), 64):
            end = min(start + 64, len(root_ids))
            scores = root_arr[start:end] @ expert_arr.T
            for local_idx, paper_id in enumerate(root_ids[start:end]):
                row_scores = scores[local_idx]
                best = np.argpartition(-row_scores, top_k - 1)[:top_k]
                best = best[np.argsort(-row_scores[best])]
                for rank, expert_idx in enumerate(best, start=1):
                    score = float(row_scores[expert_idx])
                    expert_id = expert_ids[int(expert_idx)]
                    writer.writerow([paper_id, rank, expert_id, f"{score:.9f}", f"{1.0 - score:.9f}"])
                    if rank == 1:
                        pred_writer.writerow(
                            ["virtual_root_mean_paper_top1", paper_id, 1, expert_id, f"{score:.9f}"]
                        )
            print(f"matched_root_roles={end}/{len(root_ids)}", flush=True)

    print(f"wrote_matches={out_path}", flush=True)
    print(f"wrote_top1_predictions={pred_path}", flush=True)


if __name__ == "__main__":
    main()
