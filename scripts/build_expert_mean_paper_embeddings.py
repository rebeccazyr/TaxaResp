#!/usr/bin/env python3
"""Build global expert representations from mean historical-paper embeddings."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--expert-papers-tsv",
        default="output/all_expert_paper_embeddings/expert_papers.tsv",
    )
    p.add_argument(
        "--paper-ids-tsv",
        default="output/all_expert_paper_embeddings/paper_embedding_ids.tsv",
    )
    p.add_argument(
        "--paper-embeddings",
        default="output/all_expert_paper_embeddings/paper_embeddings.npy",
    )
    p.add_argument(
        "--ids-out",
        default="output/virtual_root_role_descriptions/expert_mean_paper_embedding_ids.tsv",
    )
    p.add_argument(
        "--embeddings-out",
        default="output/virtual_root_role_descriptions/expert_mean_paper_embeddings.npy",
    )
    p.add_argument("--normalize", action="store_true", default=True)
    return p.parse_args()


def read_paper_rows(path: Path) -> Dict[str, int]:
    out: Dict[str, int] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for idx, row in enumerate(reader):
            out[str(row["id"])] = idx
    return out


def read_expert_papers(path: Path) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = defaultdict(list)
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            expert_id = str(row["expert_id"])
            paper_id = str(row["paper_id"])
            if expert_id and paper_id:
                out[expert_id].append(paper_id)
    return dict(out)


def main() -> None:
    args = parse_args()
    ids_out = Path(args.ids_out)
    emb_out = Path(args.embeddings_out)
    ids_out.parent.mkdir(parents=True, exist_ok=True)
    emb_out.parent.mkdir(parents=True, exist_ok=True)

    paper_to_row = read_paper_rows(Path(args.paper_ids_tsv))
    expert_papers = read_expert_papers(Path(args.expert_papers_tsv))
    paper_arr = np.load(args.paper_embeddings, mmap_mode="r")
    dim = int(paper_arr.shape[1])
    expert_ids = sorted(expert_papers)
    expert_arr = np.zeros((len(expert_ids), dim), dtype=np.float32)
    meta_rows = []

    for idx, expert_id in enumerate(expert_ids):
        rows = [paper_to_row[p] for p in expert_papers[expert_id] if p in paper_to_row]
        if rows:
            vec = np.asarray(paper_arr[rows], dtype=np.float32).mean(axis=0)
            if args.normalize:
                norm = float(np.linalg.norm(vec))
                if norm > 0:
                    vec = vec / norm
            expert_arr[idx] = vec
        meta_rows.append((expert_id, len(expert_papers[expert_id]), len(rows)))
        if (idx + 1) % 1000 == 0:
            print(f"expert_mean_progress={idx + 1:,}/{len(expert_ids):,}", flush=True)

    np.save(emb_out, expert_arr)
    with ids_out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["id", "paper_count", "matched_paper_count"])
        writer.writerows(meta_rows)
    print(f"wrote_ids={ids_out} rows={len(expert_ids)}", flush=True)
    print(f"wrote_embeddings={emb_out} shape={expert_arr.shape}", flush=True)


if __name__ == "__main__":
    main()
