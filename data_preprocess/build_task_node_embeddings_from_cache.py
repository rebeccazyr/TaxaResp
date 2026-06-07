#!/usr/bin/env python3
"""Build task-node embeddings from LLM requirement embeddings and node embeddings."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from embedding_pipeline_utils import l2, load_embedding_table, read_jsonl, save_embedding_table


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build task-node embeddings from caches")
    p.add_argument("--task-nodes-jsonl", required=True)
    p.add_argument("--requirement-ids", required=True)
    p.add_argument("--requirement-embeddings", required=True)
    p.add_argument("--node-ids", required=True)
    p.add_argument("--node-embeddings", required=True)
    p.add_argument("--ids-out", required=True)
    p.add_argument("--embeddings-out", required=True)
    p.add_argument(
        "--node-weight",
        type=float,
        default=0.25,
        help="Weight for taxonomy node-label embedding; requirement weight is 1.0",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    requirement_embeddings = load_embedding_table(
        Path(args.requirement_ids), Path(args.requirement_embeddings)
    )
    node_embeddings = load_embedding_table(Path(args.node_ids), Path(args.node_embeddings))

    ids = []
    vectors = []
    extra = {}
    missing_requirements = 0
    missing_nodes = 0
    for row in read_jsonl(Path(args.task_nodes_jsonl)):
        paper_id = str(row["paper_id"])
        node_id = str(row["node_id"])
        task_node_id = f"{paper_id}::{node_id}"
        req = requirement_embeddings.get(task_node_id)
        node = node_embeddings.get(node_id)
        if req is None:
            missing_requirements += 1
            continue
        if node is None:
            missing_nodes += 1
            continue
        vec = l2(req + args.node_weight * node)
        ids.append(task_node_id)
        vectors.append(vec)
        extra[task_node_id] = {
            "paper_id": paper_id,
            "node_id": node_id,
            "node_name": row.get("node_name", node_id),
            "node_level": row.get("node_level", ""),
            "team_size": row.get("team_size", ""),
            "node_importance": row.get("node_importance", ""),
            "subtree_skill_count": row.get("subtree_skill_count", ""),
        }

    if not vectors:
        raise SystemExit("No task-node embeddings built")
    arr = np.vstack(vectors).astype(np.float32)
    save_embedding_table(Path(args.ids_out), Path(args.embeddings_out), ids, arr, extra)
    print(f"task_node_embeddings={len(ids)}")
    print(f"missing_requirements={missing_requirements}")
    print(f"missing_nodes={missing_nodes}")
    print(f"dim={arr.shape[1]}")
    print(f"ids_out={args.ids_out}")
    print(f"embeddings_out={args.embeddings_out}")


if __name__ == "__main__":
    main()
