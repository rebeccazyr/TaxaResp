#!/usr/bin/env python3
"""Build HieRec-style expert-node embeddings from cached paper/node embeddings."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from embedding_pipeline_utils import (
    ancestor_cache_builder,
    l2,
    load_child_to_parents,
    load_embedding_table,
    load_fos_map,
    read_jsonl,
    softmax,
    save_embedding_table,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build expert-node embeddings from caches")
    p.add_argument("--expert-node-evidence-jsonl", required=True)
    p.add_argument("--paper-ids", required=True)
    p.add_argument("--paper-embeddings", required=True)
    p.add_argument("--node-ids", required=True)
    p.add_argument("--node-embeddings", required=True)
    p.add_argument("--fos-map", default="data/dblp/FieldsOfStudy.txt")
    p.add_argument("--fos-children", default="data/dblp/13.FieldOfStudyChildren.nt")
    p.add_argument("--ancestor-depth", type=int, default=5)
    p.add_argument("--ids-out", required=True)
    p.add_argument("--embeddings-out", required=True)
    return p.parse_args()


def build_for_expert(
    rows: List[dict],
    ancestors,
    child_to_parents: Dict[str, List[str]],
    node_embeddings: Dict[str, np.ndarray],
    paper_embeddings: Dict[str, np.ndarray],
) -> Dict[str, dict]:
    direct_weight = {row["node_id"]: float(row["direct_weight_sum"]) for row in rows}
    direct_papers = {row["node_id"]: row.get("papers") or [] for row in rows}
    nodes = set()
    for node_id in direct_weight:
        for ancestor_id, _ in ancestors(node_id):
            if ancestor_id in node_embeddings:
                nodes.add(ancestor_id)

    parent_to_children: Dict[str, set] = defaultdict(set)
    for child in nodes:
        for parent in child_to_parents.get(child, []):
            if parent in nodes:
                parent_to_children[parent].add(child)

    subtree_weight: Dict[str, float] = defaultdict(float)
    direct_leaf_count: Dict[str, int] = defaultdict(int)
    for leaf, weight in direct_weight.items():
        for node, _ in ancestors(leaf):
            if node in nodes:
                subtree_weight[node] += weight
                direct_leaf_count[node] += 1

    remaining = set(nodes)
    ordered_nodes = []
    while remaining:
        ready = sorted(
            [
                node
                for node in remaining
                if all(child not in remaining for child in parent_to_children.get(node, set()))
            ],
            key=lambda n: (len(parent_to_children.get(n, set())), n),
        )
        if not ready:
            ready = [sorted(remaining)[0]]
        ordered_nodes.extend(ready)
        remaining.difference_update(ready)

    reps: Dict[str, np.ndarray] = {}
    meta: Dict[str, dict] = {}
    zero = np.zeros_like(next(iter(node_embeddings.values())))
    for node in ordered_nodes:
        paper_items = [
            (str(p.get("paper_id")), float(p.get("weight") or 0.0))
            for p in direct_papers.get(node, [])
            if str(p.get("paper_id")) in paper_embeddings
        ]
        paper_attn = softmax(paper_items)
        paper_agg = (
            sum(paper_attn[pid] * paper_embeddings[pid] for pid, _ in paper_items)
            if paper_items
            else zero
        )

        children = [c for c in parent_to_children.get(node, set()) if c in reps]
        child_items = [(c, subtree_weight[c]) for c in children]
        child_attn = softmax(child_items)
        child_agg = (
            sum(child_attn[c] * reps[c] for c in children)
            if children
            else zero
        )

        rep = l2(node_embeddings[node] + paper_agg + child_agg)
        reps[node] = rep
        meta[node] = {
            "embedding": rep,
            "is_direct_node": int(node in direct_weight),
            "subtree_weight_sum": subtree_weight[node],
            "direct_leaf_count": direct_leaf_count[node],
            "child_count": len(children),
            "evidence_paper_count": len(paper_items),
            "top_evidence_papers": json.dumps(
                [
                    {"paper_id": pid, "attention": round(paper_attn[pid], 6)}
                    for pid, _ in sorted(paper_items, key=lambda x: paper_attn[x[0]], reverse=True)[:5]
                ],
                ensure_ascii=False,
            ),
        }
    return meta


def main() -> None:
    args = parse_args()
    _, id_to_name, id_to_level = load_fos_map(Path(args.fos_map))
    child_to_parents = load_child_to_parents(Path(args.fos_children))
    ancestors = ancestor_cache_builder(child_to_parents, args.ancestor_depth)
    paper_embeddings = load_embedding_table(Path(args.paper_ids), Path(args.paper_embeddings))
    node_embeddings = load_embedding_table(Path(args.node_ids), Path(args.node_embeddings))

    by_expert: Dict[str, List[dict]] = defaultdict(list)
    expert_names = {}
    for row in read_jsonl(Path(args.expert_node_evidence_jsonl)):
        expert_id = str(row["expert_id"])
        by_expert[expert_id].append(row)
        expert_names[expert_id] = row.get("expert_name", expert_id)

    ids = []
    vectors = []
    extra = {}
    for idx, (expert_id, rows) in enumerate(sorted(by_expert.items()), start=1):
        if idx % 500 == 0:
            print(f"expert_progress {idx:,}/{len(by_expert):,}")
        records = build_for_expert(rows, ancestors, child_to_parents, node_embeddings, paper_embeddings)
        for node_id, rec in records.items():
            emb_id = f"{expert_id}::{node_id}"
            ids.append(emb_id)
            vectors.append(rec["embedding"])
            extra[emb_id] = {
                "expert_id": expert_id,
                "expert_name": expert_names.get(expert_id, expert_id),
                "node_id": node_id,
                "node_name": id_to_name.get(node_id, node_id),
                "node_level": id_to_level.get(node_id, ""),
                "is_direct_node": rec["is_direct_node"],
                "subtree_weight_sum": f"{rec['subtree_weight_sum']:.6f}",
                "direct_leaf_count": rec["direct_leaf_count"],
                "child_count": rec["child_count"],
                "evidence_paper_count": rec["evidence_paper_count"],
                "top_evidence_papers": rec["top_evidence_papers"],
            }

    arr = np.vstack(vectors).astype(np.float32)
    save_embedding_table(Path(args.ids_out), Path(args.embeddings_out), ids, arr, extra)
    print(f"experts={len(by_expert)}")
    print(f"expert_node_embeddings={len(ids)}")
    print(f"dim={arr.shape[1]}")
    print(f"ids_out={args.ids_out}")
    print(f"embeddings_out={args.embeddings_out}")


if __name__ == "__main__":
    main()
