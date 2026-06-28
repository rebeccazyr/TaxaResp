#!/usr/bin/env python3
"""Build full expert-node embeddings from cached paper/node embeddings.

This is the streaming version of build_expert_node_embeddings_from_cache.py.
It avoids keeping all expert-node vectors in Python memory at once.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

from embedding_pipeline_utils import (
    ancestor_cache_builder,
    l2,
    load_child_to_parents,
    load_embedding_table,
    load_fos_map,
    read_jsonl,
    softmax,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build full expert-node embeddings with memmap output")
    p.add_argument("--expert-node-evidence-jsonl", required=True)
    p.add_argument("--paper-ids", required=True)
    p.add_argument("--paper-embeddings", required=True)
    p.add_argument("--node-ids", required=True)
    p.add_argument("--node-embeddings", required=True)
    p.add_argument("--fos-map", default="data/dblp/FieldsOfStudy.txt")
    p.add_argument("--fos-children", default="data/dblp/13.FieldOfStudyChildren.nt")
    p.add_argument("--ancestor-depth", type=int, default=5)
    p.add_argument(
        "--node-weight",
        type=float,
        default=1.0,
        help="Weight for taxonomy node-label embedding. Use 0 for evidence-only scheme A.",
    )
    p.add_argument("--ids-out", required=True)
    p.add_argument("--embeddings-out", required=True)
    return p.parse_args()


def load_id_to_row(path: Path) -> Dict[str, int]:
    out = {}
    with path.open("r", encoding="utf-8") as f:
        for idx, row in enumerate(csv.DictReader(f, delimiter="\t")):
            out[str(row["id"])] = idx
    return out


def iter_expert_groups(path: Path) -> Iterable[Tuple[str, List[dict]]]:
    current_expert = None
    rows: List[dict] = []
    for row in read_jsonl(path):
        expert_id = str(row["expert_id"])
        if current_expert is None:
            current_expert = expert_id
        if expert_id != current_expert:
            yield current_expert, rows
            current_expert = expert_id
            rows = []
        rows.append(row)
    if current_expert is not None:
        yield current_expert, rows


def expert_nodes(rows: List[dict], ancestors, node_ids: set) -> set:
    nodes = set()
    for row in rows:
        for ancestor_id, _ in ancestors(str(row["node_id"])):
            if ancestor_id in node_ids:
                nodes.add(ancestor_id)
    return nodes


def count_total_embeddings(path: Path, ancestors, node_ids: set) -> Tuple[int, int]:
    total = 0
    experts = 0
    for experts, (_, rows) in enumerate(iter_expert_groups(path), start=1):
        total += len(expert_nodes(rows, ancestors, node_ids))
        if experts % 1000 == 0:
            print(f"count_progress experts={experts:,} total_nodes={total:,}", flush=True)
    return experts, total


def build_for_expert(
    rows: List[dict],
    ancestors,
    child_to_parents: Dict[str, List[str]],
    node_embeddings: Dict[str, np.ndarray],
    paper_id_to_row: Dict[str, int],
    paper_arr: np.ndarray,
    node_weight: float,
) -> Dict[str, dict]:
    direct_weight = {str(row["node_id"]): float(row["direct_weight_sum"]) for row in rows}
    direct_papers = {str(row["node_id"]): row.get("papers") or [] for row in rows}
    nodes = expert_nodes(rows, ancestors, set(node_embeddings))

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
        paper_items = []
        for p in direct_papers.get(node, []):
            paper_id = str(p.get("paper_id"))
            if paper_id in paper_id_to_row:
                paper_items.append((paper_id, float(p.get("weight") or 0.0)))
        paper_attn = softmax(paper_items)
        paper_agg = (
            sum(paper_attn[pid] * paper_arr[paper_id_to_row[pid]] for pid, _ in paper_items)
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

        rep = l2(args_node_component(node_embeddings[node], node_weight) + paper_agg + child_agg).astype(np.float32)
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


def args_node_component(node_embedding: np.ndarray, node_weight: float) -> np.ndarray:
    if node_weight == 0:
        return np.zeros_like(node_embedding)
    return node_weight * node_embedding


def main() -> None:
    args = parse_args()
    evidence_path = Path(args.expert_node_evidence_jsonl)
    _, id_to_name, id_to_level = load_fos_map(Path(args.fos_map))
    child_to_parents = load_child_to_parents(Path(args.fos_children))
    ancestors = ancestor_cache_builder(child_to_parents, args.ancestor_depth)

    print("loading node embeddings", flush=True)
    node_embeddings = load_embedding_table(Path(args.node_ids), Path(args.node_embeddings))
    print("loading paper id index", flush=True)
    paper_id_to_row = load_id_to_row(Path(args.paper_ids))
    print("opening paper embedding memmap", flush=True)
    paper_arr = np.load(args.paper_embeddings, mmap_mode="r")

    print("counting expert-node embeddings", flush=True)
    expert_count, total = count_total_embeddings(evidence_path, ancestors, set(node_embeddings))
    if total <= 0:
        raise SystemExit("No expert-node embeddings to write")
    dim = int(next(iter(node_embeddings.values())).shape[0])
    print(f"experts={expert_count:,}", flush=True)
    print(f"expert_node_embeddings={total:,}", flush=True)
    print(f"dim={dim}", flush=True)

    ids_path = Path(args.ids_out)
    npy_path = Path(args.embeddings_out)
    ids_path.parent.mkdir(parents=True, exist_ok=True)
    npy_path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.lib.format.open_memmap(npy_path, mode="w+", dtype=np.float32, shape=(total, dim))

    fieldnames = [
        "id",
        "expert_id",
        "expert_name",
        "node_id",
        "node_name",
        "node_level",
        "is_direct_node",
        "subtree_weight_sum",
        "direct_leaf_count",
        "child_count",
        "evidence_paper_count",
        "top_evidence_papers",
    ]
    row_idx = 0
    with ids_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for expert_idx, (expert_id, rows) in enumerate(iter_expert_groups(evidence_path), start=1):
            expert_name = rows[0].get("expert_name", expert_id) if rows else expert_id
            records = build_for_expert(
                rows,
                ancestors,
                child_to_parents,
                node_embeddings,
                paper_id_to_row,
                paper_arr,
                args.node_weight,
            )
            for node_id, rec in records.items():
                emb_id = f"{expert_id}::{node_id}"
                arr[row_idx] = rec["embedding"]
                writer.writerow(
                    {
                        "id": emb_id,
                        "expert_id": expert_id,
                        "expert_name": expert_name,
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
                )
                row_idx += 1
            if expert_idx % 500 == 0:
                arr.flush()
                print(
                    f"write_progress experts={expert_idx:,}/{expert_count:,} rows={row_idx:,}/{total:,}",
                    flush=True,
                )

    arr.flush()
    if row_idx != total:
        raise RuntimeError(f"Expected to write {total} rows, wrote {row_idx}")
    print(f"experts={expert_count}")
    print(f"expert_node_embeddings={row_idx}")
    print(f"dim={dim}")
    print(f"ids_out={ids_path}")
    print(f"embeddings_out={npy_path}")


if __name__ == "__main__":
    main()
