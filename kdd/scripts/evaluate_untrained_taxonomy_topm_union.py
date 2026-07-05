#!/usr/bin/env python3
"""Evaluate Stage-1 candidate recall by per-node top-m union."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate_untrained_taxonomy_aggregated_expert_nodes import (
    AncestorResolver,
    MinimalTreeResolver,
    UntrainedTaxonomyAssigner,
    load_child_to_parents,
    override_paths,
    safe_div,
    write_tsv,
)
from scripts.train_stage1 import canonical_dev_paths, canonical_official_test_paths
from src.stage1_smoke_training import Stage1Paths, Stage1Task, load_fos_name_to_id, load_tasks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="outputs/untrained_taxonomy_topm_union")
    parser.add_argument("--link-mode", choices=("direct", "ancestor", "minimal_tree"), default="ancestor")
    parser.add_argument("--node-weight-mode", choices=("unweighted", "weighted"), default="unweighted")
    parser.add_argument("--taxonomy-edges", default="../data/dblp/13.FieldOfStudyChildren.nt")
    parser.add_argument("--m-grid", default="1,3,5,10,20,50")
    parser.add_argument("--skip-dev", action="store_true")
    parser.add_argument("--skip-official-test", action="store_true")
    parser.add_argument("--dev-sample-jsonl", default="")
    parser.add_argument("--dev-role-descriptions-jsonl", default="")
    parser.add_argument("--dev-role-ids-tsv", default="")
    parser.add_argument("--dev-role-embeddings", default="")
    parser.add_argument("--official-test-sample-jsonl", default="")
    parser.add_argument("--official-test-role-descriptions-jsonl", default="")
    parser.add_argument("--official-test-role-ids-tsv", default="")
    parser.add_argument("--official-test-role-embeddings", default="")
    parser.add_argument("--max-papers", type=int, default=0)
    return parser.parse_args()


def top_indices(scores: np.ndarray, m: int) -> np.ndarray:
    if len(scores) <= m:
        return np.argsort(-scores)
    idx = np.argpartition(-scores, m - 1)[:m]
    return idx[np.argsort(-scores[idx])]


def evaluate_split(
    split_name: str,
    paths: Stage1Paths,
    tasks: Sequence[Stage1Task],
    out_dir: Path,
    link_mode: str,
    node_weight_mode: str,
    ancestor_resolver: AncestorResolver | MinimalTreeResolver | None,
    m_grid: Sequence[int],
) -> dict[int, dict[str, object]]:
    split_dir = out_dir / split_name
    split_dir.mkdir(parents=True, exist_ok=True)
    assigner = UntrainedTaxonomyAssigner(
        paths=paths,
        tasks=tasks,
        link_mode=link_mode,
        node_weight_mode=node_weight_mode,
        ancestor_resolver=ancestor_resolver,
        include_root_node=False,
        root_queries=None,
    )

    per_m_pred: dict[int, int] = {m: 0 for m in m_grid}
    per_m_hits: dict[int, int] = {m: 0 for m in m_grid}
    total_gold = 0
    macro_p: dict[int, list[float]] = {m: [] for m in m_grid}
    macro_r: dict[int, list[float]] = {m: [] for m in m_grid}
    macro_f1: dict[int, list[float]] = {m: [] for m in m_grid}
    per_task_rows: list[dict[str, object]] = []
    max_m = max(m_grid)

    for task in tasks:
        gold = set(task.author_ids)
        pred_by_m: dict[int, set[str]] = {m: set() for m in m_grid}
        for role_id, node_id in zip(task.role_record_ids, task.node_ids):
            query = assigner.role_query(role_id)
            cols, expert_matrix = assigner.node_author_matrix(node_id)
            if len(cols) == 0:
                continue
            scores = expert_matrix @ query
            selected = top_indices(scores, min(max_m, len(scores)))
            ranked_authors = [assigner.author_ids[int(cols[idx])] for idx in selected]
            for m in m_grid:
                pred_by_m[m].update(ranked_authors[:m])

        total_gold += len(gold)
        for m in m_grid:
            preds = pred_by_m[m]
            hits = len(preds & gold)
            precision = safe_div(hits, len(preds))
            recall = safe_div(hits, len(gold))
            f1 = safe_div(2.0 * precision * recall, precision + recall)
            per_m_pred[m] += len(preds)
            per_m_hits[m] += hits
            macro_p[m].append(precision)
            macro_r[m].append(recall)
            macro_f1[m].append(f1)
            per_task_rows.append(
                {
                    "m": m,
                    "paper_id": task.paper_id,
                    "pred_size": len(preds),
                    "gold_size": len(gold),
                    "hits": hits,
                    "precision": f"{precision:.8g}",
                    "recall": f"{recall:.8g}",
                    "f1": f"{f1:.8g}",
                }
            )

    summary_by_m: dict[int, dict[str, object]] = {}
    summary_rows: list[dict[str, object]] = []
    for m in m_grid:
        micro_p = safe_div(per_m_hits[m], per_m_pred[m])
        micro_r = safe_div(per_m_hits[m], total_gold)
        micro_f1 = safe_div(2.0 * micro_p * micro_r, micro_p + micro_r)
        row: dict[str, object] = {
            "split": split_name,
            "m": m,
            "papers": len(tasks),
            "candidate_authors": len(assigner.author_ids),
            "task_nodes": sum(len(task.node_ids) for task in tasks),
            "link_mode": link_mode,
            "node_weight_mode": node_weight_mode,
            "micro_precision": micro_p,
            "micro_recall": micro_r,
            "micro_f1": micro_f1,
            "macro_precision": float(np.mean(macro_p[m])) if macro_p[m] else 0.0,
            "macro_recall": float(np.mean(macro_r[m])) if macro_r[m] else 0.0,
            "macro_f1": float(np.mean(macro_f1[m])) if macro_f1[m] else 0.0,
            "avg_pred_size": safe_div(per_m_pred[m], len(tasks)),
            "total_pred": per_m_pred[m],
            "total_gold": total_gold,
            "total_hits": per_m_hits[m],
            "expert_nodes_with_vectors": int(assigner.build_stats["author_node_pairs"]),
            "paper_node_links_used": int(assigner.build_stats["paper_node_links_used"]),
        }
        summary_by_m[m] = row
        summary_rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)

    write_tsv(split_dir / "topm_union_summary.tsv", summary_rows)
    write_tsv(split_dir / "topm_union_per_task.tsv", per_task_rows)
    return summary_by_m


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    m_grid = [int(x) for x in str(args.m_grid).split(",") if x.strip()]

    split_paths: dict[str, Stage1Paths] = {}
    if not args.skip_dev:
        split_paths["dev"] = override_paths(
            canonical_dev_paths(),
            sample_jsonl=args.dev_sample_jsonl,
            role_descriptions_jsonl=args.dev_role_descriptions_jsonl,
            role_ids_tsv=args.dev_role_ids_tsv,
            role_embeddings_npy=args.dev_role_embeddings,
        )
    if not args.skip_official_test:
        split_paths["official_test"] = override_paths(
            canonical_official_test_paths(),
            sample_jsonl=args.official_test_sample_jsonl,
            role_descriptions_jsonl=args.official_test_role_descriptions_jsonl,
            role_ids_tsv=args.official_test_role_ids_tsv,
            role_embeddings_npy=args.official_test_role_embeddings,
        )

    ancestor_resolver = None
    child_to_parents: dict[str, set[str]] = {}
    if args.link_mode in {"ancestor", "minimal_tree"}:
        child_to_parents = load_child_to_parents(Path(args.taxonomy_edges))
        ancestor_resolver = (
            AncestorResolver(child_to_parents)
            if args.link_mode == "ancestor"
            else MinimalTreeResolver(child_to_parents)
        )

    all_summary: list[dict[str, object]] = []
    for split_name, paths in split_paths.items():
        name_to_fos_id = load_fos_name_to_id(paths.fos_map_tsv)
        tasks = load_tasks(
            paths.sample_jsonl,
            paths.role_descriptions_jsonl,
            name_to_fos_id,
            max_papers=args.max_papers,
        )
        summary = evaluate_split(
            split_name=split_name,
            paths=paths,
            tasks=tasks,
            out_dir=out_dir,
            link_mode=args.link_mode,
            node_weight_mode=args.node_weight_mode,
            ancestor_resolver=ancestor_resolver,
            m_grid=m_grid,
        )
        all_summary.extend(summary.values())

    write_tsv(out_dir / "topm_union_summary.tsv", all_summary)
    (out_dir / "run_config.json").write_text(
        json.dumps(
            {
                "out_dir": str(out_dir),
                "link_mode": args.link_mode,
                "node_weight_mode": args.node_weight_mode,
                "m_grid": m_grid,
                "max_papers": args.max_papers,
                "splits": {
                    name: {key: str(value) for key, value in asdict(paths).items()}
                    for name, paths in split_paths.items()
                },
                "taxonomy_edges": args.taxonomy_edges,
                "taxonomy_child_count": len(child_to_parents),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
