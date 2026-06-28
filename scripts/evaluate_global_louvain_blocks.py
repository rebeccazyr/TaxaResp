#!/usr/bin/env python3
"""Evaluate citation soft matching with global all-expert Louvain blocks."""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from statistics import median
from types import SimpleNamespace
from typing import Dict, Iterable, List, Sequence, Set, Tuple

os.environ.setdefault("MPLCONFIGDIR", str(Path("cache/matplotlib").resolve()))
sys.path.insert(0, str(Path("cache/pydeps").resolve()))
sys.path.insert(0, str(Path("scripts").resolve()))

import igraph as ig  # noqa: E402

from evaluate_soft_groundtruth_methods import (  # noqa: E402
    add_random_predictions,
    iter_dblp_objects,
    load_all_predictions,
    mean,
    read_tasks,
    summarize,
)


DEFAULT_RESOLUTIONS = "10,20,50,100"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task-nodes-jsonl", default="output/hierec_embedding_server_inputs/task_nodes.jsonl")
    p.add_argument("--seq2seq-pred-csv", default="output/test.fold0.epoch15527.pred.csv")
    p.add_argument("--indexes-pkl", default="output/indexes.pkl")
    p.add_argument("--expert-papers-tsv", default="output/all_expert_paper_embeddings/expert_papers.tsv")
    p.add_argument("--profile-dir", default="output/expert_profile_year_bins/all_2000_2019")
    p.add_argument("--dblp-json", default="data/dblp/dblp.v12.json")
    p.add_argument("--out-dir", default="output/global_louvain_blocks")
    p.add_argument("--resolutions", default=DEFAULT_RESOLUTIONS)
    p.add_argument("--random-runs", type=int, default=5)
    p.add_argument("--random-seed", type=int, default=13)
    p.add_argument("--edge-batch-size", type=int, default=500000)
    return p.parse_args()


def parse_resolutions(text: str) -> List[float]:
    return [float(item) for item in text.split(",") if item.strip()]


def load_profile_experts(profile_dir: Path) -> List[str]:
    return sorted(
        path.name.replace("_direct_fos_nodes.tsv", "")
        for path in profile_dir.glob("*_direct_fos_nodes.tsv")
        if not path.name.startswith("_")
    )


def load_all_expert_papers(
    expert_papers_tsv: Path,
    profile_experts: Sequence[str],
) -> Dict[str, Set[str]]:
    profile_set = set(profile_experts)
    expert_to_papers = {expert_id: set() for expert_id in profile_experts}
    with expert_papers_tsv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            expert_id = str(row["expert_id"])
            if expert_id in profile_set:
                expert_to_papers[expert_id].add(str(row["paper_id"]))
    return expert_to_papers


def build_igraph_citation_graph(
    dblp_json: Path,
    paper_to_idx: Dict[str, int],
    batch_size: int,
) -> ig.Graph:
    graph = ig.Graph(n=len(paper_to_idx), directed=False)
    edges: List[Tuple[int, int]] = []
    parsed = 0
    added = 0
    for obj in iter_dblp_objects(dblp_json):
        parsed += 1
        if parsed % 500000 == 0:
            print(
                f"scan_progress parsed={parsed:,} edges_added={added:,}",
                flush=True,
            )
        src = paper_to_idx.get(str(obj.get("id", "")))
        if src is None:
            continue
        for ref in obj.get("references") or []:
            dst = paper_to_idx.get(str(ref))
            if dst is None or dst == src:
                continue
            edges.append((src, dst))
            if len(edges) >= batch_size:
                graph.add_edges(edges)
                added += len(edges)
                edges.clear()
    if edges:
        graph.add_edges(edges)
        added += len(edges)
    print(f"raw_graph nodes={graph.vcount():,} edges={graph.ecount():,}", flush=True)
    graph.simplify(multiple=True, loops=True, combine_edges=None)
    print(f"simple_graph nodes={graph.vcount():,} edges={graph.ecount():,}", flush=True)
    return graph


def block_counts_for_experts(
    membership: Sequence[int],
    paper_to_idx: Dict[str, int],
    expert_to_papers: Dict[str, Set[str]],
    profile_experts: Sequence[str],
) -> Tuple[Dict[str, Set[int]], List[dict], dict]:
    expert_blocks: Dict[str, Set[int]] = {}
    count_rows = []
    counts_all = []
    counts_with_papers = []
    for expert_id in profile_experts:
        blocks = {
            int(membership[paper_to_idx[paper_id]])
            for paper_id in expert_to_papers[expert_id]
            if paper_id in paper_to_idx
        }
        expert_blocks[expert_id] = blocks
        block_count = len(blocks)
        history_count = len(expert_to_papers[expert_id])
        counts_all.append(block_count)
        if history_count:
            counts_with_papers.append(block_count)
        count_rows.append(
            {
                "expert_id": expert_id,
                "history_papers": history_count,
                "block_count": block_count,
            }
        )

    summary = {
        "experts_total_profiles": len(profile_experts),
        "experts_with_history_papers": len(counts_with_papers),
        "avg_blocks_per_expert_with_papers": f"{mean(counts_with_papers):.6f}",
        "avg_blocks_per_expert_all_profiles": f"{mean(counts_all):.6f}",
        "median_blocks_per_expert_with_papers": f"{median(counts_with_papers):.6f}",
        "max_blocks_per_expert": max(counts_all) if counts_all else 0,
        "experts_zero_blocks": sum(1 for value in counts_all if value == 0),
    }
    return expert_blocks, count_rows, summary


def summarize_citation_methods(
    predictions: Dict[str, Dict[str, List[str]]],
    paper_order: Sequence[str],
    members_by_paper: Dict[str, List[str]],
    expert_blocks: Dict[str, Set[int]],
    resolution_text: str,
) -> List[dict]:
    def citation_match(pred: str, gold: str) -> bool:
        if pred == gold:
            return True
        return bool(expert_blocks.get(pred, set()) & expert_blocks.get(gold, set()))

    rows = []
    for method, by_paper in sorted(predictions.items()):
        rows.append(
            summarize(
                "global_citation_louvain",
                method,
                paper_order,
                members_by_paper,
                by_paper,
                citation_match,
                resolution_text,
            )
        )
    return rows


def aggregate_random_rows(rows: List[dict]) -> List[dict]:
    grouped: Dict[Tuple[str, str], List[dict]] = {}
    for row in rows:
        if not row["method"].startswith("random_seed_"):
            continue
        grouped.setdefault((row["eval_type"], row["threshold"]), []).append(row)

    out = []
    for (eval_type, threshold), group in sorted(grouped.items()):
        micro_hits = sum(int(row["micro_hits"]) for row in group)
        micro_pred = sum(int(row["micro_predicted"]) for row in group)
        micro_gold = sum(int(row["micro_gold"]) for row in group)
        out.append(
            {
                "eval_type": eval_type,
                "threshold": threshold,
                "method": f"random_mean_{len(group)}",
                "tasks": group[0]["tasks"],
                "macro_precision": f"{mean([float(row['macro_precision']) for row in group]):.12f}",
                "macro_recall": f"{mean([float(row['macro_recall']) for row in group]):.12f}",
                "percent_precision": f"{mean([float(row['percent_precision']) for row in group]):.6f}",
                "percent_recall": f"{mean([float(row['percent_recall']) for row in group]):.6f}",
                "micro_precision": f"{(micro_hits / micro_pred) if micro_pred else 0.0:.12f}",
                "micro_recall": f"{(micro_hits / micro_gold) if micro_gold else 0.0:.12f}",
                "micro_hits": micro_hits,
                "micro_predicted": micro_pred,
                "micro_gold": micro_gold,
                "avg_predicted": f"{mean([float(row['avg_predicted']) for row in group]):.6f}",
                "avg_gold": f"{mean([float(row['avg_gold']) for row in group]):.6f}",
            }
        )
    return out


def write_tsv(path: Path, rows: List[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    profile_experts = load_profile_experts(Path(args.profile_dir))
    expert_to_papers = load_all_expert_papers(Path(args.expert_papers_tsv), profile_experts)
    all_papers = sorted(set().union(*expert_to_papers.values()))
    paper_to_idx = {paper_id: idx for idx, paper_id in enumerate(all_papers)}
    print(
        f"profiles={len(profile_experts):,} "
        f"experts_with_papers={sum(1 for v in expert_to_papers.values() if v):,} "
        f"papers={len(all_papers):,}",
        flush=True,
    )

    paper_order, members_by_paper, team_size_by_paper = read_tasks(Path(args.task_nodes_jsonl))
    soft_args = SimpleNamespace(
        seq2seq_pred_csv=args.seq2seq_pred_csv,
        indexes_pkl=args.indexes_pkl,
        method_prediction=[],
    )
    predictions = load_all_predictions(soft_args, paper_order, members_by_paper)
    add_random_predictions(
        predictions,
        paper_order,
        team_size_by_paper,
        profile_experts,
        args.random_runs,
        args.random_seed,
    )

    graph = build_igraph_citation_graph(
        Path(args.dblp_json),
        paper_to_idx,
        args.edge_batch_size,
    )

    metric_rows = []
    block_summary_rows = []
    block_count_rows = []
    for resolution in parse_resolutions(args.resolutions):
        resolution_text = f"{resolution:.6f}"
        print(f"louvain_resolution={resolution_text}", flush=True)
        clustering = graph.community_multilevel(resolution=resolution)
        membership = clustering.membership
        community_sizes = clustering.sizes()
        expert_blocks, count_rows, block_summary = block_counts_for_experts(
            membership,
            paper_to_idx,
            expert_to_papers,
            profile_experts,
        )
        block_summary_rows.append(
            {
                "resolution": resolution_text,
                "paper_nodes": graph.vcount(),
                "citation_edges": graph.ecount(),
                "communities": len(community_sizes),
                "largest_community_papers": max(community_sizes) if community_sizes else 0,
                "avg_community_papers": f"{mean(community_sizes):.6f}",
                "singleton_communities": sum(1 for size in community_sizes if size == 1),
                **block_summary,
            }
        )
        for row in count_rows:
            block_count_rows.append({"resolution": resolution_text, **row})

        metric_rows.extend(
            summarize_citation_methods(
                predictions,
                paper_order,
                members_by_paper,
                expert_blocks,
                resolution_text,
            )
        )
        print(block_summary_rows[-1], flush=True)

    metric_rows.extend(aggregate_random_rows(metric_rows))
    write_tsv(out_dir / "global_louvain_metrics.tsv", metric_rows)
    write_tsv(out_dir / "global_louvain_block_summary.tsv", block_summary_rows)
    write_tsv(out_dir / "global_louvain_expert_block_counts.tsv", block_count_rows)
    print(f"metrics={out_dir / 'global_louvain_metrics.tsv'}")
    print(f"block_summary={out_dir / 'global_louvain_block_summary.tsv'}")
    print(f"expert_block_counts={out_dir / 'global_louvain_expert_block_counts.tsv'}")


if __name__ == "__main__":
    main()
