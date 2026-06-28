#!/usr/bin/env python3
"""Evaluate citation soft matching with Louvain blocks on an expert graph."""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import defaultdict
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task-nodes-jsonl", default="output/hierec_embedding_server_inputs/task_nodes.jsonl")
    p.add_argument("--seq2seq-pred-csv", default="output/test.fold0.epoch15527.pred.csv")
    p.add_argument("--indexes-pkl", default="output/indexes.pkl")
    p.add_argument("--expert-papers-tsv", default="output/all_expert_paper_embeddings/expert_papers.tsv")
    p.add_argument("--profile-dir", default="output/expert_profile_year_bins/all_2000_2019")
    p.add_argument("--dblp-json", default="data/dblp/dblp.v12.json")
    p.add_argument("--out-dir", default="output/expert_citation_louvain_blocks")
    p.add_argument("--resolutions", default="1,2,5,10,20,50,100")
    p.add_argument("--random-runs", type=int, default=5)
    p.add_argument("--random-seed", type=int, default=13)
    p.add_argument(
        "--method-prediction",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="Additional predictions_team_size.tsv to evaluate.",
    )
    return p.parse_args()


def parse_resolutions(text: str) -> List[float]:
    return [float(item) for item in text.split(",") if item.strip()]


def load_profile_experts(profile_dir: Path) -> List[str]:
    return sorted(
        path.name.replace("_direct_fos_nodes.tsv", "")
        for path in profile_dir.glob("*_direct_fos_nodes.tsv")
        if not path.name.startswith("_")
    )


def load_paper_experts(
    expert_papers_tsv: Path,
    profile_experts: Sequence[str],
) -> Dict[str, List[int]]:
    expert_to_idx = {expert_id: idx for idx, expert_id in enumerate(profile_experts)}
    paper_experts: Dict[str, List[int]] = defaultdict(list)
    with expert_papers_tsv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            expert_idx = expert_to_idx.get(str(row["expert_id"]))
            if expert_idx is not None:
                paper_experts[str(row["paper_id"])].append(expert_idx)
    for paper_id in list(paper_experts):
        paper_experts[paper_id] = sorted(set(paper_experts[paper_id]))
    return paper_experts


def build_expert_citation_graph(
    dblp_json: Path,
    paper_experts: Dict[str, List[int]],
    expert_count: int,
) -> ig.Graph:
    edge_weights: Dict[Tuple[int, int], int] = defaultdict(int)
    parsed = 0
    paper_citation_edges = 0
    expert_pair_events = 0
    for obj in iter_dblp_objects(dblp_json):
        parsed += 1
        if parsed % 500000 == 0:
            print(
                f"scan_progress parsed={parsed:,} "
                f"paper_edges={paper_citation_edges:,} expert_pair_events={expert_pair_events:,} "
                f"expert_edges={len(edge_weights):,}",
                flush=True,
            )
        src_experts = paper_experts.get(str(obj.get("id", "")))
        if not src_experts:
            continue
        for ref in obj.get("references") or []:
            dst_experts = paper_experts.get(str(ref))
            if not dst_experts:
                continue
            paper_citation_edges += 1
            for src in src_experts:
                for dst in dst_experts:
                    if src == dst:
                        continue
                    a, b = (src, dst) if src < dst else (dst, src)
                    edge_weights[(a, b)] += 1
                    expert_pair_events += 1

    graph = ig.Graph(n=expert_count, directed=False)
    edges = list(edge_weights)
    graph.add_edges(edges)
    graph.es["weight"] = [edge_weights[edge] for edge in edges]
    print(
        f"expert_graph nodes={graph.vcount():,} edges={graph.ecount():,} "
        f"paper_citation_edges={paper_citation_edges:,} expert_pair_events={expert_pair_events:,}",
        flush=True,
    )
    return graph


def summarize_expert_blocks(
    graph: ig.Graph,
    membership: Sequence[int],
    profile_experts: Sequence[str],
    resolution_text: str,
) -> dict:
    community_sizes = defaultdict(int)
    for block_id in membership:
        community_sizes[int(block_id)] += 1
    sizes = list(community_sizes.values())
    degrees = graph.degree()
    return {
        "resolution": resolution_text,
        "expert_nodes": graph.vcount(),
        "expert_edges": graph.ecount(),
        "isolated_experts": sum(1 for degree in degrees if degree == 0),
        "communities": len(sizes),
        "largest_community_experts": max(sizes) if sizes else 0,
        "avg_experts_per_community": f"{mean(sizes):.6f}",
        "median_experts_per_community": f"{median(sizes):.6f}" if sizes else "0.000000",
        "singleton_communities": sum(1 for size in sizes if size == 1),
        "avg_blocks_per_expert": "1.000000",
    }


def summarize_citation_methods(
    predictions: Dict[str, Dict[str, List[str]]],
    paper_order: Sequence[str],
    members_by_paper: Dict[str, List[str]],
    expert_to_block: Dict[str, int],
    resolution_text: str,
) -> List[dict]:
    def citation_match(pred: str, gold: str) -> bool:
        if pred == gold:
            return True
        pred_block = expert_to_block.get(pred)
        gold_block = expert_to_block.get(gold)
        return pred_block is not None and pred_block == gold_block

    rows = []
    for method, by_paper in sorted(predictions.items()):
        rows.append(
            summarize(
                "expert_graph_louvain",
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
    grouped: Dict[Tuple[str, str], List[dict]] = defaultdict(list)
    for row in rows:
        if row["method"].startswith("random_seed_"):
            grouped[(row["eval_type"], row["threshold"])].append(row)

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
    paper_experts = load_paper_experts(Path(args.expert_papers_tsv), profile_experts)
    print(
        f"experts={len(profile_experts):,} papers_with_experts={len(paper_experts):,}",
        flush=True,
    )

    paper_order, members_by_paper, team_size_by_paper = read_tasks(Path(args.task_nodes_jsonl))
    soft_args = SimpleNamespace(
        seq2seq_pred_csv=args.seq2seq_pred_csv,
        indexes_pkl=args.indexes_pkl,
        method_prediction=args.method_prediction,
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

    graph = build_expert_citation_graph(
        Path(args.dblp_json),
        paper_experts,
        len(profile_experts),
    )
    graph.vs["expert_id"] = profile_experts

    metric_rows = []
    block_rows = []
    for resolution in parse_resolutions(args.resolutions):
        resolution_text = f"{resolution:.6f}"
        print(f"louvain_resolution={resolution_text}", flush=True)
        clustering = graph.community_multilevel(weights="weight", resolution=resolution)
        membership = [int(x) for x in clustering.membership]
        expert_to_block = {
            expert_id: membership[idx] for idx, expert_id in enumerate(profile_experts)
        }
        block_rows.append(
            summarize_expert_blocks(graph, membership, profile_experts, resolution_text)
        )
        metric_rows.extend(
            summarize_citation_methods(
                predictions,
                paper_order,
                members_by_paper,
                expert_to_block,
                resolution_text,
            )
        )
        print(block_rows[-1], flush=True)

    metric_rows.extend(aggregate_random_rows(metric_rows))
    write_tsv(out_dir / "expert_graph_louvain_metrics.tsv", metric_rows)
    write_tsv(out_dir / "expert_graph_louvain_block_summary.tsv", block_rows)
    print(f"metrics={out_dir / 'expert_graph_louvain_metrics.tsv'}")
    print(f"block_summary={out_dir / 'expert_graph_louvain_block_summary.tsv'}")


if __name__ == "__main__":
    main()
