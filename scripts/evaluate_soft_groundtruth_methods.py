#!/usr/bin/env python3
"""Evaluate team predictions with soft ground-truth matching.

    The historical team remains the anchor set. A predicted expert can match a
    ground-truth member exactly, through citation-graph blocks, or through user
    embedding distance. Per-task hits are computed by maximum bipartite matching
    so duplicate predictions cannot all claim the same ground-truth member.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple

import networkx as nx
import numpy as np


DEFAULT_METHOD_PREDICTIONS = (
    (
        "embedding_bfs",
        "output/embedding_bfs_unique_assignment_no_label/predictions_team_size.tsv",
    ),
    (
        "responsibility_cut_assign",
        "output/embedding_taxonomy_owner_gain_cut_topm256_no_label/predictions_team_size.tsv",
    ),
    (
        "expert_distribution_cut_assign",
        "output/embedding_taxonomy_region_cut_jsd_topm256_temp015_no_label/predictions_team_size.tsv",
    ),
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task-nodes-jsonl", default="output/hierec_embedding_server_inputs/task_nodes.jsonl")
    p.add_argument("--seq2seq-pred-csv", default="output/test.fold0.epoch15527.pred.csv")
    p.add_argument("--indexes-pkl", default="output/indexes.pkl")
    p.add_argument("--expert-papers-tsv", default="output/all_expert_paper_embeddings/expert_papers.tsv")
    p.add_argument("--paper-ids-tsv", default="output/all_expert_paper_embeddings/paper_embedding_ids.tsv")
    p.add_argument("--paper-embeddings", default="output/all_expert_paper_embeddings/paper_embeddings.npy")
    p.add_argument("--dblp-json", default="data/dblp/dblp.v12.json")
    p.add_argument("--out-dir", default="output/soft_groundtruth_evaluation")
    p.add_argument(
        "--embedding-distance-thresholds",
        default="0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.50",
        help="Cosine-distance thresholds; a pair matches when 1 - cosine <= threshold.",
    )
    p.add_argument(
        "--citation-community-method",
        choices=("louvain", "greedy_modularity", "connected_components"),
        default="louvain",
        help="How to turn the induced citation graph into paper blocks.",
    )
    p.add_argument("--citation-louvain-resolution", type=float, default=1.0)
    p.add_argument(
        "--citation-louvain-resolution-grid",
        default="",
        help=(
            "Comma-separated Louvain resolution values. When provided, the "
            "citation graph is built once and citation metrics are reported "
            "for every resolution."
        ),
    )
    p.add_argument("--citation-community-seed", type=int, default=13)
    p.add_argument("--random-runs", type=int, default=5)
    p.add_argument("--random-seed", type=int, default=13)
    p.add_argument(
        "--random-pool-scope",
        choices=("method_pool", "all_experts"),
        default="method_pool",
        help=(
            "method_pool samples from experts appearing in evaluated methods or "
            "history teams; all_experts samples from the full expert_papers.tsv pool."
        ),
    )
    p.add_argument(
        "--method-prediction",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="Additional predictions_team_size.tsv to evaluate.",
    )
    return p.parse_args()


def mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def as_members(value) -> List[str]:
    if isinstance(value, list):
        return [str(x) for x in value]
    if isinstance(value, str):
        return [x for x in value.replace("|", ",").split(",") if x]
    return []


def read_tasks(path: Path) -> Tuple[List[str], Dict[str, List[str]], Dict[str, int]]:
    paper_order: List[str] = []
    members_by_paper: Dict[str, List[str]] = {}
    team_size_by_paper: Dict[str, int] = {}
    seen = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            paper_id = str(row["paper_id"])
            if paper_id in seen:
                continue
            seen.add(paper_id)
            paper_order.append(paper_id)
            members_by_paper[paper_id] = as_members(row.get("members"))
            team_size_by_paper[paper_id] = int(row.get("team_size") or len(members_by_paper[paper_id]) or 1)
    return paper_order, members_by_paper, team_size_by_paper


def author_id_from_index_value(value) -> str:
    return str(value).split("_", 1)[0]


def load_seq2seq_predictions(
    pred_csv: Path,
    indexes_pkl: Path,
    paper_order: Sequence[str],
) -> Dict[str, List[str]]:
    with indexes_pkl.open("rb") as f:
        indexes = pickle.load(f)
    i2c = indexes["i2c"]
    out: Dict[str, List[str]] = {}
    with pred_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        for idx, row in enumerate(reader):
            if idx >= len(paper_order):
                break
            tokens = row[0].split() if row else []
            experts = []
            for token in tokens:
                if not token.startswith("m"):
                    continue
                try:
                    member_idx = int(token[1:])
                except ValueError:
                    continue
                if member_idx in i2c:
                    experts.append(author_id_from_index_value(i2c[member_idx]))
            out[paper_order[idx]] = dedupe(experts)
    return out


def dedupe(values: Iterable[str]) -> List[str]:
    seen = set()
    out = []
    for value in values:
        value = str(value)
        if value and value not in seen:
            seen.add(value)
            out.append(value)
    return out


def load_prediction_tsv(path: Path) -> Dict[str, Dict[str, List[str]]]:
    grouped: Dict[str, Dict[str, List[Tuple[int, str]]]] = defaultdict(lambda: defaultdict(list))
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            method = row.get("method") or path.parent.name
            paper_id = str(row["paper_id"])
            try:
                rank = int(float(row.get("rank") or 0))
            except ValueError:
                rank = 0
            grouped[method][paper_id].append((rank, str(row["expert_id"])))
    out: Dict[str, Dict[str, List[str]]] = {}
    for method, by_paper in grouped.items():
        out[method] = {
            paper_id: dedupe(expert for _, expert in sorted(rows))
            for paper_id, rows in by_paper.items()
        }
    return out


def parse_method_predictions(items: Sequence[str]) -> List[Tuple[str, Path]]:
    parsed = [(label, Path(path)) for label, path in DEFAULT_METHOD_PREDICTIONS]
    for item in items:
        if "=" not in item:
            raise SystemExit(f"--method-prediction must be LABEL=PATH: {item}")
        label, path = item.split("=", 1)
        parsed.append((label.strip(), Path(path)))
    return parsed


def load_all_predictions(
    args: argparse.Namespace,
    paper_order: Sequence[str],
    members_by_paper: Dict[str, List[str]],
) -> Dict[str, Dict[str, List[str]]]:
    predictions: Dict[str, Dict[str, List[str]]] = {
        "groundtruth_history_team": {
            paper_id: list(members_by_paper.get(paper_id, [])) for paper_id in paper_order
        },
        "seq2seq_epoch15527": load_seq2seq_predictions(
            Path(args.seq2seq_pred_csv), Path(args.indexes_pkl), paper_order
        ),
    }
    for label, path in parse_method_predictions(args.method_prediction):
        if not path.exists():
            print(f"skip_missing_prediction={path}", flush=True)
            continue
        by_method = load_prediction_tsv(path)
        if len(by_method) == 1:
            predictions[label] = next(iter(by_method.values()))
        else:
            for method_name, by_paper in by_method.items():
                predictions[f"{label}:{method_name}"] = by_paper
    return predictions


def read_expert_pool(path: Path) -> List[str]:
    experts = set()
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            experts.add(str(row["expert_id"]))
    return sorted(experts)


def add_random_predictions(
    predictions: Dict[str, Dict[str, List[str]]],
    paper_order: Sequence[str],
    team_size_by_paper: Dict[str, int],
    candidate_pool: Sequence[str],
    runs: int,
    seed: int,
) -> None:
    if runs <= 0:
        return
    pool = list(candidate_pool)
    if not pool:
        return
    for run_idx in range(runs):
        rng = random.Random(seed + run_idx)
        by_paper = {}
        for paper_id in paper_order:
            k = max(1, int(team_size_by_paper.get(paper_id, 1)))
            if k <= len(pool):
                by_paper[paper_id] = rng.sample(pool, k)
            else:
                by_paper[paper_id] = [rng.choice(pool) for _ in range(k)]
        predictions[f"random_seed_{seed + run_idx}"] = by_paper


def load_expert_papers(path: Path, needed_experts: Set[str]) -> Dict[str, Set[str]]:
    out: Dict[str, Set[str]] = {expert_id: set() for expert_id in needed_experts}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            expert_id = str(row["expert_id"])
            if expert_id in out:
                out[expert_id].add(str(row["paper_id"]))
    return out


def iter_dblp_objects(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if not text or text in {"[", "]"}:
                continue
            if text.startswith(","):
                text = text[1:].lstrip()
            if text.endswith(","):
                text = text[:-1].rstrip()
            if text:
                yield json.loads(text)


def build_citation_graph(
    dblp_json: Path,
    needed_papers: Set[str],
) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(sorted(needed_papers))

    for obj in iter_dblp_objects(dblp_json):
        paper_id = str(obj.get("id", ""))
        refs = [str(ref) for ref in obj.get("references") or []]
        if paper_id not in needed_papers:
            continue
        for ref in refs:
            if ref in needed_papers and ref != paper_id:
                graph.add_edge(paper_id, ref)
    return graph


def detect_citation_communities(
    graph: nx.Graph,
    method: str,
    resolution: float,
    seed: int,
) -> List[Set[str]]:
    if graph.number_of_nodes() == 0:
        return []
    if method == "connected_components":
        return [set(component) for component in nx.connected_components(graph)]
    if method == "greedy_modularity":
        return [set(community) for community in nx.community.greedy_modularity_communities(graph)]
    return [
        set(community)
        for community in nx.community.louvain_communities(
            graph,
            resolution=resolution,
            seed=seed,
        )
    ]


def citation_block_sets_from_graph(
    graph: nx.Graph,
    expert_papers: Dict[str, Set[str]],
    method: str,
    resolution: float,
    seed: int,
) -> Tuple[Dict[str, Set[str]], dict]:
    communities = detect_citation_communities(graph, method, resolution, seed)
    paper_to_block = {}
    for idx, community in enumerate(communities):
        block_id = f"c{idx}"
        for paper_id in community:
            paper_to_block[paper_id] = block_id

    block_sizes = [len(community) for community in communities]
    stats = {
        "community_method": method,
        "resolution": f"{resolution:.6f}" if method == "louvain" else "",
        "seed": seed if method == "louvain" else "",
        "papers": graph.number_of_nodes(),
        "citation_edges": graph.number_of_edges(),
        "communities": len(communities),
        "largest_community_papers": max(block_sizes) if block_sizes else 0,
        "avg_community_papers": f"{mean(block_sizes):.6f}",
        "singleton_communities": sum(1 for size in block_sizes if size == 1),
    }
    return {
        expert_id: {paper_to_block[paper_id] for paper_id in papers if paper_id in paper_to_block}
        for expert_id, papers in expert_papers.items()
    }, stats


def build_citation_block_sets(
    dblp_json: Path,
    expert_papers: Dict[str, Set[str]],
    method: str,
    resolution: float,
    seed: int,
) -> Tuple[Dict[str, Set[str]], dict]:
    needed_papers = set().union(*expert_papers.values()) if expert_papers else set()
    graph = build_citation_graph(dblp_json, needed_papers)
    return citation_block_sets_from_graph(graph, expert_papers, method, resolution, seed)


def read_paper_id_to_row(path: Path) -> Dict[str, int]:
    out = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for idx, row in enumerate(reader):
            out[str(row["id"])] = idx
    return out


def build_user_embeddings(
    paper_ids_tsv: Path,
    paper_embeddings_path: Path,
    expert_papers: Dict[str, Set[str]],
) -> Dict[str, np.ndarray]:
    paper_to_row = read_paper_id_to_row(paper_ids_tsv)
    arr = np.load(paper_embeddings_path, mmap_mode="r")
    out = {}
    for expert_id, papers in sorted(expert_papers.items()):
        rows = [paper_to_row[p] for p in papers if p in paper_to_row]
        if not rows:
            continue
        vec = np.asarray(arr[rows], dtype=np.float32).mean(axis=0)
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            out[expert_id] = vec / norm
    return out


def max_bipartite_hits(preds: Sequence[str], golds: Sequence[str], is_match) -> int:
    preds = dedupe(preds)
    golds = dedupe(golds)
    matched_pred: Dict[int, int] = {}

    def dfs(gold_idx: int, seen: Set[int]) -> bool:
        for pred_idx, pred in enumerate(preds):
            if pred_idx in seen or not is_match(pred, golds[gold_idx]):
                continue
            seen.add(pred_idx)
            if pred_idx not in matched_pred or dfs(matched_pred[pred_idx], seen):
                matched_pred[pred_idx] = gold_idx
                return True
        return False

    hits = 0
    for gold_idx in range(len(golds)):
        if dfs(gold_idx, set()):
            hits += 1
    return hits


def summarize(
    eval_type: str,
    method: str,
    paper_order: Sequence[str],
    members_by_paper: Dict[str, List[str]],
    predictions_by_paper: Dict[str, List[str]],
    is_match,
    threshold: str = "",
) -> dict:
    task_rows = []
    for paper_id in paper_order:
        golds = members_by_paper.get(paper_id, [])
        preds = predictions_by_paper.get(paper_id, [])
        hits = max_bipartite_hits(preds, golds, is_match)
        task_rows.append(
            {
                "hits": hits,
                "predicted": len(dedupe(preds)),
                "gold": len(dedupe(golds)),
                "precision": hits / len(dedupe(preds)) if preds else 0.0,
                "recall": hits / len(dedupe(golds)) if golds else 0.0,
            }
        )

    micro_hits = sum(row["hits"] for row in task_rows)
    micro_pred = sum(row["predicted"] for row in task_rows)
    micro_gold = sum(row["gold"] for row in task_rows)
    return {
        "eval_type": eval_type,
        "threshold": threshold,
        "method": method,
        "tasks": len(task_rows),
        "macro_precision": f"{mean([row['precision'] for row in task_rows]):.12f}",
        "macro_recall": f"{mean([row['recall'] for row in task_rows]):.12f}",
        "percent_precision": f"{100 * mean([row['precision'] for row in task_rows]):.6f}",
        "percent_recall": f"{100 * mean([row['recall'] for row in task_rows]):.6f}",
        "micro_precision": f"{(micro_hits / micro_pred) if micro_pred else 0.0:.12f}",
        "micro_recall": f"{(micro_hits / micro_gold) if micro_gold else 0.0:.12f}",
        "micro_hits": micro_hits,
        "micro_predicted": micro_pred,
        "micro_gold": micro_gold,
        "avg_predicted": f"{mean([row['predicted'] for row in task_rows]):.6f}",
        "avg_gold": f"{mean([row['gold'] for row in task_rows]):.6f}",
    }


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


def parse_resolution_values(args: argparse.Namespace) -> List[float]:
    if not args.citation_louvain_resolution_grid:
        return [float(args.citation_louvain_resolution)]
    return [
        float(item)
        for item in args.citation_louvain_resolution_grid.split(",")
        if item.strip()
    ]


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    paper_order, members_by_paper, team_size_by_paper = read_tasks(Path(args.task_nodes_jsonl))
    predictions = load_all_predictions(args, paper_order, members_by_paper)
    method_pool = set()
    for members in members_by_paper.values():
        method_pool.update(members)
    for by_paper in predictions.values():
        for experts in by_paper.values():
            method_pool.update(experts)
    random_pool = (
        read_expert_pool(Path(args.expert_papers_tsv))
        if args.random_pool_scope == "all_experts"
        else sorted(method_pool)
    )
    add_random_predictions(
        predictions,
        paper_order,
        team_size_by_paper,
        random_pool,
        args.random_runs,
        args.random_seed,
    )

    needed_experts = set()
    for members in members_by_paper.values():
        needed_experts.update(members)
    for by_paper in predictions.values():
        for experts in by_paper.values():
            needed_experts.update(experts)

    print(f"tasks={len(paper_order)} methods={len(predictions)} needed_experts={len(needed_experts)}", flush=True)
    expert_papers = load_expert_papers(Path(args.expert_papers_tsv), needed_experts)

    print(f"building citation blocks method={args.citation_community_method}", flush=True)
    needed_papers = set().union(*expert_papers.values()) if expert_papers else set()
    citation_graph = build_citation_graph(Path(args.dblp_json), needed_papers)
    citation_configs = []
    if args.citation_community_method == "louvain":
        for resolution in parse_resolution_values(args):
            citation_configs.append((resolution, f"{resolution:.6f}"))
    else:
        citation_configs.append((args.citation_louvain_resolution, ""))

    print("building user embeddings", flush=True)
    user_embeddings = build_user_embeddings(
        Path(args.paper_ids_tsv), Path(args.paper_embeddings), expert_papers
    )

    rows = []

    def exact_match(pred: str, gold: str) -> bool:
        return pred == gold

    for method, by_paper in sorted(predictions.items()):
        rows.append(
            summarize(
                "exact_history_member",
                method,
                paper_order,
                members_by_paper,
                by_paper,
                exact_match,
            )
        )

    citation_stats_rows = []

    for citation_resolution, citation_param in citation_configs:
        print(
            f"detecting citation communities method={args.citation_community_method} "
            f"resolution={citation_resolution:.6f}",
            flush=True,
        )
        citation_blocks, citation_stats = citation_block_sets_from_graph(
            citation_graph,
            expert_papers,
            args.citation_community_method,
            citation_resolution,
            args.citation_community_seed,
        )
        citation_stats_rows.append(citation_stats)

        def citation_match(pred: str, gold: str) -> bool:
            if pred == gold:
                return True
            return bool(citation_blocks.get(pred, set()) & citation_blocks.get(gold, set()))

        for method, by_paper in sorted(predictions.items()):
            rows.append(
                summarize(
                    "citation_block",
                    method,
                    paper_order,
                    members_by_paper,
                    by_paper,
                    citation_match,
                    citation_param,
                )
            )

    thresholds = [
        float(item)
        for item in args.embedding_distance_thresholds.split(",")
        if item.strip()
    ]
    for threshold in thresholds:
        threshold_text = f"{threshold:.6f}"

        def embedding_match(pred: str, gold: str, threshold: float = threshold) -> bool:
            if pred == gold:
                return True
            pred_vec = user_embeddings.get(pred)
            gold_vec = user_embeddings.get(gold)
            if pred_vec is None or gold_vec is None:
                return False
            distance = 1.0 - float(np.dot(pred_vec, gold_vec))
            return distance <= threshold

        for method, by_paper in sorted(predictions.items()):
            rows.append(
                summarize(
                    "user_embedding_distance",
                    method,
                    paper_order,
                    members_by_paper,
                    by_paper,
                    embedding_match,
                    threshold_text,
                )
            )

    rows.extend(aggregate_random_rows(rows))

    output_path = out_dir / "soft_groundtruth_metrics.tsv"
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    with (out_dir / "citation_block_summary.tsv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(citation_stats_rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(citation_stats_rows)

    print(f"metrics={output_path}")
    for row in rows:
        if (
            row["eval_type"] in {"exact_history_member", "citation_block"}
            or row["threshold"] in {"0.200000", "0.300000"}
        ) and not row["method"].startswith("random_seed_"):
            print(
                f"{row['eval_type']} threshold={row['threshold'] or '-'} "
                f"{row['method']} p={row['percent_precision']} r={row['percent_recall']}"
            )


if __name__ == "__main__":
    main()
