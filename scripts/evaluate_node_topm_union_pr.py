#!/usr/bin/env python3
"""Non-team-size-constrained per-node top-m union precision/recall.

For each task, take the top-m same-node expert candidates for every task node,
union them into a single predicted expert set (NOT truncated to the gold team
size), and score precision/recall/F1 against the gold team members. Sweeps a
grid of m values.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--task-nodes-jsonl",
        default="output/hierec_embedding_server_inputs/task_nodes.jsonl",
    )
    p.add_argument(
        "--node-topm-candidates",
        default="output/completed_task_node_role_descriptions_eval/node_topm_candidates.tsv",
    )
    p.add_argument(
        "--m-grid",
        default="1,3,5,10,20",
        help="Comma-separated per-node top-m values to sweep.",
    )
    p.add_argument(
        "--out-dir",
        default="output/completed_task_node_role_descriptions_eval",
    )
    return p.parse_args()


def as_members(value) -> List[str]:
    if isinstance(value, list):
        return [str(x) for x in value]
    if isinstance(value, str):
        return [x for x in value.replace("|", ",").split(",") if x]
    return []


def dedupe(values) -> List[str]:
    seen = set()
    out = []
    for v in values:
        v = str(v)
        if v and v not in seen:
            seen.add(v)
            out.append(v)
    return out


def load_gold(path: Path) -> tuple[Dict[str, List[str]], List[str]]:
    gold: Dict[str, List[str]] = {}
    order: List[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            paper_id = str(row["paper_id"])
            if paper_id not in gold:
                gold[paper_id] = dedupe(as_members(row.get("members")))
                order.append(paper_id)
    return gold, order


def load_candidates(path: Path, max_m: int) -> Dict[str, Dict[str, List[str]]]:
    """paper_id -> node_id -> ranked expert_ids (rank<=max_m)."""
    by_paper: Dict[str, Dict[str, List[tuple]]] = defaultdict(lambda: defaultdict(list))
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            try:
                rank = int(float(row.get("rank") or 0))
            except (TypeError, ValueError):
                continue
            if rank <= 0 or rank > max_m:
                continue
            paper_id = str(row["paper_id"])
            node_id = str(row["node_id"])
            by_paper[paper_id][node_id].append((rank, str(row["expert_id"])))
    out: Dict[str, Dict[str, List[str]]] = {}
    for paper_id, nodes in by_paper.items():
        out[paper_id] = {}
        for node_id, pairs in nodes.items():
            pairs.sort(key=lambda x: x[0])
            out[paper_id][node_id] = [e for _, e in pairs]
    return out


def prf(pred: Sequence[str], gold: Sequence[str]) -> tuple[int, float, float, float]:
    pred_u = set(dedupe(pred))
    gold_u = set(dedupe(gold))
    hits = len(pred_u & gold_u)
    precision = hits / len(pred_u) if pred_u else 0.0
    recall = hits / len(gold_u) if gold_u else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return hits, precision, recall, f1


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    m_grid = [int(x) for x in str(args.m_grid).split(",") if x.strip()]
    max_m = max(m_grid)

    gold, paper_order = load_gold(Path(args.task_nodes_jsonl))
    cand = load_candidates(Path(args.node_topm_candidates), max_m)

    summary_rows = []
    per_task_rows = []
    for m in m_grid:
        macro_p = macro_r = macro_f1 = 0.0
        micro_hits = micro_pred = micro_gold = 0
        n = 0
        pred_sizes = []
        for paper_id in paper_order:
            golds = gold.get(paper_id, [])
            nodes = cand.get(paper_id, {})
            preds = []
            for node_id, experts in nodes.items():
                preds.extend(experts[:m])
            preds = dedupe(preds)
            hits, p, r, f1 = prf(preds, golds)
            macro_p += p
            macro_r += r
            macro_f1 += f1
            micro_hits += hits
            micro_pred += len(set(preds))
            micro_gold += len(set(golds))
            pred_sizes.append(len(preds))
            n += 1
            per_task_rows.append(
                {
                    "m": m,
                    "paper_id": paper_id,
                    "pred_size": len(preds),
                    "gold_size": len(set(golds)),
                    "hits": hits,
                    "precision": f"{p:.12f}",
                    "recall": f"{r:.12f}",
                    "f1": f"{f1:.12f}",
                }
            )
        n = max(n, 1)
        micro_p = micro_hits / micro_pred if micro_pred else 0.0
        micro_r = micro_hits / micro_gold if micro_gold else 0.0
        micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) else 0.0
        summary_rows.append(
            {
                "m": m,
                "tasks": n,
                "macro_precision_percent": f"{100 * macro_p / n:.6f}",
                "macro_recall_percent": f"{100 * macro_r / n:.6f}",
                "macro_f1_percent": f"{100 * macro_f1 / n:.6f}",
                "micro_precision_percent": f"{100 * micro_p:.6f}",
                "micro_recall_percent": f"{100 * micro_r:.6f}",
                "micro_f1_percent": f"{100 * micro_f1:.6f}",
                "avg_pred_size": f"{sum(pred_sizes) / n:.4f}",
                "micro_hits": micro_hits,
                "micro_predicted": micro_pred,
                "micro_gold": micro_gold,
            }
        )
        print(
            f"m={m} macro_p={summary_rows[-1]['macro_precision_percent']} "
            f"macro_r={summary_rows[-1]['macro_recall_percent']} "
            f"micro_p={summary_rows[-1]['micro_precision_percent']} "
            f"micro_r={summary_rows[-1]['micro_recall_percent']} "
            f"avg_pred={summary_rows[-1]['avg_pred_size']}",
            flush=True,
        )

    with (out_dir / "node_topm_union_pr_summary.tsv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()), delimiter="\t")
        writer.writeheader()
        writer.writerows(summary_rows)
    with (out_dir / "node_topm_union_pr_per_task.tsv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_task_rows[0].keys()), delimiter="\t")
        writer.writeheader()
        writer.writerows(per_task_rows)
    print(f"summary={out_dir / 'node_topm_union_pr_summary.tsv'}")


if __name__ == "__main__":
    main()
