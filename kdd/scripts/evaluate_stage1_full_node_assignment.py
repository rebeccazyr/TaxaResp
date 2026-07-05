#!/usr/bin/env python3
"""Evaluate Stage-1 by full split-pool node-to-expert assignment."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_stage1 import canonical_dev_paths, canonical_official_test_paths
from src.stage1_smoke_training import Stage1Dataset, Stage1ProjectionModel, Stage1Task, TrainConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        default="outputs/stage1_training/task_expert_node_dev_and_test_v1/checkpoint_last.pt",
    )
    parser.add_argument("--out-dir", default="outputs/stage1_full_node_assignment/task_expert_node_v1")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-papers", type=int, default=0)
    parser.add_argument("--include-dev", action="store_true", default=True)
    parser.add_argument("--include-official-test", action="store_true", default=True)
    parser.add_argument("--assignment-mode", choices=("independent", "hungarian"), default="independent")
    parser.add_argument("--include-root-node", action="store_true")
    parser.add_argument(
        "--dev-root-ids-tsv",
        default="outputs/paper_to_expert_mean_embedding/dev_paper_embedding_ids.tsv",
    )
    parser.add_argument(
        "--dev-root-embeddings",
        default="outputs/paper_to_expert_mean_embedding/dev_paper_embeddings.npy",
    )
    parser.add_argument(
        "--official-test-root-ids-tsv",
        default="outputs/paper_to_expert_mean_embedding/test_paper_embedding_ids.tsv",
    )
    parser.add_argument(
        "--official-test-root-embeddings",
        default="outputs/paper_to_expert_mean_embedding/test_paper_embeddings.npy",
    )
    return parser.parse_args()


def safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def load_model(checkpoint_path: Path, input_dim: int, device: torch.device) -> Stage1ProjectionModel:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get("config") or {}
    projection_dim = int(checkpoint.get("projection_dim") or config.get("projection_dim") or 256)
    model = Stage1ProjectionModel(input_dim, projection_dim).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def read_ids_tsv(path: Path) -> list[str]:
    ids: list[str] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            ids.append(str(row["id"]))
    return ids


def load_root_queries(ids_tsv: Path, embeddings_npy: Path) -> dict[str, np.ndarray]:
    ids = read_ids_tsv(ids_tsv)
    embeddings = np.load(embeddings_npy)
    if len(ids) != int(embeddings.shape[0]):
        raise ValueError(f"root ids and embeddings row mismatch: {ids_tsv} {embeddings_npy}")
    return {paper_id: embeddings[idx].astype(np.float32) for idx, paper_id in enumerate(ids)}


class FullNodeAssigner:
    def __init__(
        self,
        dataset: Stage1Dataset,
        model: Stage1ProjectionModel,
        device: torch.device,
        root_queries: dict[str, np.ndarray] | None = None,
    ) -> None:
        self.dataset = dataset
        self.model = model
        self.device = device
        self.author_ids = list(dataset.author_ids)
        self.history_embeddings_np = dataset.history_embeddings
        self.role_embeddings = torch.from_numpy(dataset.role_embeddings).to(device)
        self.expert_cache: dict[tuple[str, str], torch.Tensor] = {}
        self.root_queries = root_queries or {}
        self.root_expert_cache: dict[str, torch.Tensor] = {}

    def projected_role(self, role_record_id: str) -> torch.Tensor:
        idx = self.dataset.role_id_to_idx[role_record_id]
        with torch.no_grad():
            q = self.role_embeddings[idx].unsqueeze(0)
            return F.normalize(self.model.role_proj(q), dim=-1).squeeze(0)

    def projected_root_query(self, paper_id: str) -> torch.Tensor | None:
        query = self.root_queries.get(paper_id)
        if query is None:
            return None
        with torch.no_grad():
            q = torch.from_numpy(query.astype(np.float32)).to(self.device).unsqueeze(0)
            return F.normalize(self.model.role_proj(q), dim=-1).squeeze(0).cpu()

    def projected_root_expert(self, author_id: str) -> torch.Tensor | None:
        cached = self.root_expert_cache.get(author_id)
        if cached is not None:
            return cached
        history_indices = self.dataset.author_history_indices.get(author_id, [])
        if not history_indices:
            return None
        history_np = self.history_embeddings_np[history_indices]
        z_np = np.asarray(history_np, dtype=np.float32).mean(axis=0)
        with torch.no_grad():
            z = torch.from_numpy(z_np.astype(np.float32)).to(self.device).unsqueeze(0)
            projected = F.normalize(self.model.expert_proj(z), dim=-1).squeeze(0).cpu()
        self.root_expert_cache[author_id] = projected
        return projected

    def projected_expert_node(self, author_id: str, node_id: str) -> torch.Tensor | None:
        key = (author_id, node_id)
        cached = self.expert_cache.get(key)
        if cached is not None:
            return cached

        weighted_indices = self.dataset.author_node_history_weights.get(author_id, {}).get(node_id, [])
        if not weighted_indices:
            return None
        history_indices = [idx for idx, _ in weighted_indices]
        weights_np = np.asarray([weight for _, weight in weighted_indices], dtype=np.float32)
        alpha_sum = float(weights_np.sum())
        if alpha_sum <= 0.0:
            return None

        history_np = self.history_embeddings_np[history_indices]
        z_np = (history_np * weights_np[:, None]).sum(axis=0) / alpha_sum
        with torch.no_grad():
            z = torch.from_numpy(z_np.astype(np.float32)).to(self.device).unsqueeze(0)
            projected = F.normalize(self.model.expert_proj(z), dim=-1).squeeze(0).cpu()
        self.expert_cache[key] = projected
        return projected

    def assign_node(self, role_record_id: str, node_id: str) -> tuple[str, float, int] | None:
        q = self.projected_role(role_record_id).cpu()
        best_author = ""
        best_score = -1.0e9
        valid_candidates = 0
        for author_id in self.author_ids:
            expert = self.projected_expert_node(author_id, node_id)
            if expert is None:
                continue
            valid_candidates += 1
            score = float(torch.dot(q, expert))
            if score > best_score:
                best_author = author_id
                best_score = score
        if not best_author:
            return None
        return best_author, best_score, valid_candidates

    def assign_task_hungarian(self, task: Stage1Task, include_root_node: bool = False) -> list[dict[str, object]]:
        rows = list(zip(task.role_record_ids, task.node_ids, task.node_names))
        if include_root_node:
            rows = [("__root__", "__root__", "Root paper abstract")] + rows
        if not rows:
            return []

        score_matrix = np.full((len(rows), len(self.author_ids)), -1.0e9, dtype=np.float32)
        valid_counts = np.zeros(len(rows), dtype=np.int32)
        for row_idx, (role_id, node_id, _) in enumerate(rows):
            q = self.projected_root_query(task.paper_id) if node_id == "__root__" else self.projected_role(role_id).cpu()
            if q is None:
                continue
            for col_idx, author_id in enumerate(self.author_ids):
                expert = (
                    self.projected_root_expert(author_id)
                    if node_id == "__root__"
                    else self.projected_expert_node(author_id, node_id)
                )
                if expert is None:
                    continue
                valid_counts[row_idx] += 1
                score_matrix[row_idx, col_idx] = float(torch.dot(q, expert))

        valid_rows = np.flatnonzero(valid_counts > 0)
        assignments: list[dict[str, object]] = []
        if len(valid_rows) > 0:
            sub_scores = score_matrix[valid_rows]
            row_ind, col_ind = linear_sum_assignment(sub_scores, maximize=True)
            selected_by_row = {int(valid_rows[int(r)]): int(c) for r, c in zip(row_ind, col_ind)}
        else:
            selected_by_row = {}

        for row_idx, (role_id, node_id, node_name) in enumerate(rows):
            col_idx = selected_by_row.get(row_idx)
            if col_idx is None or score_matrix[row_idx, col_idx] <= -1.0e8:
                assignments.append(
                    {
                        "paper_id": task.paper_id,
                        "role_record_id": role_id,
                        "node_id": node_id,
                        "node_name": node_name,
                        "pred_author_id": "",
                        "score": "",
                        "valid_candidates": int(valid_counts[row_idx]),
                        "is_gold": 0,
                    }
                )
                continue
            author_id = self.author_ids[col_idx]
            assignments.append(
                {
                    "paper_id": task.paper_id,
                    "role_record_id": role_id,
                    "node_id": node_id,
                    "node_name": node_name,
                    "pred_author_id": author_id,
                    "score": f"{float(score_matrix[row_idx, col_idx]):.8g}",
                    "valid_candidates": int(valid_counts[row_idx]),
                    "is_gold": int(author_id in set(task.author_ids)),
                }
            )
        return assignments


def evaluate_split(
    split_name: str,
    dataset: Stage1Dataset,
    model: Stage1ProjectionModel,
    device: torch.device,
    out_dir: Path,
    assignment_mode: str,
    include_root_node: bool,
    root_queries: dict[str, np.ndarray] | None = None,
) -> dict[str, float | int | str]:
    split_dir = out_dir / split_name
    split_dir.mkdir(parents=True, exist_ok=True)
    assigner = FullNodeAssigner(dataset, model, device, root_queries=root_queries)

    node_rows: list[dict[str, object]] = []
    task_rows: list[dict[str, object]] = []
    macro_precision_values: list[float] = []
    macro_recall_values: list[float] = []
    macro_f1_values: list[float] = []
    raw_precision_values: list[float] = []

    total_pred = 0
    total_gold = 0
    total_hits = 0
    raw_total = 0
    raw_hits = 0
    skipped_nodes = 0
    assigned_nodes = 0

    for task in dataset.tasks:
        gold = set(task.author_ids)
        predictions: list[str] = []
        raw_task_hits = 0
        if assignment_mode == "hungarian":
            task_assignments = assigner.assign_task_hungarian(task, include_root_node=include_root_node)
        else:
            task_assignments = []
            if include_root_node:
                raise ValueError("--include-root-node currently requires --assignment-mode hungarian")
            for role_id, node_id, node_name in zip(task.role_record_ids, task.node_ids, task.node_names):
                assignment = assigner.assign_node(role_id, node_id)
                if assignment is None:
                    task_assignments.append(
                        {
                            "paper_id": task.paper_id,
                            "role_record_id": role_id,
                            "node_id": node_id,
                            "node_name": node_name,
                            "pred_author_id": "",
                            "score": "",
                            "valid_candidates": 0,
                            "is_gold": 0,
                        }
                    )
                    continue
                pred_author, score, valid_candidates = assignment
                task_assignments.append(
                    {
                        "paper_id": task.paper_id,
                        "role_record_id": role_id,
                        "node_id": node_id,
                        "node_name": node_name,
                        "pred_author_id": pred_author,
                        "score": f"{score:.8g}",
                        "valid_candidates": valid_candidates,
                        "is_gold": int(pred_author in gold),
                    }
                )

        for assignment_row in task_assignments:
            pred_author = str(assignment_row["pred_author_id"])
            if not pred_author:
                skipped_nodes += 1
                node_rows.append(assignment_row)
                continue
            is_gold = int(assignment_row["is_gold"])
            predictions.append(pred_author)
            raw_task_hits += is_gold
            raw_hits += is_gold
            raw_total += 1
            assigned_nodes += 1
            node_rows.append(assignment_row)

        pred_set = set(predictions)
        hits = len(pred_set & gold)
        precision = safe_div(hits, len(pred_set))
        recall = safe_div(hits, len(gold))
        f1 = safe_div(2.0 * precision * recall, precision + recall)
        raw_precision = safe_div(raw_task_hits, len(predictions))

        macro_precision_values.append(precision)
        macro_recall_values.append(recall)
        macro_f1_values.append(f1)
        raw_precision_values.append(raw_precision)
        total_pred += len(pred_set)
        total_gold += len(gold)
        total_hits += hits
        task_rows.append(
            {
                "paper_id": task.paper_id,
                "gold_team_size": len(gold),
                "assigned_nodes": len(predictions),
                "skipped_nodes": (len(task.role_record_ids) + int(include_root_node)) - len(predictions),
                "pred_team_size_dedup": len(pred_set),
                "hits_dedup": hits,
                "precision_dedup": f"{precision:.8g}",
                "recall_dedup": f"{recall:.8g}",
                "f1_dedup": f"{f1:.8g}",
                "raw_node_assignment_precision": f"{raw_precision:.8g}",
                "gold_author_ids": ",".join(task.author_ids),
                "pred_author_ids_raw": ",".join(predictions),
                "pred_author_ids_dedup": ",".join(sorted(pred_set)),
            }
        )

    write_tsv(split_dir / "node_assignments.tsv", node_rows)
    write_tsv(split_dir / "per_task_metrics.tsv", task_rows)

    micro_precision = safe_div(total_hits, total_pred)
    micro_recall = safe_div(total_hits, total_gold)
    micro_f1 = safe_div(2.0 * micro_precision * micro_recall, micro_precision + micro_recall)
    metrics: dict[str, float | int | str] = {
        "split": split_name,
        "assignment_mode": assignment_mode,
        "include_root_node": int(include_root_node),
        "papers": len(dataset.tasks),
        "candidate_authors": len(dataset.author_ids),
        "assigned_nodes": assigned_nodes,
        "skipped_nodes_no_candidate": skipped_nodes,
        "raw_node_assignment_precision": safe_div(raw_hits, raw_total),
        "macro_precision_dedup": float(np.mean(macro_precision_values)) if macro_precision_values else 0.0,
        "macro_recall_dedup": float(np.mean(macro_recall_values)) if macro_recall_values else 0.0,
        "macro_f1_dedup": float(np.mean(macro_f1_values)) if macro_f1_values else 0.0,
        "micro_precision_dedup": micro_precision,
        "micro_recall_dedup": micro_recall,
        "micro_f1_dedup": micro_f1,
        "total_pred_dedup": total_pred,
        "total_gold": total_gold,
        "total_hits_dedup": total_hits,
    }
    write_tsv(split_dir / "metrics_summary.tsv", [metrics])
    return metrics


def write_tsv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    split_paths = {
        "dev": canonical_dev_paths(),
        "official_test": canonical_official_test_paths(),
    }
    datasets: dict[str, Stage1Dataset] = {}
    for split_name, paths in split_paths.items():
        print(f"loading_{split_name}=1", flush=True)
        datasets[split_name] = Stage1Dataset(paths, max_papers=args.max_papers)

    first_dataset = next(iter(datasets.values()))
    model = load_model(Path(args.checkpoint), first_dataset.input_dim, device)
    root_query_paths = {
        "dev": (Path(args.dev_root_ids_tsv), Path(args.dev_root_embeddings)),
        "official_test": (Path(args.official_test_root_ids_tsv), Path(args.official_test_root_embeddings)),
    }

    summaries = []
    for split_name, dataset in datasets.items():
        print(f"evaluating_{split_name}=1", flush=True)
        root_queries = None
        if args.include_root_node:
            ids_tsv, embeddings_npy = root_query_paths[split_name]
            root_queries = load_root_queries(ids_tsv, embeddings_npy)
        summaries.append(
            evaluate_split(
                split_name,
                dataset,
                model,
                device,
                out_dir,
                args.assignment_mode,
                args.include_root_node,
                root_queries=root_queries,
            )
        )
        print(json.dumps(summaries[-1], ensure_ascii=False), flush=True)

    write_tsv(out_dir / "metrics_summary.tsv", summaries)
    (out_dir / "run_config.json").write_text(
        json.dumps(
            {
                "checkpoint": args.checkpoint,
                "out_dir": args.out_dir,
                "device": args.device,
                "max_papers": args.max_papers,
                "assignment_mode": args.assignment_mode,
                "include_root_node": bool(args.include_root_node),
                "splits": {name: {key: str(value) for key, value in asdict(paths).items()} for name, paths in split_paths.items()},
                "root_query_paths": {
                    name: {"ids_tsv": str(paths[0]), "embeddings_npy": str(paths[1])}
                    for name, paths in root_query_paths.items()
                },
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
