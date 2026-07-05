#!/usr/bin/env python3
"""Evaluate trained Stage-1 projections by per-node Top-M union recall."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate_stage1_full_node_assignment import safe_div
from scripts.train_stage1 import canonical_dev_paths, canonical_official_test_paths
from src.stage1_smoke_training import Stage1Dataset, Stage1Paths, Stage1ProjectionModel, Stage1Task


MIN_TREE_DEV = Path("outputs/expanded_fos_role_descriptions_minimal_tree_dev_500_v3_retrieval_gpt5mini")
MIN_TREE_TEST = Path("outputs/expanded_fos_role_descriptions_minimal_tree_official_test_500_v3_retrieval_gpt5mini")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        default="outputs/stage1_training/minimal_tree_v3_retrieval_v1_untrained_topm20/checkpoint_last.pt",
    )
    parser.add_argument(
        "--out-dir",
        default="outputs/stage1_trained_topm_union/minimal_tree_v3_retrieval_v1_untrained_topm20",
    )
    parser.add_argument("--m-grid", default="1,3,5,10,20,50")
    parser.add_argument("--node-link-mode", choices=("direct", "minimal_tree"), default="minimal_tree")
    parser.add_argument("--node-weight-mode", choices=("unweighted", "weighted"), default="unweighted")
    parser.add_argument("--taxonomy-edges", default="../data/dblp/13.FieldOfStudyChildren.nt")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-papers", type=int, default=0)
    parser.add_argument("--skip-dev", action="store_true")
    parser.add_argument("--skip-official-test", action="store_true")
    parser.add_argument("--dev-role-descriptions-jsonl", default=str(MIN_TREE_DEV / "stage1_task_node_role_descriptions.jsonl"))
    parser.add_argument("--dev-role-ids-tsv", default=str(MIN_TREE_DEV / "role_description_embedding_ids.tsv"))
    parser.add_argument("--dev-role-embeddings", default=str(MIN_TREE_DEV / "role_description_embeddings.npy"))
    parser.add_argument(
        "--official-test-role-descriptions-jsonl",
        default=str(MIN_TREE_TEST / "stage1_task_node_role_descriptions.jsonl"),
    )
    parser.add_argument("--official-test-role-ids-tsv", default=str(MIN_TREE_TEST / "role_description_embedding_ids.tsv"))
    parser.add_argument("--official-test-role-embeddings", default=str(MIN_TREE_TEST / "role_description_embeddings.npy"))
    return parser.parse_args()


def override_role_paths(
    base: Stage1Paths,
    role_descriptions_jsonl: str,
    role_ids_tsv: str,
    role_embeddings_npy: str,
) -> Stage1Paths:
    return Stage1Paths(
        sample_jsonl=base.sample_jsonl,
        role_descriptions_jsonl=Path(role_descriptions_jsonl),
        role_ids_tsv=Path(role_ids_tsv),
        role_embeddings_npy=Path(role_embeddings_npy),
        history_author_papers_tsv=base.history_author_papers_tsv,
        history_ids_tsv=base.history_ids_tsv,
        history_embeddings_npy=base.history_embeddings_npy,
        history_paper_fos_weights_tsv=base.history_paper_fos_weights_tsv,
        expert_profile_dir=base.expert_profile_dir,
        fos_map_tsv=base.fos_map_tsv,
    )


def write_tsv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def top_indices(scores: np.ndarray, m: int) -> np.ndarray:
    if len(scores) <= m:
        return np.argsort(-scores)
    idx = np.argpartition(-scores, m - 1)[:m]
    return idx[np.argsort(-scores[idx])]


def load_model(checkpoint_path: Path, input_dim: int, device: torch.device) -> Stage1ProjectionModel:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get("config") or {}
    projection_dim = int(checkpoint.get("projection_dim") or config.get("projection_dim") or 256)
    model = Stage1ProjectionModel(input_dim, projection_dim).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


class TrainedTopMUnionEvaluator:
    def __init__(self, dataset: Stage1Dataset, model: Stage1ProjectionModel, device: torch.device) -> None:
        self.dataset = dataset
        self.model = model
        self.device = device
        self.role_embeddings = torch.from_numpy(dataset.role_embeddings).to(device)
        self.history_embeddings_np = dataset.history_embeddings
        self.node_matrix_cache: dict[str, tuple[list[str], np.ndarray]] = {}

    def role_query(self, role_record_id: str) -> np.ndarray:
        idx = self.dataset.role_id_to_idx[role_record_id]
        with torch.no_grad():
            raw = self.role_embeddings[idx].unsqueeze(0)
            projected = F.normalize(self.model.role_proj(raw), dim=-1)
        return projected.squeeze(0).detach().cpu().numpy().astype(np.float32)

    def node_author_matrix(self, node_id: str) -> tuple[list[str], np.ndarray]:
        cached = self.node_matrix_cache.get(node_id)
        if cached is not None:
            return cached
        author_ids: list[str] = []
        raw_vectors: list[np.ndarray] = []
        for author_id in self.dataset.author_ids:
            weighted_indices = self.dataset.author_node_history_weights.get(author_id, {}).get(node_id, [])
            if not weighted_indices:
                continue
            indices = [idx for idx, _ in weighted_indices]
            weights = np.asarray([weight for _, weight in weighted_indices], dtype=np.float32)
            weight_sum = float(weights.sum())
            if weight_sum <= 0.0:
                continue
            raw = (self.history_embeddings_np[indices] * weights[:, None]).sum(axis=0) / weight_sum
            raw_vectors.append(raw.astype(np.float32))
            author_ids.append(author_id)
        if not raw_vectors:
            matrix = np.zeros((0, self.model.expert_proj.out_features), dtype=np.float32)
            self.node_matrix_cache[node_id] = (author_ids, matrix)
            return self.node_matrix_cache[node_id]
        with torch.no_grad():
            raw_tensor = torch.from_numpy(np.vstack(raw_vectors).astype(np.float32)).to(self.device)
            projected = F.normalize(self.model.expert_proj(raw_tensor), dim=-1).cpu().numpy().astype(np.float32)
        self.node_matrix_cache[node_id] = (author_ids, projected)
        return self.node_matrix_cache[node_id]


def evaluate_split(
    split_name: str,
    dataset: Stage1Dataset,
    model: Stage1ProjectionModel,
    device: torch.device,
    out_dir: Path,
    m_grid: Sequence[int],
) -> dict[int, dict[str, object]]:
    split_dir = out_dir / split_name
    split_dir.mkdir(parents=True, exist_ok=True)
    evaluator = TrainedTopMUnionEvaluator(dataset, model, device)

    per_m_pred: dict[int, int] = {m: 0 for m in m_grid}
    per_m_hits: dict[int, int] = {m: 0 for m in m_grid}
    total_gold = 0
    macro_p: dict[int, list[float]] = {m: [] for m in m_grid}
    macro_r: dict[int, list[float]] = {m: [] for m in m_grid}
    macro_f1: dict[int, list[float]] = {m: [] for m in m_grid}
    per_task_rows: list[dict[str, object]] = []
    max_m = max(m_grid)

    for task in dataset.tasks:
        gold = set(task.author_ids)
        pred_by_m: dict[int, set[str]] = {m: set() for m in m_grid}
        for role_id, node_id in zip(task.role_record_ids, task.node_ids):
            query = evaluator.role_query(role_id)
            author_ids, expert_matrix = evaluator.node_author_matrix(node_id)
            if len(author_ids) == 0:
                continue
            scores = expert_matrix @ query
            selected = top_indices(scores, min(max_m, len(scores)))
            ranked_authors = [author_ids[int(idx)] for idx in selected]
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
            "papers": len(dataset.tasks),
            "candidate_authors": len(dataset.author_ids),
            "task_nodes": sum(len(task.node_ids) for task in dataset.tasks),
            "micro_precision": micro_p,
            "micro_recall": micro_r,
            "micro_f1": micro_f1,
            "macro_precision": float(np.mean(macro_p[m])) if macro_p[m] else 0.0,
            "macro_recall": float(np.mean(macro_r[m])) if macro_r[m] else 0.0,
            "macro_f1": float(np.mean(macro_f1[m])) if macro_f1[m] else 0.0,
            "avg_pred_size": safe_div(per_m_pred[m], len(dataset.tasks)),
            "total_pred": per_m_pred[m],
            "total_gold": total_gold,
            "total_hits": per_m_hits[m],
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
    device = torch.device(args.device)
    m_grid = [int(x) for x in str(args.m_grid).split(",") if x.strip()]

    split_paths: dict[str, Stage1Paths] = {}
    if not args.skip_dev:
        split_paths["dev"] = override_role_paths(
            canonical_dev_paths(),
            args.dev_role_descriptions_jsonl,
            args.dev_role_ids_tsv,
            args.dev_role_embeddings,
        )
    if not args.skip_official_test:
        split_paths["official_test"] = override_role_paths(
            canonical_official_test_paths(),
            args.official_test_role_descriptions_jsonl,
            args.official_test_role_ids_tsv,
            args.official_test_role_embeddings,
        )
    if not split_paths:
        raise SystemExit("No split selected")

    datasets: dict[str, Stage1Dataset] = {}
    for split_name, paths in split_paths.items():
        print(f"loading_{split_name}=1", flush=True)
        datasets[split_name] = Stage1Dataset(
            paths,
            max_papers=args.max_papers,
            node_link_mode=args.node_link_mode,
            node_weight_mode=args.node_weight_mode,
            taxonomy_edges=Path(args.taxonomy_edges),
        )

    first_dataset = next(iter(datasets.values()))
    model = load_model(Path(args.checkpoint), first_dataset.input_dim, device)
    all_summary: list[dict[str, object]] = []
    for split_name, dataset in datasets.items():
        if dataset.input_dim != first_dataset.input_dim:
            raise ValueError(f"{split_name} input dim does not match checkpoint input dim")
        summary = evaluate_split(split_name, dataset, model, device, out_dir, m_grid)
        all_summary.extend(summary.values())

    write_tsv(out_dir / "topm_union_summary.tsv", all_summary)
    (out_dir / "run_config.json").write_text(
        json.dumps(
            {
                "checkpoint": args.checkpoint,
                "out_dir": str(out_dir),
                "m_grid": m_grid,
                "node_link_mode": args.node_link_mode,
                "node_weight_mode": args.node_weight_mode,
                "taxonomy_edges": args.taxonomy_edges,
                "max_papers": args.max_papers,
                "splits": {
                    name: {key: str(value) for key, value in asdict(paths).items()}
                    for name, paths in split_paths.items()
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
