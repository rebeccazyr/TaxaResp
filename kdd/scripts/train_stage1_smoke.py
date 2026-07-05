#!/usr/bin/env python3
"""Run Stage-1 smoke training on the KDD smoke_200 artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.stage1_smoke_training import Stage1Paths, Stage1SmokeDataset, TrainConfig, train_stage1_smoke


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-jsonl", default="outputs/stage1_pilot_samples/smoke_200.jsonl")
    parser.add_argument(
        "--role-descriptions-jsonl",
        default="outputs/stage1_task_node_role_descriptions_smoke_200/stage1_task_node_role_descriptions.jsonl",
    )
    parser.add_argument(
        "--role-ids-tsv",
        default="outputs/stage1_smoke_embeddings/role_description_embedding_ids.tsv",
    )
    parser.add_argument(
        "--role-embeddings",
        default="outputs/stage1_smoke_embeddings/role_description_embeddings.npy",
    )
    parser.add_argument(
        "--history-author-papers-tsv",
        default="outputs/stage1_smoke_embedding_inputs/author_history_papers.tsv",
    )
    parser.add_argument(
        "--history-ids-tsv",
        default="outputs/stage1_smoke_embeddings/history_paper_embedding_ids.tsv",
    )
    parser.add_argument(
        "--history-embeddings",
        default="outputs/stage1_smoke_embeddings/history_paper_embeddings.npy",
    )
    parser.add_argument(
        "--history-paper-fos-weights-tsv",
        default="outputs/stage1_smoke_embedding_inputs/history_paper_fos_weights.tsv",
    )
    parser.add_argument(
        "--expert-profile-dir",
        default="outputs/expert_profile_cutoffs/pre_2018_for_valid_2018",
    )
    parser.add_argument("--fos-map-tsv", default="../data/dblp/FieldsOfStudy.txt")
    parser.add_argument("--out-dir", default="outputs/stage1_smoke_training/v1_untrained_topm20")
    parser.add_argument("--citation-graph-dir", default="")

    parser.add_argument("--projection-dim", type=int, default=256)
    parser.add_argument("--tau", type=float, default=0.07)
    parser.add_argument("--tau-m", type=float, default=0.1)
    parser.add_argument("--alpha-top-k", type=int, default=16)
    parser.add_argument("--node-link-mode", choices=("direct", "minimal_tree"), default="minimal_tree")
    parser.add_argument("--node-weight-mode", choices=("unweighted", "weighted"), default="unweighted")
    parser.add_argument("--taxonomy-edges", default="../data/dblp/13.FieldOfStudyChildren.nt")
    parser.add_argument("--negative-mode", choices=("v1", "v2", "v3", "v4"), default="v1")
    parser.add_argument("--negative-pool-mode", choices=("batch", "global", "untrained_topm", "mixed"), default="untrained_topm")
    parser.add_argument(
        "--untrained-negative-top-m",
        type=int,
        default=20,
        help="Per-node frozen-embedding top-M union used as hard negatives for untrained_topm/mixed modes.",
    )
    parser.add_argument(
        "--untrained-node-cache-size",
        type=int,
        default=128,
        help="LRU cache size for frozen expert-node matrices used by untrained_topm hard-negative mining.",
    )
    parser.add_argument(
        "--global-negative-sample-size",
        type=int,
        default=128,
        help="Per-task random pool negatives for global/mixed modes. Use 0 for all pool negatives.",
    )
    parser.add_argument(
        "--author-node-cache-size",
        type=int,
        default=100000,
        help="LRU cache size for computed author-node vectors during Stage-1 loss.",
    )
    parser.add_argument("--pi0", type=float, default=0.5)
    parser.add_argument("--w-near", type=float, default=0.1)
    parser.add_argument("--w-far", type=float, default=1.0)
    parser.add_argument(
        "--prox-threshold",
        type=float,
        default=None,
        help="V3 threshold. If omitted, estimate from the train split by --prox-quantile.",
    )
    parser.add_argument("--prox-quantile", type=float, default=0.75)
    parser.add_argument("--prox-beta", type=float, default=20.0)
    parser.add_argument("--prox-bias", type=float, default=1.0)

    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--eval-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-papers", type=int, default=0, help="Debug limit; 0 means all papers.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = Stage1Paths(
        sample_jsonl=Path(args.sample_jsonl),
        role_descriptions_jsonl=Path(args.role_descriptions_jsonl),
        role_ids_tsv=Path(args.role_ids_tsv),
        role_embeddings_npy=Path(args.role_embeddings),
        history_author_papers_tsv=Path(args.history_author_papers_tsv),
        history_ids_tsv=Path(args.history_ids_tsv),
        history_embeddings_npy=Path(args.history_embeddings),
        history_paper_fos_weights_tsv=Path(args.history_paper_fos_weights_tsv),
        expert_profile_dir=Path(args.expert_profile_dir),
        fos_map_tsv=Path(args.fos_map_tsv),
    )
    config = TrainConfig(
        projection_dim=args.projection_dim,
        tau=args.tau,
        tau_m=args.tau_m,
        alpha_top_k=args.alpha_top_k,
        node_link_mode=args.node_link_mode,
        node_weight_mode=args.node_weight_mode,
        taxonomy_edges=args.taxonomy_edges,
        negative_mode=args.negative_mode,
        negative_pool_mode=args.negative_pool_mode,
        untrained_negative_top_m=args.untrained_negative_top_m,
        untrained_node_cache_size=args.untrained_node_cache_size,
        global_negative_sample_size=args.global_negative_sample_size,
        author_node_cache_size=args.author_node_cache_size,
        pi0=args.pi0,
        w_near=args.w_near,
        w_far=args.w_far,
        prox_threshold=args.prox_threshold,
        prox_quantile=args.prox_quantile,
        prox_beta=args.prox_beta,
        prox_bias=args.prox_bias,
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        epochs=args.epochs,
        eval_frac=args.eval_frac,
        seed=args.seed,
        device=args.device,
        max_papers=args.max_papers,
    )

    print("loading_dataset=1", flush=True)
    dataset = Stage1SmokeDataset(
        paths,
        max_papers=args.max_papers,
        node_link_mode=args.node_link_mode,
        node_weight_mode=args.node_weight_mode,
        taxonomy_edges=Path(args.taxonomy_edges),
    )
    print(
        f"tasks={len(dataset.tasks)} authors={len(dataset.author_ids)} "
        f"role_embeddings={dataset.role_embeddings.shape} history_embeddings={dataset.history_embeddings.shape}",
        flush=True,
    )
    summary = train_stage1_smoke(
        dataset=dataset,
        config=config,
        out_dir=Path(args.out_dir),
        citation_graph_dir=Path(args.citation_graph_dir) if args.citation_graph_dir else None,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
