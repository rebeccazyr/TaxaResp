#!/usr/bin/env python3
"""Run Stage-1 training on the canonical KDD role-aware splits."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.stage1_smoke_training import Stage1Dataset, Stage1Paths, TrainConfig, train_stage1_splits


ROLE_SPLIT_DIR = Path("outputs/role_training_splits_author_count3_jsd_l0_ge0p03186_highconf2")
OFFICIAL_TEST_DIR = Path("outputs/role_test_set_2019_author_count3_jsd_l0_ge0p03186_highconf2_histge5")
FOS_MAP = Path("../data/dblp/FieldsOfStudy.txt")
TAXONOMY_EDGES = Path("../data/dblp/13.FieldOfStudyChildren.nt")
VALID_EXPERT_PROFILE_DIR = Path("outputs/expert_profile_cutoffs/pre_2018_for_valid_2018")
TEST_EXPERT_PROFILE_DIR = Path("outputs/expert_profile_cutoffs/pre_2019_for_test_2019_2020")


def canonical_train_paths() -> Stage1Paths:
    return Stage1Paths(
        sample_jsonl=ROLE_SPLIT_DIR / "train_5000.jsonl",
        role_descriptions_jsonl=ROLE_SPLIT_DIR / "train_role_descriptions_llm_gptoss120b/stage1_task_node_role_descriptions.jsonl",
        role_ids_tsv=ROLE_SPLIT_DIR / "train_role_descriptions_llm_gptoss120b/role_description_embedding_ids.tsv",
        role_embeddings_npy=ROLE_SPLIT_DIR / "train_role_descriptions_llm_gptoss120b/role_description_embeddings.npy",
        history_author_papers_tsv=ROLE_SPLIT_DIR / "train_history_paper_embeddings_inputs/author_history_papers.tsv",
        history_ids_tsv=ROLE_SPLIT_DIR / "train_history_paper_embeddings_inputs/history_paper_embedding_ids.tsv",
        history_embeddings_npy=ROLE_SPLIT_DIR / "train_history_paper_embeddings_inputs/history_paper_embeddings.npy",
        history_paper_fos_weights_tsv=ROLE_SPLIT_DIR / "train_history_paper_embeddings_inputs/history_paper_fos_weights.tsv",
        expert_profile_dir=VALID_EXPERT_PROFILE_DIR,
        fos_map_tsv=FOS_MAP,
    )


def canonical_dev_paths() -> Stage1Paths:
    return Stage1Paths(
        sample_jsonl=ROLE_SPLIT_DIR / "test_500.jsonl",
        role_descriptions_jsonl=ROLE_SPLIT_DIR / "test_role_descriptions_llm_gptoss120b/stage1_task_node_role_descriptions.jsonl",
        role_ids_tsv=ROLE_SPLIT_DIR / "test_role_descriptions_llm_gptoss120b/role_description_embedding_ids.tsv",
        role_embeddings_npy=ROLE_SPLIT_DIR / "test_role_descriptions_llm_gptoss120b/role_description_embeddings.npy",
        history_author_papers_tsv=ROLE_SPLIT_DIR / "test_history_paper_embeddings_inputs/author_history_papers.tsv",
        history_ids_tsv=ROLE_SPLIT_DIR / "test_history_paper_embeddings_inputs/history_paper_embedding_ids.tsv",
        history_embeddings_npy=ROLE_SPLIT_DIR / "test_history_paper_embeddings_inputs/history_paper_embeddings.npy",
        history_paper_fos_weights_tsv=ROLE_SPLIT_DIR / "test_history_paper_embeddings_inputs/history_paper_fos_weights.tsv",
        expert_profile_dir=VALID_EXPERT_PROFILE_DIR,
        fos_map_tsv=FOS_MAP,
    )


def canonical_official_test_paths() -> Stage1Paths:
    return Stage1Paths(
        sample_jsonl=OFFICIAL_TEST_DIR / "test_500.jsonl",
        role_descriptions_jsonl=OFFICIAL_TEST_DIR / "test_role_descriptions_llm_gptoss120b/stage1_task_node_role_descriptions.jsonl",
        role_ids_tsv=OFFICIAL_TEST_DIR / "test_role_descriptions_llm_gptoss120b/role_description_embedding_ids.tsv",
        role_embeddings_npy=OFFICIAL_TEST_DIR / "test_role_descriptions_llm_gptoss120b/role_description_embeddings.npy",
        history_author_papers_tsv=OFFICIAL_TEST_DIR / "test_history_embedding_inputs/author_history_papers.tsv",
        history_ids_tsv=OFFICIAL_TEST_DIR / "test_history_embeddings_openai/history_paper_embedding_ids.tsv",
        history_embeddings_npy=OFFICIAL_TEST_DIR / "test_history_embeddings_openai/history_paper_embeddings.npy",
        history_paper_fos_weights_tsv=OFFICIAL_TEST_DIR / "test_history_embedding_inputs/history_paper_fos_weights.tsv",
        expert_profile_dir=TEST_EXPERT_PROFILE_DIR,
        fos_map_tsv=FOS_MAP,
    )


def path_args(prefix: str, defaults: Stage1Paths, parser: argparse.ArgumentParser) -> None:
    parser.add_argument(f"--{prefix}-sample-jsonl", default=str(defaults.sample_jsonl))
    parser.add_argument(f"--{prefix}-role-descriptions-jsonl", default=str(defaults.role_descriptions_jsonl))
    parser.add_argument(f"--{prefix}-role-ids-tsv", default=str(defaults.role_ids_tsv))
    parser.add_argument(f"--{prefix}-role-embeddings", default=str(defaults.role_embeddings_npy))
    parser.add_argument(f"--{prefix}-history-author-papers-tsv", default=str(defaults.history_author_papers_tsv))
    parser.add_argument(f"--{prefix}-history-ids-tsv", default=str(defaults.history_ids_tsv))
    parser.add_argument(f"--{prefix}-history-embeddings", default=str(defaults.history_embeddings_npy))
    parser.add_argument(f"--{prefix}-history-paper-fos-weights-tsv", default=str(defaults.history_paper_fos_weights_tsv))
    parser.add_argument(f"--{prefix}-expert-profile-dir", default=str(defaults.expert_profile_dir))
    parser.add_argument(f"--{prefix}-fos-map-tsv", default=str(defaults.fos_map_tsv))


def paths_from_args(args: argparse.Namespace, prefix: str) -> Stage1Paths:
    return Stage1Paths(
        sample_jsonl=Path(getattr(args, f"{prefix}_sample_jsonl")),
        role_descriptions_jsonl=Path(getattr(args, f"{prefix}_role_descriptions_jsonl")),
        role_ids_tsv=Path(getattr(args, f"{prefix}_role_ids_tsv")),
        role_embeddings_npy=Path(getattr(args, f"{prefix}_role_embeddings")),
        history_author_papers_tsv=Path(getattr(args, f"{prefix}_history_author_papers_tsv")),
        history_ids_tsv=Path(getattr(args, f"{prefix}_history_ids_tsv")),
        history_embeddings_npy=Path(getattr(args, f"{prefix}_history_embeddings")),
        history_paper_fos_weights_tsv=Path(getattr(args, f"{prefix}_history_paper_fos_weights_tsv")),
        expert_profile_dir=Path(getattr(args, f"{prefix}_expert_profile_dir")),
        fos_map_tsv=Path(getattr(args, f"{prefix}_fos_map_tsv")),
    )


def missing_paths(named_paths: dict[str, Stage1Paths]) -> list[str]:
    missing: list[str] = []
    for split_name, paths in named_paths.items():
        for field_name, path in paths.__dict__.items():
            if not Path(path).exists():
                missing.append(f"{split_name}.{field_name}: {path}")
    return missing


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    path_args("train", canonical_train_paths(), parser)
    path_args("dev", canonical_dev_paths(), parser)
    path_args("official-test", canonical_official_test_paths(), parser)
    parser.add_argument("--skip-dev", action="store_true")
    parser.add_argument("--skip-official-test", action="store_true")
    parser.add_argument("--out-dir", default="outputs/stage1_training/canonical_v1_untrained_topm20")
    parser.add_argument("--citation-graph-dir", default="")

    parser.add_argument("--projection-dim", type=int, default=256)
    parser.add_argument("--tau", type=float, default=0.07)
    parser.add_argument("--tau-m", type=float, default=0.1)
    parser.add_argument("--alpha-top-k", type=int, default=16)
    parser.add_argument("--node-link-mode", choices=("direct", "minimal_tree"), default="minimal_tree")
    parser.add_argument("--node-weight-mode", choices=("unweighted", "weighted"), default="unweighted")
    parser.add_argument("--taxonomy-edges", default=str(TAXONOMY_EDGES))
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
    parser.add_argument("--prox-threshold", type=float, default=None)
    parser.add_argument("--prox-quantile", type=float, default=0.75)
    parser.add_argument("--prox-beta", type=float, default=20.0)
    parser.add_argument("--prox-bias", type=float, default=1.0)

    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-train-papers", type=int, default=0)
    parser.add_argument("--max-eval-papers", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_paths = paths_from_args(args, "train")
    eval_paths: dict[str, Stage1Paths] = {}
    if not args.skip_dev:
        eval_paths["dev"] = paths_from_args(args, "dev")
    if not args.skip_official_test:
        eval_paths["official_test"] = paths_from_args(args, "official_test")

    named_paths = {"train": train_paths, **eval_paths}
    missing = missing_paths(named_paths)
    if missing:
        print("missing_required_stage1_artifacts=1", file=sys.stderr)
        for item in missing:
            print(item, file=sys.stderr)
        raise SystemExit(2)

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
        eval_frac=0.0,
        seed=args.seed,
        device=args.device,
        max_papers=args.max_train_papers,
    )

    print("loading_train_dataset=1", flush=True)
    train_dataset = Stage1Dataset(
        train_paths,
        max_papers=args.max_train_papers,
        node_link_mode=args.node_link_mode,
        node_weight_mode=args.node_weight_mode,
        taxonomy_edges=Path(args.taxonomy_edges),
    )
    eval_datasets: dict[str, Stage1Dataset] = {}
    for split_name, paths in eval_paths.items():
        print(f"loading_{split_name}_dataset=1", flush=True)
        eval_datasets[split_name] = Stage1Dataset(
            paths,
            max_papers=args.max_eval_papers,
            node_link_mode=args.node_link_mode,
            node_weight_mode=args.node_weight_mode,
            taxonomy_edges=Path(args.taxonomy_edges),
        )

    print(
        f"train_tasks={len(train_dataset.tasks)} train_authors={len(train_dataset.author_ids)} "
        f"input_dim={train_dataset.input_dim}",
        flush=True,
    )
    for split_name, dataset in eval_datasets.items():
        print(
            f"{split_name}_tasks={len(dataset.tasks)} {split_name}_authors={len(dataset.author_ids)} "
            f"input_dim={dataset.input_dim}",
            flush=True,
        )

    summary = train_stage1_splits(
        train_dataset=train_dataset,
        eval_datasets=eval_datasets,
        config=config,
        out_dir=Path(args.out_dir),
        citation_graph_dir=Path(args.citation_graph_dir) if args.citation_graph_dir else None,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
