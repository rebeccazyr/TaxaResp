#!/usr/bin/env python3
"""Prepare small leakage-free Stage 1 task samples from filtered validation data.

The output samples are meant for the first role-expert alignment training runs:
they keep papers whose authors all have enough pre-cutoff history, then stratify
by team size and citation-community dispersion so the pilot data contains both
homogeneous and cross-community teams.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


DEFAULT_AUTHOR_BUCKET_WEIGHTS = {
    "2": 0.25,
    "3": 0.30,
    "4": 0.25,
    "5-6": 0.20,
}

DEFAULT_BLOCK_BUCKET_WEIGHTS = {
    "1": 0.10,
    "2": 0.30,
    "3": 0.35,
    "4+": 0.25,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample small Stage 1 pilot datasets from the filtered 2018 validation split."
    )
    parser.add_argument(
        "--validation-jsonl",
        default="outputs/temporal_task_splits_full/validation_2018_all_authors_hist_ge5.jsonl",
        help="Filtered validation JSONL. Defaults to the all-authors-history>=5 subset.",
    )
    parser.add_argument(
        "--membership-tsv",
        default="outputs/validation_author_citation_louvain_blocks/membership_resolution_20.tsv",
        help="Author-to-interpretable-citation-community membership TSV. Default is r=20.",
    )
    parser.add_argument(
        "--stratify-membership-tsv",
        default="outputs/validation_author_citation_louvain_blocks_refine_gamma1/membership_resolution_1p1.tsv",
        help=(
            "Author-to-community TSV used only for dispersion stratification. "
            "Default is coarse modularity-best r=1.1; use --membership-tsv value "
            "here if you want stratification by interpretable r=20 communities."
        ),
    )
    parser.add_argument(
        "--nodes-tsv",
        default="outputs/expert_citation_graph_valid2018_pre2018/nodes.tsv",
        help="Citation graph nodes TSV containing historical_papers_pre_cutoff.",
    )
    parser.add_argument(
        "--out-dir",
        default="outputs/stage1_pilot_samples",
        help="Output directory.",
    )
    parser.add_argument(
        "--sizes",
        default="smoke_200:200,pilot_1000:1000,dev_5000:5000",
        help="Comma-separated LABEL:SIZE sample definitions.",
    )
    parser.add_argument("--seed", type=int, default=13, help="Random seed.")
    parser.add_argument("--min-authors", type=int, default=2)
    parser.add_argument("--max-authors", type=int, default=6)
    parser.add_argument(
        "--require-abstract",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require a reconstructable indexed abstract.",
    )
    parser.add_argument(
        "--require-fos",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require non-empty FoS annotations.",
    )
    return parser.parse_args()


def iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def parse_sample_sizes(value: str) -> List[Tuple[str, int]]:
    result: List[Tuple[str, int]] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"sample size must be LABEL:SIZE, got {item!r}")
        label, raw_size = item.split(":", 1)
        label = label.strip()
        size = int(raw_size)
        if not label or size <= 0:
            raise ValueError(f"invalid sample definition: {item!r}")
        result.append((label, size))
    if not result:
        raise ValueError("--sizes must define at least one sample")
    return result


def load_membership(path: Path) -> Dict[str, str]:
    membership: Dict[str, str] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            author_id = str(row.get("author_id", "")).strip()
            block_id = str(row.get("block_id", "")).strip()
            if author_id and block_id:
                membership[author_id] = block_id
    return membership


def load_history_counts(path: Path) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            author_id = str(row.get("author_id", "")).strip()
            if not author_id:
                continue
            try:
                counts[author_id] = int(float(row.get("historical_papers_pre_cutoff", 0)))
            except (TypeError, ValueError):
                counts[author_id] = 0
    return counts


def ordered_authors(obj: dict) -> List[dict]:
    authors = obj.get("authors") or []
    if not isinstance(authors, list):
        return []
    seen = set()
    result: List[dict] = []
    for author in authors:
        if not isinstance(author, dict):
            continue
        author_id = str(author.get("id", "")).strip()
        if not author_id or author_id in seen:
            continue
        seen.add(author_id)
        result.append(author)
    return result


def reconstruct_abstract(indexed_abstract: object) -> str:
    if not isinstance(indexed_abstract, dict):
        return ""
    inverted = indexed_abstract.get("InvertedIndex")
    if not isinstance(inverted, dict):
        return ""
    length = indexed_abstract.get("IndexLength")
    try:
        total = int(length)
    except (TypeError, ValueError):
        total = 0
    positions: Dict[int, str] = {}
    for token, raw_indices in inverted.items():
        if not isinstance(raw_indices, list):
            continue
        for raw_idx in raw_indices:
            try:
                idx = int(raw_idx)
            except (TypeError, ValueError):
                continue
            positions[idx] = str(token)
    if total <= 0:
        total = max(positions, default=-1) + 1
    return " ".join(positions.get(i, "") for i in range(total)).strip()


def author_bucket(author_count: int) -> str:
    if author_count <= 4:
        return str(author_count)
    return "5-6"


def block_bucket(distinct_blocks: int) -> str:
    if distinct_blocks <= 3:
        return str(distinct_blocks)
    return "4+"


def weighted_mean(values: Sequence[int]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def paper_metadata(
    obj: dict,
    membership: Dict[str, str],
    stratify_membership: Dict[str, str],
    history_counts: Dict[str, int],
    abstract: str,
) -> dict:
    authors = ordered_authors(obj)
    author_ids = [str(author.get("id", "")).strip() for author in authors]
    block_ids = [membership[author_id] for author_id in author_ids]
    stratify_block_ids = [stratify_membership[author_id] for author_id in author_ids]
    history = [history_counts[author_id] for author_id in author_ids]
    fos_items = obj.get("fos") or []
    fos_names = [
        str(item.get("name", "")).strip()
        for item in fos_items
        if isinstance(item, dict) and str(item.get("name", "")).strip()
    ]
    return {
        "paper_id": str(obj.get("id", "")).strip(),
        "title": str(obj.get("title", "")).strip(),
        "year": obj.get("year"),
        "venue": (obj.get("venue") or {}).get("raw", "") if isinstance(obj.get("venue"), dict) else "",
        "author_count": len(author_ids),
        "author_ids": author_ids,
        "author_names": [str(author.get("name", "")).strip() for author in authors],
        "author_history_papers_pre_2018": history,
        "min_author_history_papers_pre_2018": min(history) if history else 0,
        "mean_author_history_papers_pre_2018": weighted_mean(history),
        "block_ids_r20": block_ids,
        "distinct_blocks_r20": len(set(block_ids)),
        "stratify_block_ids": stratify_block_ids,
        "distinct_stratify_blocks": len(set(stratify_block_ids)),
        "author_bucket": author_bucket(len(author_ids)),
        "block_bucket": block_bucket(len(set(stratify_block_ids))),
        "fos_names": fos_names,
        "fos_count": len(fos_names),
        "abstract": abstract,
    }


def eligible_records(
    validation_path: Path,
    membership: Dict[str, str],
    stratify_membership: Dict[str, str],
    history_counts: Dict[str, int],
    min_authors: int,
    max_authors: int,
    require_abstract: bool,
    require_fos: bool,
) -> Tuple[List[dict], Counter]:
    records: List[dict] = []
    skip_reasons: Counter = Counter()
    seen_papers = set()

    for obj in iter_jsonl(validation_path):
        paper_id = str(obj.get("id", "")).strip()
        if not paper_id or paper_id in seen_papers:
            skip_reasons["duplicate_or_missing_paper_id"] += 1
            continue
        seen_papers.add(paper_id)

        authors = ordered_authors(obj)
        author_ids = [str(author.get("id", "")).strip() for author in authors]
        if not (min_authors <= len(author_ids) <= max_authors):
            skip_reasons["author_count_out_of_range"] += 1
            continue
        if any(author_id not in membership for author_id in author_ids):
            skip_reasons["missing_block_membership"] += 1
            continue
        if any(author_id not in stratify_membership for author_id in author_ids):
            skip_reasons["missing_stratify_block_membership"] += 1
            continue
        if any(author_id not in history_counts for author_id in author_ids):
            skip_reasons["missing_history_count"] += 1
            continue

        abstract = reconstruct_abstract(obj.get("indexed_abstract"))
        if require_abstract and not abstract:
            skip_reasons["missing_abstract"] += 1
            continue
        fos_items = obj.get("fos") or []
        if require_fos and not fos_items:
            skip_reasons["missing_fos"] += 1
            continue

        meta = paper_metadata(obj, membership, stratify_membership, history_counts, abstract)
        if meta["author_bucket"] not in DEFAULT_AUTHOR_BUCKET_WEIGHTS:
            skip_reasons["unsupported_author_bucket"] += 1
            continue
        row = {"record": obj, "metadata": meta}
        records.append(row)

    return records, skip_reasons


def target_cell_counts(size: int) -> Dict[Tuple[str, str], int]:
    exact: Dict[Tuple[str, str], float] = {}
    floors: Dict[Tuple[str, str], int] = {}
    for author_key, author_weight in DEFAULT_AUTHOR_BUCKET_WEIGHTS.items():
        for block_key, block_weight in DEFAULT_BLOCK_BUCKET_WEIGHTS.items():
            value = size * author_weight * block_weight
            exact[(author_key, block_key)] = value
            floors[(author_key, block_key)] = int(value)
    remaining = size - sum(floors.values())
    by_fraction = sorted(
        exact,
        key=lambda key: (exact[key] - floors[key], exact[key]),
        reverse=True,
    )
    for key in by_fraction[:remaining]:
        floors[key] += 1
    return floors


def stratified_sample(records: Sequence[dict], size: int, seed: int) -> List[dict]:
    rng = random.Random(seed)
    by_cell: Dict[Tuple[str, str], List[dict]] = defaultdict(list)
    for row in records:
        meta = row["metadata"]
        by_cell[(meta["author_bucket"], meta["block_bucket"])].append(row)
    for cell_rows in by_cell.values():
        rng.shuffle(cell_rows)

    selected_ids = set()
    selected: List[dict] = []
    targets = target_cell_counts(size)
    for cell, target in targets.items():
        for row in by_cell.get(cell, [])[:target]:
            paper_id = row["metadata"]["paper_id"]
            selected_ids.add(paper_id)
            selected.append(row)

    if len(selected) < size:
        remainder = [
            row for row in records if row["metadata"]["paper_id"] not in selected_ids
        ]
        rng.shuffle(remainder)
        needed = size - len(selected)
        selected.extend(remainder[:needed])

    rng.shuffle(selected)
    return selected[:size]


def write_jsonl(path: Path, rows: Sequence[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            obj = dict(row["record"])
            obj["stage1_sample"] = row["metadata"]
            f.write(json.dumps(obj, ensure_ascii=False, separators=(",", ":")))
            f.write("\n")


def write_llm_inputs(path: Path, rows: Sequence[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            meta = row["metadata"]
            obj = {
                "paper_id": meta["paper_id"],
                "title": meta["title"],
                "year": meta["year"],
                "venue": meta["venue"],
                "abstract": meta["abstract"],
                "fos_names": meta["fos_names"],
                "authors": [
                    {
                        "author_id": author_id,
                        "name": name,
                "history_papers_pre_2018": history,
                "citation_block_r20": block_id,
                "stratify_block": stratify_block_id,
            }
                    for author_id, name, history, block_id, stratify_block_id in zip(
                        meta["author_ids"],
                        meta["author_names"],
                        meta["author_history_papers_pre_2018"],
                        meta["block_ids_r20"],
                        meta["stratify_block_ids"],
                    )
                ],
                "author_count": meta["author_count"],
                "distinct_blocks_r20": meta["distinct_blocks_r20"],
                "distinct_stratify_blocks": meta["distinct_stratify_blocks"],
            }
            f.write(json.dumps(obj, ensure_ascii=False, separators=(",", ":")))
            f.write("\n")


def write_ids_tsv(path: Path, rows: Sequence[dict]) -> None:
    fieldnames = [
        "paper_id",
        "title",
        "year",
        "author_count",
        "distinct_blocks_r20",
        "block_ids_r20",
        "distinct_stratify_blocks",
        "stratify_block_ids",
        "min_author_history_papers_pre_2018",
        "mean_author_history_papers_pre_2018",
        "fos_count",
        "author_bucket",
        "block_bucket",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            meta = row["metadata"]
            writer.writerow(
                {
                    "paper_id": meta["paper_id"],
                    "title": meta["title"],
                    "year": meta["year"],
                    "author_count": meta["author_count"],
                    "distinct_blocks_r20": meta["distinct_blocks_r20"],
                    "block_ids_r20": ",".join(meta["block_ids_r20"]),
                    "distinct_stratify_blocks": meta["distinct_stratify_blocks"],
                    "stratify_block_ids": ",".join(meta["stratify_block_ids"]),
                    "min_author_history_papers_pre_2018": meta[
                        "min_author_history_papers_pre_2018"
                    ],
                    "mean_author_history_papers_pre_2018": (
                        f"{meta['mean_author_history_papers_pre_2018']:.6f}"
                    ),
                    "fos_count": meta["fos_count"],
                    "author_bucket": meta["author_bucket"],
                    "block_bucket": meta["block_bucket"],
                }
            )


def summarize_rows(label: str, rows: Sequence[dict]) -> List[dict]:
    author_counts = Counter(row["metadata"]["author_bucket"] for row in rows)
    block_counts = Counter(row["metadata"]["block_bucket"] for row in rows)
    cells = Counter(
        (row["metadata"]["author_bucket"], row["metadata"]["block_bucket"])
        for row in rows
    )
    summary = [
        {"sample": label, "kind": "total", "bucket": "all", "count": len(rows)}
    ]
    for bucket in sorted(author_counts):
        summary.append(
            {
                "sample": label,
                "kind": "author_bucket",
                "bucket": bucket,
                "count": author_counts[bucket],
            }
        )
    for bucket in sorted(block_counts):
        summary.append(
            {
                "sample": label,
                "kind": "block_bucket",
                "bucket": bucket,
                "count": block_counts[bucket],
            }
        )
    for author_key in DEFAULT_AUTHOR_BUCKET_WEIGHTS:
        for block_key in DEFAULT_BLOCK_BUCKET_WEIGHTS:
            summary.append(
                {
                    "sample": label,
                    "kind": "cell",
                    "bucket": f"authors={author_key};blocks={block_key}",
                    "count": cells[(author_key, block_key)],
                }
            )
    return summary


def write_summary(path: Path, rows: Sequence[dict]) -> None:
    fieldnames = ["sample", "kind", "bucket", "count"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    validation_path = Path(args.validation_jsonl)
    membership_path = Path(args.membership_tsv)
    nodes_path = Path(args.nodes_tsv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    membership = load_membership(membership_path)
    stratify_membership_path = Path(args.stratify_membership_tsv)
    stratify_membership = load_membership(stratify_membership_path)
    history_counts = load_history_counts(nodes_path)
    records, skip_reasons = eligible_records(
        validation_path=validation_path,
        membership=membership,
        stratify_membership=stratify_membership,
        history_counts=history_counts,
        min_authors=args.min_authors,
        max_authors=args.max_authors,
        require_abstract=args.require_abstract,
        require_fos=args.require_fos,
    )
    sample_sizes = parse_sample_sizes(args.sizes)

    all_summary: List[dict] = summarize_rows("eligible_pool", records)
    all_summary.extend(
        {
            "sample": "eligible_filter",
            "kind": "skip_reason",
            "bucket": reason,
            "count": count,
        }
        for reason, count in sorted(skip_reasons.items())
    )

    manifest = {
        "validation_jsonl": str(validation_path),
        "membership_tsv": str(membership_path),
        "stratify_membership_tsv": str(stratify_membership_path),
        "nodes_tsv": str(nodes_path),
        "seed": args.seed,
        "min_authors": args.min_authors,
        "max_authors": args.max_authors,
        "require_abstract": args.require_abstract,
        "require_fos": args.require_fos,
        "eligible_records": len(records),
        "skip_reasons": dict(skip_reasons),
        "samples": {},
    }

    if not records:
        raise RuntimeError("no eligible records found")

    for idx, (label, size) in enumerate(sample_sizes):
        if size > len(records):
            raise ValueError(
                f"requested sample {label} size {size} exceeds eligible pool {len(records)}"
            )
        rows = stratified_sample(records, size=size, seed=args.seed + idx)
        jsonl_path = out_dir / f"{label}.jsonl"
        llm_path = out_dir / f"{label}_llm_inputs.jsonl"
        ids_path = out_dir / f"{label}_ids.tsv"
        write_jsonl(jsonl_path, rows)
        write_llm_inputs(llm_path, rows)
        write_ids_tsv(ids_path, rows)
        all_summary.extend(summarize_rows(label, rows))
        manifest["samples"][label] = {
            "size": len(rows),
            "jsonl": str(jsonl_path),
            "llm_inputs_jsonl": str(llm_path),
            "ids_tsv": str(ids_path),
        }

    summary_path = out_dir / "sample_summary.tsv"
    manifest_path = out_dir / "manifest.json"
    write_summary(summary_path, all_summary)
    manifest["summary_tsv"] = str(summary_path)
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(f"eligible_records={len(records)}")
    print(f"out_dir={out_dir}")
    print(f"summary_tsv={summary_path}")
    print(f"manifest_json={manifest_path}")


if __name__ == "__main__":
    main()
