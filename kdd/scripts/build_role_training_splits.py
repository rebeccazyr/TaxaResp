#!/usr/bin/env python3
"""Build train/test splits from the role-aware selected-paper pool."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Sequence


DEFAULT_INPUT_JSONL = (
    "outputs/cross_domain_eval_selection/"
    "author_count3_jsd_l0_ge0p03186_highconf2/selected_papers.jsonl"
)
DEFAULT_OUT_DIR = "outputs/role_training_splits_author_count3_jsd_l0_ge0p03186_highconf2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", default=DEFAULT_INPUT_JSONL)
    parser.add_argument(
        "--features-tsv",
        default="",
        help="Optional TSV keyed by paper_id with filter features to embed in output records.",
    )
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--train-size", type=int, default=5000)
    parser.add_argument("--test-size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=13)
    return parser.parse_args()


def iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def reconstruct_abstract(indexed_abstract: object) -> str:
    if not isinstance(indexed_abstract, dict):
        return ""
    inverted = indexed_abstract.get("InvertedIndex")
    if not isinstance(inverted, dict):
        return ""
    try:
        total = int(indexed_abstract.get("IndexLength") or 0)
    except (TypeError, ValueError):
        total = 0
    positions: dict[int, str] = {}
    for token, raw_positions in inverted.items():
        if not isinstance(raw_positions, list):
            continue
        for raw_idx in raw_positions:
            try:
                idx = int(raw_idx)
            except (TypeError, ValueError):
                continue
            positions[idx] = str(token)
    if total <= 0:
        total = max(positions, default=-1) + 1
    return " ".join(positions.get(idx, "") for idx in range(total)).strip()


def author_count_bucket(author_count: int) -> str:
    return "8+" if author_count >= 8 else str(author_count)


def author_rows(obj: dict) -> list[dict[str, str]]:
    rows = []
    for author in obj.get("authors") or []:
        if not isinstance(author, dict):
            continue
        author_id = str(author.get("id") or "").strip()
        if not author_id:
            continue
        rows.append({"author_id": author_id, "name": str(author.get("name") or "").strip()})
    return rows


def load_features(path: Path | None) -> dict[str, dict[str, object]]:
    if path is None:
        return {}
    features: dict[str, dict[str, object]] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            paper_id = str(row.get("paper_id") or "").strip()
            if not paper_id:
                continue
            parsed: dict[str, object] = {}
            for key, value in row.items():
                if key in {"paper_id", "title"}:
                    continue
                try:
                    parsed[key] = float(value)
                except (TypeError, ValueError):
                    parsed[key] = value
            features[paper_id] = parsed
    return features


def compact_record(obj: dict, external_features: dict[str, dict[str, object]] | None = None) -> dict:
    paper_id = str(obj.get("id") or "").strip()
    authors = author_rows(obj)
    filter_features = dict(obj.get("_filter_features") or {})
    if external_features and paper_id in external_features:
        filter_features.update(external_features[paper_id])
    abstract = reconstruct_abstract(obj.get("indexed_abstract"))
    fos_items = [
        {
            "name": str(item.get("name") or "").strip(),
            "weight": float(item.get("w") or 0.0),
        }
        for item in obj.get("fos") or []
        if isinstance(item, dict) and str(item.get("name") or "").strip()
    ]
    fos_items.sort(key=lambda item: (-item["weight"], item["name"]))
    venue = obj.get("venue") or {}
    return {
        "paper_id": paper_id,
        "title": str(obj.get("title") or "").strip(),
        "abstract": abstract,
        "year": obj.get("year"),
        "venue": venue.get("raw", "") if isinstance(venue, dict) else "",
        "team_size": len(authors),
        "team_members": authors,
        "team_author_ids": [row["author_id"] for row in authors],
        "team_author_names": [row["name"] for row in authors],
        "fos": fos_items,
        "filter_features": filter_features,
        "author_count_bucket": author_count_bucket(len(authors)),
    }


def load_records(path: Path, features: dict[str, dict[str, object]] | None = None) -> list[dict]:
    seen = set()
    records = []
    for obj in iter_jsonl(path):
        row = compact_record(obj, features)
        if not row["paper_id"] or row["paper_id"] in seen:
            continue
        seen.add(row["paper_id"])
        records.append(row)
    return records


def largest_remainder_quotas(counts: Counter[str], size: int) -> dict[str, int]:
    total = sum(counts.values())
    if size > total:
        raise ValueError(f"requested size {size} exceeds available records {total}")
    exact = {bucket: size * count / total for bucket, count in counts.items()}
    quotas = {bucket: int(value) for bucket, value in exact.items()}
    remaining = size - sum(quotas.values())
    buckets = sorted(
        counts,
        key=lambda bucket: (exact[bucket] - quotas[bucket], counts[bucket], bucket),
        reverse=True,
    )
    for bucket in buckets[:remaining]:
        quotas[bucket] += 1
    return quotas


def stratified_take(
    by_bucket: dict[str, list[dict]],
    quotas: dict[str, int],
) -> list[dict]:
    selected = []
    for bucket in sorted(quotas):
        quota = quotas[bucket]
        if quota > len(by_bucket[bucket]):
            raise ValueError(f"bucket {bucket} has {len(by_bucket[bucket])} records, needs {quota}")
        selected.extend(by_bucket[bucket][:quota])
        del by_bucket[bucket][:quota]
    return selected


def split_records(records: Sequence[dict], train_size: int, test_size: int, seed: int) -> tuple[list[dict], list[dict]]:
    rng = random.Random(seed)
    by_bucket: dict[str, list[dict]] = defaultdict(list)
    for row in records:
        by_bucket[row["author_count_bucket"]].append(row)
    for rows in by_bucket.values():
        rng.shuffle(rows)

    counts = Counter(row["author_count_bucket"] for row in records)
    test_quotas = largest_remainder_quotas(counts, test_size)
    train_quotas = largest_remainder_quotas(counts, train_size)
    test = stratified_take(by_bucket, test_quotas)
    train = stratified_take(by_bucket, train_quotas)
    rng.shuffle(train)
    rng.shuffle(test)
    return train, test


def top_fos_text(row: dict, limit: int = 12) -> str:
    return "|".join(f"{item['name']}:{item['weight']:.5f}" for item in row["fos"][:limit])


def write_jsonl(path: Path, rows: Sequence[dict], split: str) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            obj = dict(row)
            obj["split"] = split
            f.write(json.dumps(obj, ensure_ascii=False, separators=(",", ":")))
            f.write("\n")


def write_ids(path: Path, rows: Sequence[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("paper_id\n")
        for row in rows:
            f.write(f"{row['paper_id']}\n")


def write_tsv(path: Path, rows: Sequence[dict], split: str) -> None:
    fieldnames = [
        "split",
        "paper_id",
        "title",
        "year",
        "venue",
        "team_size",
        "team_author_ids",
        "team_author_names",
        "author_jsd_l0_mean",
        "high_conf_direct_node_count",
        "top_fos",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            features = row.get("filter_features") or {}
            writer.writerow(
                {
                    "split": split,
                    "paper_id": row["paper_id"],
                    "title": row["title"],
                    "year": row["year"],
                    "venue": row["venue"],
                    "team_size": row["team_size"],
                    "team_author_ids": "|".join(row["team_author_ids"]),
                    "team_author_names": "|".join(row["team_author_names"]),
                    "author_jsd_l0_mean": features.get("author_jsd_l0_mean", ""),
                    "high_conf_direct_node_count": features.get("high_conf_direct_node_count", ""),
                    "top_fos": top_fos_text(row),
                }
            )


def summary_row(split: str, rows: Sequence[dict]) -> dict:
    author_ids = {author_id for row in rows for author_id in row["team_author_ids"]}
    appearances = sum(row["team_size"] for row in rows)
    return {
        "split": split,
        "papers": len(rows),
        "author_appearances": appearances,
        "unique_authors": len(author_ids),
        "mean_authors_per_paper": f"{(appearances / len(rows)) if rows else 0.0:.6f}",
        "author_count_distribution": json.dumps(
            dict(sorted(Counter(row["author_count_bucket"] for row in rows).items())),
            sort_keys=True,
        ),
    }


def write_summary(path: Path, rows: Sequence[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def output_stem(split: str, size: int) -> str:
    return f"{split}_{size}"


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_jsonl)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    features_path = Path(args.features_tsv) if args.features_tsv else None
    features = load_features(features_path)
    records = load_records(input_path, features)
    if args.train_size + args.test_size > len(records):
        raise ValueError(
            f"requested train+test={args.train_size + args.test_size}, "
            f"but only {len(records)} records are available"
        )
    train, test = split_records(records, args.train_size, args.test_size, args.seed)
    overlap = {row["paper_id"] for row in train} & {row["paper_id"] for row in test}
    if overlap:
        raise AssertionError(f"train/test overlap: {sorted(overlap)[:5]}")

    files = []
    if args.train_size > 0:
        train_stem = output_stem("train", args.train_size)
        write_jsonl(out_dir / f"{train_stem}.jsonl", train, "train")
        write_tsv(out_dir / f"{train_stem}.tsv", train, "train")
        write_ids(out_dir / f"{train_stem}_ids.tsv", train)
        files.extend([f"{train_stem}.jsonl", f"{train_stem}.tsv", f"{train_stem}_ids.tsv"])

    test_stem = output_stem("test", args.test_size)
    write_jsonl(out_dir / f"{test_stem}.jsonl", test, "test")
    write_tsv(out_dir / f"{test_stem}.tsv", test, "test")
    write_ids(out_dir / f"{test_stem}_ids.tsv", test)
    files.extend([f"{test_stem}.jsonl", f"{test_stem}.tsv", f"{test_stem}_ids.tsv"])

    summary_rows = [
        summary_row("source_pool", records),
        summary_row("test", test),
    ]
    if args.train_size > 0:
        summary_rows.insert(1, summary_row("train", train))
    write_summary(out_dir / "summary.tsv", summary_rows)
    files.append("summary.tsv")
    manifest = {
        "input_jsonl": str(input_path),
        "features_tsv": str(features_path) if features_path else "",
        "out_dir": str(out_dir),
        "seed": args.seed,
        "train_size": args.train_size,
        "test_size": args.test_size,
        "paper_disjoint": True,
        "author_disjoint": False,
        "team_source": "groundtruth paper authors",
        "filter": "author_count>=3 AND author_jsd_l0_mean>=0.03186 AND high_conf_direct_node_count>=2",
        "files": files,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"records={len(records)} train={len(train)} test={len(test)} out_dir={out_dir}")


if __name__ == "__main__":
    main()
