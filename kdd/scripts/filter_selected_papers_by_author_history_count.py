#!/usr/bin/env python3
"""Filter selected papers by every groundtruth author's pre-cutoff history count."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--features-tsv", default="")
    parser.add_argument("--dblp-json", default="data/dblp/dblp.v12.json")
    parser.add_argument("--cutoff-year", type=int, default=2019)
    parser.add_argument("--min-history-papers", type=int, default=5)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--progress-every", type=int, default=500000)
    return parser.parse_args()


def iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            text = raw.strip()
            if not text:
                continue
            if text.startswith(","):
                text = text[1:]
            if text.endswith(","):
                text = text[:-1]
            if not text or text in {"[", "]"} or not text.startswith("{"):
                continue
            try:
                yield json.loads(text)
            except json.JSONDecodeError:
                continue


def author_ids(obj: dict) -> list[str]:
    ids: list[str] = []
    for author in obj.get("authors") or []:
        if not isinstance(author, dict):
            continue
        author_id = author.get("id")
        if author_id is not None and str(author_id).strip():
            ids.append(str(author_id).strip())
    return ids


def safe_year(obj: dict) -> int | None:
    try:
        return int(obj.get("year"))
    except (TypeError, ValueError):
        return None


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


def count_history_papers(dblp_path: Path, target_authors: set[str], cutoff_year: int, progress_every: int) -> Counter[str]:
    counts: Counter[str] = Counter()
    scanned = 0
    matched_papers = 0
    for obj in iter_jsonl(dblp_path):
        scanned += 1
        if progress_every > 0 and scanned % progress_every == 0:
            print(
                f"scanned={scanned:,} matched_history_papers={matched_papers:,}",
                flush=True,
            )
        year = safe_year(obj)
        if year is None or year >= cutoff_year:
            continue
        matched = set(author_ids(obj)) & target_authors
        if not matched:
            continue
        matched_papers += 1
        for author_id in matched:
            counts[author_id] += 1
    print(f"scanned={scanned:,} matched_history_papers={matched_papers:,}", flush=True)
    return counts


def write_author_counts(path: Path, target_authors: set[str], counts: Counter[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["author_id", "history_papers_pre_cutoff"])
        for author_id in sorted(target_authors, key=lambda x: (not x.isdigit(), int(x) if x.isdigit() else x)):
            writer.writerow([author_id, counts.get(author_id, 0)])


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_jsonl)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    features = load_features(Path(args.features_tsv) if args.features_tsv else None)

    papers = list(iter_jsonl(input_path))
    target_authors = {author_id for obj in papers for author_id in author_ids(obj)}
    if not target_authors:
        raise SystemExit("No authors found in input JSONL")

    counts = count_history_papers(
        Path(args.dblp_json),
        target_authors,
        args.cutoff_year,
        args.progress_every,
    )
    write_author_counts(out_dir / "author_history_counts.tsv", target_authors, counts)

    selected_objects: list[dict] = []
    selected_rows: list[dict[str, object]] = []
    before_author_count_dist: Counter[int] = Counter()
    after_author_count_dist: Counter[int] = Counter()
    min_history_before: Counter[int] = Counter()
    min_history_after: Counter[int] = Counter()
    before_author_appearances = 0
    after_author_appearances = 0
    before_unique_authors: set[str] = set()
    after_unique_authors: set[str] = set()

    for obj in papers:
        paper_id = str(obj.get("id") or "").strip()
        ids = author_ids(obj)
        histories = [counts.get(author_id, 0) for author_id in ids]
        author_count = len(ids)
        min_history = min(histories) if histories else 0
        mean_history = sum(histories) / len(histories) if histories else 0.0
        before_author_count_dist[author_count] += 1
        min_history_before[min(min_history, args.min_history_papers)] += 1
        before_author_appearances += author_count
        before_unique_authors.update(ids)

        keep = bool(ids) and all(value >= args.min_history_papers for value in histories)
        if not keep:
            continue

        after_author_count_dist[author_count] += 1
        min_history_after[min(min_history, args.min_history_papers)] += 1
        after_author_appearances += author_count
        after_unique_authors.update(ids)

        obj_out = dict(obj)
        filter_features = dict(obj_out.get("_filter_features") or {})
        filter_features.update(features.get(paper_id, {}))
        filter_features.update(
            {
                "min_author_history_papers_pre_cutoff": min_history,
                "mean_author_history_papers_pre_cutoff": mean_history,
                "history_cutoff_year": args.cutoff_year,
                "min_required_author_history_papers": args.min_history_papers,
            }
        )
        obj_out["_filter_features"] = filter_features
        selected_objects.append(obj_out)
        selected_rows.append(
            {
                "paper_id": paper_id,
                "title": str(obj.get("title") or ""),
                "author_count": author_count,
                "author_jsd_l0_mean": filter_features.get("author_jsd_l0_mean", ""),
                "high_conf_direct_node_count": filter_features.get("high_conf_direct_node_count", ""),
                "min_author_history_papers_pre_cutoff": min_history,
                "mean_author_history_papers_pre_cutoff": f"{mean_history:.6f}",
            }
        )

    with (out_dir / "selected_papers.jsonl").open("w", encoding="utf-8") as f:
        for obj in selected_objects:
            f.write(json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n")

    fieldnames = [
        "paper_id",
        "title",
        "author_count",
        "author_jsd_l0_mean",
        "high_conf_direct_node_count",
        "min_author_history_papers_pre_cutoff",
        "mean_author_history_papers_pre_cutoff",
    ]
    with (out_dir / "selected_papers.tsv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(selected_rows)

    with (out_dir / "author_count_distribution.tsv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["author_count", "before_papers", "after_papers"])
        for count in sorted(set(before_author_count_dist) | set(after_author_count_dist)):
            writer.writerow([count, before_author_count_dist[count], after_author_count_dist[count]])

    with (out_dir / "min_author_history_distribution.tsv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["min_author_history_papers_pre_cutoff_bucket", "before_papers", "after_papers"])
        for count in range(args.min_history_papers + 1):
            label = f">={args.min_history_papers}" if count == args.min_history_papers else str(count)
            writer.writerow([label, min_history_before[count], min_history_after[count]])

    summary = {
        "input_jsonl": str(input_path),
        "features_tsv": args.features_tsv,
        "dblp_json": args.dblp_json,
        "cutoff_year": args.cutoff_year,
        "min_history_papers": args.min_history_papers,
        "papers_before": len(papers),
        "papers_after": len(selected_objects),
        "paper_keep_rate": len(selected_objects) / len(papers) if papers else 0.0,
        "unique_authors_before": len(before_unique_authors),
        "unique_authors_after": len(after_unique_authors),
        "author_appearances_before": before_author_appearances,
        "author_appearances_after": after_author_appearances,
        "mean_authors_per_paper_before": before_author_appearances / len(papers) if papers else 0.0,
        "mean_authors_per_paper_after": after_author_appearances / len(selected_objects) if selected_objects else 0.0,
        "author_history_counts_tsv": str(out_dir / "author_history_counts.tsv"),
        "selected_jsonl": str(out_dir / "selected_papers.jsonl"),
        "selected_tsv": str(out_dir / "selected_papers.tsv"),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
