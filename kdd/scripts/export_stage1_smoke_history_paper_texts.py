#!/usr/bin/env python3
"""Export unique pre-cutoff history paper texts for sampled task authors."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List


DEFAULT_SAMPLE_JSONL = "outputs/stage1_pilot_samples/smoke_200.jsonl"
DEFAULT_DBLP_JSON = "data/dblp/dblp.v12.json"
DEFAULT_OUT_DIR = "outputs/stage1_smoke_embedding_inputs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-jsonl", default=DEFAULT_SAMPLE_JSONL)
    parser.add_argument("--dblp-json", default=DEFAULT_DBLP_JSON)
    parser.add_argument("--cutoff-year", type=int, default=2018)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--progress-every",
        type=int,
        default=500_000,
        help="Print DBLP scan progress every N records; use 0 to disable.",
    )
    return parser.parse_args()


def iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith(","):
                line = line[1:]
            if line.endswith(","):
                line = line[:-1]
            if not line or line in {"[", "]"}:
                continue
            if line:
                yield json.loads(line)


def reconstruct_abstract(indexed_abstract: object) -> str:
    if not isinstance(indexed_abstract, dict):
        return ""
    inverted = indexed_abstract.get("InvertedIndex")
    if not isinstance(inverted, dict):
        return ""
    try:
        length = int(indexed_abstract.get("IndexLength") or 0)
    except (TypeError, ValueError):
        length = 0
    positions: Dict[int, str] = {}
    for token, raw_indices in inverted.items():
        if not isinstance(raw_indices, list):
            continue
        for raw_idx in raw_indices:
            try:
                positions[int(raw_idx)] = str(token)
            except (TypeError, ValueError):
                continue
    if length <= 0:
        length = max(positions, default=-1) + 1
    return " ".join(positions.get(idx, "") for idx in range(length)).strip()


def extract_author_ids(row: dict) -> List[str]:
    if isinstance(row.get("team_author_ids"), list):
        return [str(author_id) for author_id in row["team_author_ids"] if str(author_id).strip()]

    ids: List[str] = []
    for author in row.get("team_members") or []:
        if not isinstance(author, dict):
            continue
        author_id = author.get("author_id") or author.get("id")
        if author_id is not None:
            ids.append(str(author_id))
    if ids:
        return ids

    for author in row.get("authors") or []:
        if not isinstance(author, dict):
            continue
        author_id = author.get("id")
        if author_id is not None:
            ids.append(str(author_id))
    return ids


def load_smoke_authors(path: Path) -> set[str]:
    author_ids: set[str] = set()
    for row in iter_jsonl(path):
        stage = row.get("stage1_sample") or {}
        if isinstance(stage.get("author_ids"), list):
            author_ids.update(str(x) for x in stage["author_ids"])
        else:
            author_ids.update(extract_author_ids(row))
    return author_ids


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            f.write("\n")


def main() -> None:
    args = parse_args()
    sample_path = Path(args.sample_jsonl)
    dblp_path = Path(args.dblp_json)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    smoke_authors = load_smoke_authors(sample_path)
    if not smoke_authors:
        raise SystemExit("No smoke authors found")

    history_rows: List[dict] = []
    author_papers: dict[str, list[str]] = defaultdict(list)
    author_names: dict[str, str] = {}
    skipped_no_text = 0
    scanned = 0

    for row in iter_jsonl(dblp_path):
        scanned += 1
        if args.progress_every > 0 and scanned % args.progress_every == 0:
            print(
                f"scanned_records={scanned} matched_history_papers={len(history_rows)}",
                file=sys.stderr,
                flush=True,
            )
        try:
            year = int(row.get("year"))
        except (TypeError, ValueError):
            continue
        if year >= args.cutoff_year:
            continue
        paper_id = row.get("id")
        if paper_id is None:
            continue
        all_author_ids = extract_author_ids(row)
        matched = sorted(set(all_author_ids) & smoke_authors)
        if not matched:
            continue

        title = str(row.get("title") or "").strip()
        abstract = reconstruct_abstract(row.get("indexed_abstract"))
        text = "\n".join(part for part in (title, abstract) if part).strip()
        if not text:
            skipped_no_text += 1
            continue

        paper_id_str = str(paper_id)
        for author in row.get("authors") or []:
            if not isinstance(author, dict):
                continue
            author_id = author.get("id")
            if author_id is None:
                continue
            author_id_str = str(author_id)
            if author_id_str in smoke_authors:
                author_names.setdefault(author_id_str, str(author.get("name") or ""))
        for author_id in matched:
            author_papers[author_id].append(paper_id_str)

        history_rows.append(
            {
                "id": paper_id_str,
                "paper_id": paper_id_str,
                "title": title,
                "abstract": abstract,
                "text": text,
                "year": year,
                "matched_smoke_author_ids": matched,
                "all_author_ids": all_author_ids,
            }
        )

    history_rows.sort(key=lambda x: (int(x["year"]), x["paper_id"]))
    history_path = out_dir / "history_paper_texts.jsonl"
    write_jsonl(history_path, history_rows)

    author_history_path = out_dir / "author_history_papers.tsv"
    with author_history_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["author_id", "author_name", "history_paper_id"],
            delimiter="\t",
        )
        writer.writeheader()
        for author_id in sorted(author_papers):
            for paper_id in sorted(set(author_papers[author_id]), key=lambda x: int(x)):
                writer.writerow(
                    {
                        "author_id": author_id,
                        "author_name": author_names.get(author_id, ""),
                        "history_paper_id": paper_id,
                    }
                )

    hist_counts = Counter({author_id: len(set(papers)) for author_id, papers in author_papers.items()})
    missing_authors = sorted(smoke_authors - set(author_papers))
    summary_path = out_dir / "summary.tsv"
    with summary_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["metric", "value"])
        writer.writerow(["sample_jsonl", str(sample_path)])
        writer.writerow(["dblp_json", str(dblp_path)])
        writer.writerow(["cutoff_year", args.cutoff_year])
        writer.writerow(["scanned_records", scanned])
        writer.writerow(["smoke_authors", len(smoke_authors)])
        writer.writerow(["authors_with_text_history", len(author_papers)])
        writer.writerow(["authors_missing_text_history", len(missing_authors)])
        writer.writerow(["unique_history_papers_with_text", len(history_rows)])
        writer.writerow(["author_history_edges", sum(hist_counts.values())])
        writer.writerow(["min_history_per_author", min(hist_counts.values()) if hist_counts else 0])
        writer.writerow(["max_history_per_author", max(hist_counts.values()) if hist_counts else 0])
        writer.writerow(
            ["mean_history_per_author", sum(hist_counts.values()) / len(hist_counts) if hist_counts else 0]
        )
        writer.writerow(["skipped_history_papers_no_text", skipped_no_text])

    print(f"smoke_authors={len(smoke_authors)}")
    print(f"authors_with_text_history={len(author_papers)}")
    print(f"authors_missing_text_history={len(missing_authors)}")
    print(f"unique_history_papers_with_text={len(history_rows)}")
    print(f"author_history_edges={sum(hist_counts.values())}")
    print(f"history_paper_texts={history_path}")
    print(f"author_history_papers={author_history_path}")
    print(f"summary={summary_path}")


if __name__ == "__main__":
    main()
