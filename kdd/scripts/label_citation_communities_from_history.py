#!/usr/bin/env python3
"""Label citation communities with historical FoS, venues, authors, and papers."""

from __future__ import annotations

import argparse
import csv
import heapq
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


GENERIC_FOS = {
    "Computer science",
    "Mathematics",
    "Machine learning",
    "Artificial intelligence",
    "Algorithm",
    "Mathematical optimization",
    "Pattern recognition",
    "Data mining",
    "Information retrieval",
    "World Wide Web",
    "Distributed computing",
    "Computer network",
    "Computer vision",
    "Discrete mathematics",
    "Electronic engineering",
    "Engineering",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--membership-tsv", required=True)
    p.add_argument("--dblp-json", default="data/dblp/dblp.v12.json")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--cutoff-year", type=int, default=2018)
    p.add_argument("--progress-every", type=int, default=500000)
    p.add_argument("--sample-rows", type=int, default=80)
    return p.parse_args()


def safe_int(value: object) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def iter_json_objects(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            text = raw.strip()
            if not text or text in {"[", "]"}:
                continue
            if text.startswith(","):
                text = text[1:]
            if text.endswith(","):
                text = text[:-1]
            if not text.startswith("{"):
                continue
            try:
                yield json.loads(text)
            except json.JSONDecodeError:
                continue


def load_membership(path: Path) -> tuple[Dict[str, int], Dict[str, str], Dict[str, int], Dict[int, List[str]]]:
    author_block: Dict[str, int] = {}
    author_name: Dict[str, str] = {}
    author_hist: Dict[str, int] = {}
    block_authors: Dict[int, List[str]] = defaultdict(list)
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            author_id = row["author_id"]
            block_id = int(row["block_id"])
            author_block[author_id] = block_id
            author_name[author_id] = row.get("display_name", "")
            author_hist[author_id] = int(float(row.get("historical_papers_pre_cutoff") or 0))
            block_authors[block_id].append(author_id)
    return author_block, author_name, author_hist, dict(block_authors)


def top_counter(counter: Counter[str], n: int = 10) -> str:
    return "; ".join(f"{key} ({value:.1f})" for key, value in counter.most_common(n))


def top_authors(author_ids: List[str], author_name: Dict[str, str], author_hist: Dict[str, int], n: int = 8) -> str:
    ranked = sorted(
        author_ids,
        key=lambda author_id: (-author_hist.get(author_id, 0), author_name.get(author_id, ""), author_id),
    )[:n]
    return "; ".join(
        f"{author_name.get(author_id, '') or author_id} [{author_id}, hist={author_hist.get(author_id, 0)}]"
        for author_id in ranked
    )


def readable_label(fos_text: str) -> str:
    terms: List[str] = []
    for part in fos_text.split("; "):
        match = re.match(r"(.+) \(([0-9.]+)\)$", part.strip())
        if not match:
            continue
        term = match.group(1)
        if term not in GENERIC_FOS:
            terms.append(term)
    return " / ".join(terms[:4]) if terms else "broad computer science / mixed"


def representative_papers(items: List[Tuple[int, int, int, str, str]]) -> str:
    ranked = sorted(items, reverse=True)
    return " || ".join(
        f"{title} ({year}, id={paper_id}, in_block_authors={in_block_authors}, cites={citations})"
        for in_block_authors, citations, year, paper_id, title in ranked
    )


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    author_block, author_name, author_hist, block_authors = load_membership(Path(args.membership_tsv))
    block_fos: Dict[int, Counter[str]] = defaultdict(Counter)
    block_specific_fos: Dict[int, Counter[str]] = defaultdict(Counter)
    block_venues: Dict[int, Counter[str]] = defaultdict(Counter)
    block_hist_papers: Counter[int] = Counter()
    block_author_mentions: Counter[int] = Counter()
    rep_heap: Dict[int, List[Tuple[int, int, int, str, str]]] = defaultdict(list)

    parsed = 0
    for obj in iter_json_objects(Path(args.dblp_json)):
        parsed += 1
        if args.progress_every > 0 and parsed % args.progress_every == 0:
            print(f"scan_progress parsed={parsed:,}", flush=True)

        year = safe_int(obj.get("year"))
        if year is None or year >= args.cutoff_year:
            continue

        block_counts: Counter[int] = Counter()
        for author in obj.get("authors") or []:
            if not isinstance(author, dict):
                continue
            block_id = author_block.get(str(author.get("id", "")).strip())
            if block_id is not None:
                block_counts[block_id] += 1
        if not block_counts:
            continue

        fos_items: List[Tuple[str, float]] = []
        for item in obj.get("fos") or []:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name", "")).strip()
            if not name:
                continue
            try:
                weight = float(item.get("w", 1.0) or 0.0)
            except (TypeError, ValueError):
                weight = 1.0
            if weight > 0.0:
                fos_items.append((name, weight))

        venue = ""
        if isinstance(obj.get("venue"), dict):
            venue = str(obj["venue"].get("raw", "")).strip()
        title = str(obj.get("title", "")).strip().replace("\t", " ")
        paper_id = str(obj.get("id", "")).strip()
        citations = safe_int(obj.get("n_citation")) or 0

        for block_id, in_block_authors in block_counts.items():
            block_hist_papers[block_id] += 1
            block_author_mentions[block_id] += in_block_authors
            for name, weight in fos_items:
                block_fos[block_id][name] += weight
                if name not in GENERIC_FOS:
                    block_specific_fos[block_id][name] += weight
            if venue:
                block_venues[block_id][venue] += 1

            heap = rep_heap[block_id]
            item = (in_block_authors, citations, year, paper_id, title)
            if len(heap) < 3:
                heapq.heappush(heap, item)
            else:
                heapq.heappushpop(heap, item)

    rows = []
    for block_id in sorted(block_authors, key=lambda value: (-len(block_authors[value]), value)):
        author_ids = block_authors[block_id]
        fos_text = top_counter(block_fos[block_id])
        row = {
            "community_id": block_id,
            "authors": len(author_ids),
            "history_papers_with_community_author": block_hist_papers[block_id],
            "history_author_mentions": block_author_mentions[block_id],
            "avg_history_papers_per_author": f"{sum(author_hist[a] for a in author_ids) / len(author_ids):.6f}",
            "heuristic_label": readable_label(fos_text),
            "top_specific_fos": top_counter(block_specific_fos[block_id]),
            "top_fos": fos_text,
            "top_venues": top_counter(block_venues[block_id], 5),
            "top_authors_by_history_count": top_authors(author_ids, author_name, author_hist),
            "representative_history_papers": representative_papers(rep_heap[block_id]),
        }
        rows.append(row)

    labels_path = out_dir / "community_labels.tsv"
    with labels_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    size_dist = Counter(len(author_ids) for author_ids in block_authors.values())
    bucket_defs = [
        ("1", 1, 1),
        ("2-5", 2, 5),
        ("6-10", 6, 10),
        ("11-20", 11, 20),
        ("21-50", 21, 50),
        ("51-100", 51, 100),
        ("101-200", 101, 200),
        ("201-500", 201, 500),
        ("501-1000", 501, 1000),
        ("1001+", 1001, None),
    ]
    size_path = out_dir / "community_size_distribution.tsv"
    total_blocks = len(block_authors)
    total_authors = sum(len(author_ids) for author_ids in block_authors.values())
    with size_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["community_size_bucket", "communities", "authors", "pct_communities", "pct_authors"])
        for label, low, high in bucket_defs:
            selected_sizes = [
                size for size in size_dist.elements() if size >= low and (high is None or size <= high)
            ]
            writer.writerow(
                [
                    label,
                    len(selected_sizes),
                    sum(selected_sizes),
                    f"{len(selected_sizes) / total_blocks * 100:.6f}",
                    f"{sum(selected_sizes) / total_authors * 100:.6f}",
                ]
            )

    sample_rows = []
    for row in rows:
        size = int(row["authors"])
        if size >= 500 or 20 <= size <= 200:
            sample_rows.append(row)
        if len(sample_rows) >= args.sample_rows:
            break
    sample_path = out_dir / "sample_communities.tsv"
    with sample_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(sample_rows)

    print(f"labels={labels_path}")
    print(f"size_distribution={size_path}")
    print(f"sample={sample_path}")
    print(f"communities={len(rows):,}")


if __name__ == "__main__":
    main()
