#!/usr/bin/env python3
"""Build full temporal validation/test task sets from DBLP.

Default outputs:
- validation_2018.jsonl: all DBLP records with year == 2018
- test_2019_2020.jsonl: all DBLP records with year in {2019, 2020}

No expert-id allowlist is used. Author lists are preserved exactly as present in
DBLP so downstream experiments can decide their own candidate pools.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Set


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export full DBLP temporal validation/test task JSONL files."
    )
    parser.add_argument(
        "--dblp-json",
        default="data/dblp/dblp.v12.json",
        help="Path to dblp.v12.json.",
    )
    parser.add_argument(
        "--out-dir",
        default="outputs/temporal_task_splits_full",
        help="Output directory for JSONL splits and summary files.",
    )
    parser.add_argument(
        "--validation-years",
        default="2018",
        help="Comma-separated validation years.",
    )
    parser.add_argument(
        "--test-years",
        default="2019,2020",
        help="Comma-separated test years.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=500000,
        help="Print progress every N parsed DBLP records. Use 0 to disable.",
    )
    return parser.parse_args()


def parse_years(value: str) -> Set[int]:
    years: Set[int] = set()
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        years.add(int(item))
    if not years:
        raise ValueError("year list must not be empty")
    return years


def safe_int(value: object) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def iter_json_objects(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            text = raw.strip()
            if not text:
                continue
            if text[0] == ",":
                text = text[1:]
            if not text.startswith("{"):
                continue
            try:
                yield json.loads(text)
            except json.JSONDecodeError:
                continue


def collect_author_ids(obj: dict) -> Set[str]:
    authors = obj.get("authors") or []
    if not isinstance(authors, list):
        return set()
    return {
        str(author.get("id", "")).strip()
        for author in authors
        if isinstance(author, dict) and str(author.get("id", "")).strip()
    }


def has_fos(obj: dict) -> bool:
    fos_items = obj.get("fos") or []
    return isinstance(fos_items, list) and bool(fos_items)


def write_record(handle, obj: dict) -> None:
    handle.write(json.dumps(obj, ensure_ascii=False, separators=(",", ":")))
    handle.write("\n")


def main() -> None:
    args = parse_args()
    dblp_path = Path(args.dblp_json)
    if not dblp_path.exists():
        raise FileNotFoundError(f"DBLP JSON not found: {dblp_path}")

    validation_years = parse_years(args.validation_years)
    test_years = parse_years(args.test_years)
    overlap = validation_years & test_years
    if overlap:
        raise ValueError(f"validation/test years overlap: {sorted(overlap)}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    validation_label = "_".join(str(year) for year in sorted(validation_years))
    validation_path = out_dir / f"validation_{validation_label}.jsonl"
    test_label = "_".join(str(year) for year in sorted(test_years))
    test_path = out_dir / f"test_{test_label}.jsonl"
    summary_path = out_dir / "_summary.tsv"

    stats: Dict[str, dict] = {
        "validation": {
            "years": validation_years,
            "path": validation_path,
            "records": 0,
            "author_edges": 0,
            "unique_author_ids": set(),
            "records_with_authors": 0,
            "records_with_fos": 0,
        },
        "test": {
            "years": test_years,
            "path": test_path,
            "records": 0,
            "author_edges": 0,
            "unique_author_ids": set(),
            "records_with_authors": 0,
            "records_with_fos": 0,
        },
    }

    parsed = 0
    with validation_path.open("w", encoding="utf-8") as validation_out, test_path.open(
        "w", encoding="utf-8"
    ) as test_out:
        handles = {"validation": validation_out, "test": test_out}
        for obj in iter_json_objects(dblp_path):
            parsed += 1
            if args.progress_every > 0 and parsed % args.progress_every == 0:
                print(
                    "progress "
                    f"parsed={parsed:,} "
                    f"validation={stats['validation']['records']:,} "
                    f"test={stats['test']['records']:,}",
                    flush=True,
                )

            year = safe_int(obj.get("year"))
            if year is None:
                continue
            if year in validation_years:
                split = "validation"
            elif year in test_years:
                split = "test"
            else:
                continue

            author_ids = collect_author_ids(obj)
            split_stats = stats[split]
            split_stats["records"] += 1
            split_stats["author_edges"] += len(author_ids)
            split_stats["unique_author_ids"].update(author_ids)
            if author_ids:
                split_stats["records_with_authors"] += 1
            if has_fos(obj):
                split_stats["records_with_fos"] += 1
            write_record(handles[split], obj)

    with summary_path.open("w", encoding="utf-8") as summary:
        summary.write(
            "split\tyears\trecords\trecords_with_authors\trecords_with_fos\t"
            "author_edges\tunique_author_ids\toutput_file\n"
        )
        for split in ("validation", "test"):
            split_stats = stats[split]
            summary.write(
                f"{split}\t"
                f"{','.join(str(year) for year in sorted(split_stats['years']))}\t"
                f"{split_stats['records']}\t"
                f"{split_stats['records_with_authors']}\t"
                f"{split_stats['records_with_fos']}\t"
                f"{split_stats['author_edges']}\t"
                f"{len(split_stats['unique_author_ids'])}\t"
                f"{split_stats['path']}\n"
            )

    print(f"parsed_records={parsed}")
    print(f"validation_jsonl={validation_path}")
    print(f"test_jsonl={test_path}")
    print(f"summary={summary_path}")


if __name__ == "__main__":
    main()
