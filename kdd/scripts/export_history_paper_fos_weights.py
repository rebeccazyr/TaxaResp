#!/usr/bin/env python3
"""Export FoS alpha weights for already embedded history papers."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history-ids-tsv", required=True)
    parser.add_argument("--dblp-json", default="data/dblp/dblp.v12.json")
    parser.add_argument("--fos-map", default="../data/dblp/FieldsOfStudy.txt")
    parser.add_argument("--out-tsv", required=True)
    parser.add_argument("--progress-every", type=int, default=500000)
    return parser.parse_args()


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


def load_history_ids(path: Path) -> set[str]:
    ids: set[str] = set()
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            paper_id = str(row.get("id") or "").strip()
            if paper_id:
                ids.add(paper_id)
    return ids


def load_fos_maps(path: Path) -> tuple[dict[str, str], dict[str, str]]:
    name_to_id: dict[str, str] = {}
    id_to_name: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                continue
            fos_id = parts[0].strip()
            norm_name = parts[2].strip()
            display_name = parts[3].strip()
            if not fos_id:
                continue
            id_to_name[fos_id] = display_name or norm_name or fos_id
            if norm_name:
                name_to_id[norm_name.lower()] = fos_id
            if display_name:
                name_to_id[display_name.lower()] = fos_id
    return name_to_id, id_to_name


def safe_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def main() -> None:
    args = parse_args()
    history_ids = load_history_ids(Path(args.history_ids_tsv))
    name_to_id, id_to_name = load_fos_maps(Path(args.fos_map))
    out_path = Path(args.out_tsv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    scanned = 0
    matched = 0
    rows = 0
    remaining = set(history_ids)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["paper_id", "fos_id", "fos_name", "weight"],
            delimiter="\t",
        )
        writer.writeheader()
        for obj in iter_json_objects(Path(args.dblp_json)):
            scanned += 1
            if args.progress_every > 0 and scanned % args.progress_every == 0:
                print(
                    f"scanned_records={scanned} matched_history_papers={matched} remaining={len(remaining)}",
                    file=sys.stderr,
                    flush=True,
                )
            paper_id = str(obj.get("id") or "").strip()
            if paper_id not in remaining:
                continue
            remaining.remove(paper_id)
            matched += 1
            for item in obj.get("fos") or []:
                if not isinstance(item, dict):
                    continue
                raw_name = str(item.get("name") or "").strip()
                if not raw_name:
                    continue
                weight = safe_float(item.get("w", item.get("weight", 0.0)))
                if weight <= 0.0:
                    continue
                fos_id = name_to_id.get(raw_name.lower(), "")
                if not fos_id:
                    continue
                writer.writerow(
                    {
                        "paper_id": paper_id,
                        "fos_id": fos_id,
                        "fos_name": id_to_name.get(fos_id, raw_name),
                        "weight": f"{weight:.8g}",
                    }
                )
                rows += 1
            if not remaining:
                break

    print(f"history_ids={len(history_ids)}")
    print(f"matched_history_papers={matched}")
    print(f"missing_history_papers={len(remaining)}")
    print(f"fos_weight_rows={rows}")
    print(f"out_tsv={out_path}")


if __name__ == "__main__":
    main()
