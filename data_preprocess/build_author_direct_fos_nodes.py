#!/usr/bin/env python3
"""Build direct FoS node table for one author.

This script only keeps FoS that explicitly appear in the author's papers.
It does NOT add ancestor/bridge FoS for taxonomy completion.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export direct FoS nodes (no taxonomy completion) for an author id"
    )
    parser.add_argument("--author-id", required=True, help="Author id in dblp.v12.json")
    parser.add_argument(
        "--dblp-json",
        default="data/dblp/dblp.v12.json",
        help="Path to dblp.v12.json",
    )
    parser.add_argument(
        "--fos-map",
        default="data/dblp/FieldsOfStudy.txt",
        help="Path to FieldsOfStudy.txt",
    )
    parser.add_argument(
        "--out-dir",
        default="output/author_papers",
        help="Output directory",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Optional output file path; default is <out-dir>/<author_id>_direct_fos_nodes.tsv",
    )
    return parser.parse_args()


def load_fos_map(path: Path) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Return (name_to_id, id_to_name)."""
    name_to_id: Dict[str, str] = {}
    id_to_name: Dict[str, str] = {}

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                continue
            fos_id = parts[0].strip()
            norm_name = parts[2].strip()
            disp_name = parts[3].strip()
            if not fos_id:
                continue

            id_to_name[fos_id] = disp_name or norm_name or fos_id
            if norm_name:
                name_to_id[norm_name.lower()] = fos_id
            if disp_name:
                name_to_id[disp_name.lower()] = fos_id

    return name_to_id, id_to_name


def iter_json_objects(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            s = raw.strip()
            if not s:
                continue
            if s[0] == ",":
                s = s[1:]
            if not s.startswith("{"):
                continue
            try:
                yield json.loads(s)
            except json.JSONDecodeError:
                continue


def safe_float(v: object) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def main() -> None:
    args = parse_args()

    author_id = str(args.author_id)
    dblp_path = Path(args.dblp_json)
    fos_map_path = Path(args.fos_map)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = Path(args.output) if args.output else out_dir / f"{author_id}_direct_fos_nodes.tsv"

    name_to_id, id_to_name = load_fos_map(fos_map_path)

    # key -> summary
    # key is fos_id when mapped, otherwise "__NAME__:<lower_name>" for unmapped FoS.
    fos_rows: Dict[str, dict] = {}
    papers_seen = set()

    for obj in iter_json_objects(dblp_path):
        authors = obj.get("authors") or []
        if not isinstance(authors, list):
            continue

        matched = any(str((a or {}).get("id", "")) == author_id for a in authors if isinstance(a, dict))
        if not matched:
            continue

        paper_id = str(obj.get("id", ""))
        if not paper_id:
            continue
        papers_seen.add(paper_id)

        year = obj.get("year", "")
        title = obj.get("title", "")
        fos_items = obj.get("fos") or []
        if not isinstance(fos_items, list):
            continue

        for item in fos_items:
            if not isinstance(item, dict):
                continue
            fos_name = str(item.get("name", "")).strip()
            if not fos_name:
                continue
            w = safe_float(item.get("w", 0.0))
            # User requirement: zero/non-positive weights are excluded from stats.
            if w <= 0.0:
                continue
            mapped_id = name_to_id.get(fos_name.lower(), "")
            key = mapped_id if mapped_id else f"__NAME__:{fos_name.lower()}"

            if key not in fos_rows:
                resolved_name = id_to_name.get(mapped_id, fos_name) if mapped_id else fos_name
                fos_rows[key] = {
                    "fos_id": mapped_id,
                    "fos_name": resolved_name,
                    "paper_weights": {},  # paper_id -> detail dict
                }

            # keep max weight if same paper/fos repeats
            prev = fos_rows[key]["paper_weights"].get(paper_id)
            if prev is None or w > prev["weight"]:
                fos_rows[key]["paper_weights"][paper_id] = {
                    "paper_id": paper_id,
                    "weight": round(w, 5),
                }

    rows_out: List[dict] = []
    for rec in fos_rows.values():
        details = [rec["paper_weights"][pid] for pid in sorted(rec["paper_weights"].keys())]
        weight_sum = sum(d["weight"] for d in details)
        cnt = len(details)
        rows_out.append(
            {
                "fos_id": rec["fos_id"],
                "fos_name": rec["fos_name"],
                "in_author_papers": 1,
                "direct_paper_count": cnt,
                "direct_weight_sum": weight_sum,
                "direct_weight_avg": (weight_sum / cnt) if cnt else 0.0,
                "paper_weight_details": json.dumps(details, ensure_ascii=False),
            }
        )

    rows_out.sort(key=lambda r: (-r["direct_weight_sum"], -r["direct_paper_count"], r["fos_name"].lower()))

    with out_path.open("w", encoding="utf-8") as f:
        f.write(
            "fos_id\tfos_name\tin_author_papers\tdirect_paper_count\t"
            "direct_weight_sum\tdirect_weight_avg\tpaper_weight_details\n"
        )
        for r in rows_out:
            f.write(
                f"{r['fos_id']}\t{r['fos_name']}\t{r['in_author_papers']}\t"
                f"{r['direct_paper_count']}\t{r['direct_weight_sum']:.5f}\t"
                f"{r['direct_weight_avg']:.5f}\t{r['paper_weight_details']}\n"
            )

    print(f"author_id={author_id}")
    print(f"papers={len(papers_seen)}")
    print(f"direct_fos_nodes={len(rows_out)}")
    print(f"output={out_path}")


if __name__ == "__main__":
    main()
