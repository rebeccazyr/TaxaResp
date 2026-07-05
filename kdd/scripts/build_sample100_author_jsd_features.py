#!/usr/bin/env python3
"""Build direct/L0/L1/L2 author-history JSD features for the manual sample100 set."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from build_smoke200_cross_domain_all_features import (
    add_profile_counts,
    fos_items,
    iter_json_records,
    load_fos,
    load_parents,
    make_projector,
    pairwise_jsd_stats,
)


DEFAULT_SAMPLE_JSONL = "outputs/cross_domain_eval_selection/author_jsd_l0_ge0p019_author_ge3/full_fos_sample100_audit.jsonl"
DEFAULT_ANNOTATIONS = "outputs/cross_domain_eval_selection/author_jsd_l0_ge0p019_author_ge3/sample100_manual_cross_domain_annotations.tsv"
DEFAULT_DBLP_JSON = "data/dblp/dblp.v12.json"
DEFAULT_FOS_MAP = "../data/dblp/FieldsOfStudy.txt"
DEFAULT_FOS_PARENTS = "../data/dblp/13.FieldOfStudyChildren.nt"
DEFAULT_OUT = "outputs/cross_domain_eval_selection/author_jsd_l0_ge0p019_author_ge3/sample100_author_jsd_all_levels.tsv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-jsonl", default=DEFAULT_SAMPLE_JSONL)
    parser.add_argument("--annotations-tsv", default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--dblp-json", default=DEFAULT_DBLP_JSON)
    parser.add_argument("--fos-map", default=DEFAULT_FOS_MAP)
    parser.add_argument("--fos-parents", default=DEFAULT_FOS_PARENTS)
    parser.add_argument("--out-tsv", default=DEFAULT_OUT)
    parser.add_argument("--cutoff-year", type=int, default=2018)
    parser.add_argument("--history-min-fos-weight", type=float, default=0.4)
    parser.add_argument("--progress-every", type=int, default=500000)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def read_annotations(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return {row["paper_id"]: row for row in csv.DictReader(f, delimiter="\t")}


def sample_author_ids(obj: dict) -> list[str]:
    ids = []
    for author in obj.get("authors") or []:
        if not isinstance(author, dict):
            continue
        author_id = author.get("author_id", author.get("id"))
        if author_id is not None:
            ids.append(str(author_id))
    return ids


def dblp_author_ids(obj: dict) -> list[str]:
    ids = []
    for author in obj.get("authors") or []:
        if isinstance(author, dict) and author.get("id") is not None:
            ids.append(str(author["id"]))
    return ids


def main() -> None:
    args = parse_args()
    sample = read_jsonl(Path(args.sample_jsonl))
    annotations = read_annotations(Path(args.annotations_tsv))
    id_to_name, id_to_level, name_to_id = load_fos(Path(args.fos_map))
    parents = load_parents(Path(args.fos_parents))
    ancestors_at_level = make_projector(parents, id_to_level, id_to_name)

    all_sample_author_ids = {author_id for obj in sample for author_id in sample_author_ids(obj)}
    profiles = {}
    from collections import Counter, defaultdict

    profiles = defaultdict(lambda: defaultdict(Counter))
    scanned = 0
    matched = 0
    for obj in iter_json_records(Path(args.dblp_json)):
        scanned += 1
        if args.progress_every and scanned % args.progress_every == 0:
            print(f"scanned={scanned:,} matched_history_papers={matched:,}", flush=True)
        try:
            year = int(obj.get("year"))
        except (TypeError, ValueError):
            continue
        if year >= args.cutoff_year:
            continue
        matched_authors = sorted(set(dblp_author_ids(obj)) & all_sample_author_ids)
        if not matched_authors:
            continue
        items = fos_items(obj, args.history_min_fos_weight, name_to_id, id_to_level)
        if not items:
            continue
        matched += 1
        for author_id in matched_authors:
            add_profile_counts(profiles, author_id, items, ancestors_at_level, id_to_name)
    print(f"scanned={scanned:,} matched_history_papers={matched:,}", flush=True)

    rows = []
    for index, obj in enumerate(sample, 1):
        paper_id = str(obj.get("paper_id") or obj.get("id"))
        authors = sample_author_ids(obj)
        ann = annotations.get(paper_id, {})
        row = {
            "sample_index": index,
            "paper_id": paper_id,
            "title": str(obj.get("title") or ""),
            "author_count": len(authors),
            "manual_label": ann.get("manual_label", ""),
            "theme_cross_domain": ann.get("theme_cross_domain", ""),
            "author_cross_domain_evidence": ann.get("author_cross_domain_evidence", ""),
        }
        for key in ("direct", "l0", "l1", "l2"):
            mean, min_value, pair_count = pairwise_jsd_stats(authors, profiles, key)
            row[f"author_jsd_{key}_mean"] = mean
            row[f"author_jsd_{key}_min"] = min_value
            row[f"author_jsd_{key}_pair_count"] = pair_count
        rows.append(row)

    out_path = Path(args.out_tsv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote={out_path}")


if __name__ == "__main__":
    main()
