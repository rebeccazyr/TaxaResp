#!/usr/bin/env python3
"""Build per-author direct FoS profiles before temporal cutoffs.

Default outputs:
- pre_2018_for_valid_2018: papers with 2000 <= year < 2018
- pre_2019_for_test_2019_2020: papers with 2000 <= year < 2019

By default this script uses every DBLP author id observed in qualifying papers.
Pass --expert-tsv only when a deliberate legacy expert subset is needed.

The per-author TSV columns match the legacy direct FoS profile builders.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, Dict, Iterable, List, Optional, Sequence, Set, Tuple


DEFAULT_CUTOFFS: Tuple[Tuple[str, int], ...] = (
    ("pre_2018_for_valid_2018", 2018),
    ("pre_2019_for_test_2019_2020", 2019),
)

HEADER = (
    "fos_id\tfos_name\tin_author_papers\tdirect_paper_count\t"
    "direct_weight_sum\tdirect_weight_avg\tpaper_weight_details\n"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build author direct FoS profiles from papers strictly before cutoff years."
    )
    parser.add_argument(
        "--dblp-json",
        default="data/dblp/dblp.v12.json",
        help="Path to dblp.v12.json.",
    )
    parser.add_argument(
        "--expert-tsv",
        default="",
        help=(
            "Optional TSV containing expert ids to use as a deliberate subset. "
            "If omitted, all DBLP author ids observed in qualifying papers are used."
        ),
    )
    parser.add_argument(
        "--fos-map",
        default="",
        help=(
            "FieldsOfStudy.txt path. If omitted, the script tries "
            "data/dblp/FieldsOfStudy.txt then ../data/dblp/FieldsOfStudy.txt. "
            "When absent, raw FoS names are still used with empty fos_id."
        ),
    )
    parser.add_argument(
        "--out-dir",
        default="outputs/expert_profile_cutoffs",
        help="Output root containing one subdirectory per cutoff.",
    )
    parser.add_argument(
        "--cutoff",
        action="append",
        default=[],
        metavar="LABEL:YEAR",
        help=(
            "Cutoff definition. Papers with year < YEAR are included in LABEL. "
            "Can be repeated. Defaults to validation/test cutoffs."
        ),
    )
    parser.add_argument(
        "--min-year",
        type=int,
        default=2000,
        help="Ignore papers before this year. Use 0 to include all parseable years.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=500000,
        help="Print progress every N parsed DBLP records. Use 0 to disable.",
    )
    return parser.parse_args()


def resolve_existing_path(value: str, candidates: Sequence[str], required: bool) -> Optional[Path]:
    paths = [Path(value)] if value else [Path(p) for p in candidates]
    for path in paths:
        if path.exists():
            return path
    if required:
        tried = ", ".join(str(p) for p in paths)
        raise FileNotFoundError(f"required input not found; tried: {tried}")
    return None


def parse_cutoffs(values: Sequence[str]) -> List[Tuple[str, int]]:
    if not values:
        return list(DEFAULT_CUTOFFS)

    cutoffs: List[Tuple[str, int]] = []
    seen: Set[str] = set()
    for value in values:
        if ":" not in value:
            raise ValueError(f"invalid --cutoff {value!r}; expected LABEL:YEAR")
        label, year_s = value.split(":", 1)
        label = label.strip()
        if not label:
            raise ValueError(f"invalid --cutoff {value!r}; empty label")
        if label in seen:
            raise ValueError(f"duplicate cutoff label: {label}")
        try:
            year = int(year_s)
        except ValueError as exc:
            raise ValueError(f"invalid cutoff year in {value!r}") from exc
        seen.add(label)
        cutoffs.append((label, year))
    return cutoffs


def safe_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def safe_int(value: object) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def load_expert_ids(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if reader.fieldnames and "expert_id" in reader.fieldnames:
            ids = [str(row.get("expert_id") or "").strip() for row in reader]
        else:
            f.seek(0)
            ids = [line.split("\t", 1)[0].strip() for line in f if line.strip()]

    seen: Set[str] = set()
    unique: List[str] = []
    for expert_id in ids:
        if not expert_id or expert_id in seen:
            continue
        seen.add(expert_id)
        unique.append(expert_id)
    return unique


def load_fos_map(path: Optional[Path]) -> Tuple[Dict[str, str], Dict[str, str]]:
    if path is None:
        return {}, {}

    name_to_id: Dict[str, str] = {}
    id_to_name: Dict[str, str] = {}
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


def collect_valid_fos(
    obj: dict,
    name_to_id: Dict[str, str],
    id_to_name: Dict[str, str],
) -> List[Tuple[str, str, str, float]]:
    fos_items = obj.get("fos") or []
    if not isinstance(fos_items, list):
        return []

    valid_fos: List[Tuple[str, str, str, float]] = []
    for item in fos_items:
        if not isinstance(item, dict):
            continue
        fos_name = str(item.get("name", "")).strip()
        if not fos_name:
            continue
        weight = safe_float(item.get("w", 0.0))
        if weight <= 0.0:
            continue
        mapped_id = name_to_id.get(fos_name.lower(), "")
        key = mapped_id if mapped_id else f"__NAME__:{fos_name.lower()}"
        resolved_name = id_to_name.get(mapped_id, fos_name) if mapped_id else fos_name
        valid_fos.append((key, mapped_id, resolved_name, round(weight, 5)))
    return valid_fos


def build_rows(fos_rows: Dict[str, dict]) -> List[dict]:
    rows: List[dict] = []
    for rec in fos_rows.values():
        details = [rec["paper_weights"][paper_id] for paper_id in sorted(rec["paper_weights"])]
        weight_sum = sum(float(item["weight"]) for item in details)
        count = len(details)
        rows.append(
            {
                "fos_id": rec["fos_id"],
                "fos_name": rec["fos_name"],
                "in_author_papers": 1,
                "direct_paper_count": count,
                "direct_weight_sum": weight_sum,
                "direct_weight_avg": weight_sum / count if count else 0.0,
                "paper_weight_details": json.dumps(details, ensure_ascii=False),
            }
        )
    rows.sort(
        key=lambda row: (
            -row["direct_weight_sum"],
            -row["direct_paper_count"],
            row["fos_name"].lower(),
        )
    )
    return rows


def author_sort_key(author_id: str) -> Tuple[int, object]:
    if author_id.isdigit():
        return (0, int(author_id))
    return (1, author_id)


def write_profile(path: Path, rows: List[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write(HEADER)
        for row in rows:
            f.write(
                f"{row['fos_id']}\t{row['fos_name']}\t{row['in_author_papers']}\t"
                f"{row['direct_paper_count']}\t{row['direct_weight_sum']:.5f}\t"
                f"{row['direct_weight_avg']:.5f}\t{row['paper_weight_details']}\n"
            )


def main() -> None:
    args = parse_args()

    dblp_path = resolve_existing_path(args.dblp_json, [args.dblp_json], required=True)
    expert_path = resolve_existing_path(args.expert_tsv, [], required=bool(args.expert_tsv))
    fos_map_path = resolve_existing_path(
        args.fos_map,
        ["data/dblp/FieldsOfStudy.txt", "../data/dblp/FieldsOfStudy.txt"],
        required=False,
    )
    assert dblp_path is not None

    cutoffs = parse_cutoffs(args.cutoff)
    min_year = args.min_year if args.min_year > 0 else None
    explicit_expert_ids = load_expert_ids(expert_path) if expert_path is not None else []
    expert_set = set(explicit_expert_ids) if explicit_expert_ids else None
    name_to_id, id_to_name = load_fos_map(fos_map_path)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for label, _ in cutoffs:
        (out_dir / label).mkdir(parents=True, exist_ok=True)

    ProfileMap = DefaultDict[str, Dict[str, dict]]
    profiles: Dict[str, ProfileMap] = {label: defaultdict(dict) for label, _ in cutoffs}
    papers_seen: Dict[str, DefaultDict[str, Set[str]]] = {
        label: defaultdict(set) for label, _ in cutoffs
    }

    parsed = 0
    matched_papers = 0
    cutoff_matches = {label: 0 for label, _ in cutoffs}
    for obj in iter_json_objects(dblp_path):
        parsed += 1
        if args.progress_every > 0 and parsed % args.progress_every == 0:
            progress = " ".join(
                f"{label}={cutoff_matches[label]:,}" for label, _ in cutoffs
            )
            print(
                f"progress parsed={parsed:,} matched_papers={matched_papers:,} {progress}",
                flush=True,
            )

        year = safe_int(obj.get("year"))
        if year is None:
            continue
        if min_year is not None and year < min_year:
            continue

        active_cutoffs = [(label, cutoff_year) for label, cutoff_year in cutoffs if year < cutoff_year]
        if not active_cutoffs:
            continue

        authors = obj.get("authors") or []
        if not isinstance(authors, list):
            continue
        author_ids = {
            str(author.get("id", "")).strip()
            for author in authors
            if isinstance(author, dict) and str(author.get("id", "")).strip()
        }
        if expert_set is not None:
            author_ids = {author_id for author_id in author_ids if author_id in expert_set}
        if not author_ids:
            continue

        paper_id = str(obj.get("id", "")).strip()
        if not paper_id:
            continue
        valid_fos = collect_valid_fos(obj, name_to_id, id_to_name)
        if not valid_fos:
            continue

        matched_papers += 1
        for label, _ in active_cutoffs:
            cutoff_matches[label] += 1
            for expert_id in author_ids:
                papers_seen[label][expert_id].add(paper_id)
                expert_profile = profiles[label][expert_id]
                for key, mapped_id, resolved_name, weight in valid_fos:
                    if key not in expert_profile:
                        expert_profile[key] = {
                            "fos_id": mapped_id,
                            "fos_name": resolved_name,
                            "paper_weights": {},
                        }
                    previous = expert_profile[key]["paper_weights"].get(paper_id)
                    if previous is None or weight > previous["weight"]:
                        expert_profile[key]["paper_weights"][paper_id] = {
                            "paper_id": paper_id,
                            "year": year,
                            "weight": weight,
                        }

    run_summary_path = out_dir / "_run_summary.tsv"
    with run_summary_path.open("w", encoding="utf-8") as run_summary:
        run_summary.write(
            "label\tcutoff_year\tincluded_years\tauthor_mode\tauthors\tmatched_papers\toutput_dir\tsummary_file\n"
        )
        for label, cutoff_year in cutoffs:
            cutoff_dir = out_dir / label
            output_author_ids = (
                explicit_expert_ids
                if explicit_expert_ids
                else sorted(profiles[label], key=author_sort_key)
            )
            summary_path = cutoff_dir / "_summary.tsv"
            with summary_path.open("w", encoding="utf-8") as summary:
                summary.write("author_id\tpapers\tdirect_fos_nodes\toutput_file\n")
                for expert_id in output_author_ids:
                    rows = build_rows(profiles[label].get(expert_id, {}))
                    out_path = cutoff_dir / f"{expert_id}_direct_fos_nodes.tsv"
                    write_profile(out_path, rows)
                    summary.write(
                        f"{expert_id}\t{len(papers_seen[label].get(expert_id, set()))}\t"
                        f"{len(rows)}\t{out_path}\n"
                    )

            included_years = (
                f"{min_year or '-inf'}-{cutoff_year - 1}"
                if cutoff_year > (min_year or cutoff_year)
                else f"<{cutoff_year}"
            )
            run_summary.write(
                f"{label}\t{cutoff_year}\t{included_years}\t"
                f"{'expert_tsv_subset' if explicit_expert_ids else 'all_dblp_author_ids'}\t"
                f"{len(output_author_ids)}\t"
                f"{cutoff_matches[label]}\t{cutoff_dir}\t{summary_path}\n"
            )

    print(f"author_mode={'expert_tsv_subset' if explicit_expert_ids else 'all_dblp_author_ids'}")
    if explicit_expert_ids:
        print(f"experts={len(explicit_expert_ids)}")
    print(f"parsed_records={parsed}")
    print(f"matched_papers={matched_papers}")
    print(f"output_dir={out_dir}")
    print(f"run_summary={run_summary_path}")


if __name__ == "__main__":
    main()
