#!/usr/bin/env python3
"""Build cross-domain classifier features for every paper in smoke_200."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict, deque
from functools import lru_cache
from itertools import combinations
from pathlib import Path
from typing import Iterable


DEFAULT_SAMPLE = "outputs/stage1_pilot_samples/smoke_200.jsonl"
DEFAULT_DBLP_JSON = "data/dblp/dblp.v12.json"
DEFAULT_FOS_MAP = "../data/dblp/FieldsOfStudy.txt"
DEFAULT_FOS_PARENTS = "../data/dblp/13.FieldOfStudyChildren.nt"
DEFAULT_EXISTING_LABELS = "outputs/cross_domain_eval_selection_smoke_200/smoke_200_cross_domain_labels.tsv"
DEFAULT_MANUAL = "outputs/cross_domain_eval_selection_smoke_200/smoke_200_manual_cross_domain_annotations.tsv"
DEFAULT_OUT = "outputs/cross_domain_eval_selection_smoke_200/smoke_200_cross_domain_all_features.tsv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-jsonl", default=DEFAULT_SAMPLE)
    parser.add_argument("--dblp-json", default=DEFAULT_DBLP_JSON)
    parser.add_argument("--fos-map", default=DEFAULT_FOS_MAP)
    parser.add_argument("--fos-parents", default=DEFAULT_FOS_PARENTS)
    parser.add_argument("--existing-labels-tsv", default=DEFAULT_EXISTING_LABELS)
    parser.add_argument("--manual-tsv", default=DEFAULT_MANUAL)
    parser.add_argument("--out-tsv", default=DEFAULT_OUT)
    parser.add_argument("--cutoff-year", type=int, default=2018)
    parser.add_argument("--task-min-fos-weight", type=float, default=0.5)
    parser.add_argument("--history-min-fos-weight", type=float, default=0.4)
    parser.add_argument("--progress-every", type=int, default=500000)
    return parser.parse_args()


def iter_json_records(path: Path) -> Iterable[dict]:
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


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def read_tsv_by_id(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as f:
        return {row["paper_id"]: row for row in csv.DictReader(f, delimiter="\t")}


def author_ids(obj: dict) -> list[str]:
    return [str(author["id"]) for author in obj.get("authors") or [] if isinstance(author, dict) and author.get("id") is not None]


def load_fos(path: Path) -> tuple[dict[int, str], dict[int, int], dict[str, int]]:
    id_to_name: dict[int, str] = {}
    id_to_level: dict[int, int] = {}
    name_to_id: dict[str, int] = {}
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 6:
                continue
            try:
                fos_id = int(parts[0])
                level = int(parts[5])
            except ValueError:
                continue
            display = parts[3].strip() or parts[2].strip()
            normalized = parts[2].strip()
            if not display:
                continue
            id_to_name[fos_id] = display
            id_to_level[fos_id] = level
            for name in {display, normalized}:
                if name:
                    name_to_id.setdefault(name.lower(), fos_id)
    return id_to_name, id_to_level, name_to_id


def load_parents(path: Path) -> dict[int, list[int]]:
    parents: dict[int, list[int]] = defaultdict(list)
    entity_re = re.compile(r"/entity/(\d+)>")
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            ids = [int(match) for match in entity_re.findall(line)]
            if len(ids) >= 2:
                child, parent = ids[0], ids[1]
                parents[child].append(parent)
    return dict(parents)


def fos_items(obj: dict, min_weight: float, name_to_id: dict[str, int], id_to_level: dict[int, int]) -> list[tuple[int, str, int, float]]:
    items = []
    seen: set[int] = set()
    for item in obj.get("fos") or []:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        try:
            weight = float(item.get("w") or 0.0)
        except (TypeError, ValueError):
            weight = 0.0
        if not name or weight < min_weight:
            continue
        fos_id = name_to_id.get(name.lower())
        if fos_id is None or fos_id in seen:
            continue
        level = id_to_level.get(fos_id)
        if level is None:
            continue
        seen.add(fos_id)
        items.append((fos_id, name, level, weight))
    return items


def make_projector(
    parents: dict[int, list[int]],
    id_to_level: dict[int, int],
    id_to_name: dict[int, str],
):
    @lru_cache(maxsize=None)
    def ancestors_at_level(fos_id: int, target_level: int) -> tuple[int, ...]:
        start_level = id_to_level.get(fos_id)
        if start_level is None:
            return ()
        if start_level == target_level:
            return (fos_id,)
        if start_level < target_level:
            return ()
        found: set[int] = set()
        queue: deque[int] = deque(parents.get(fos_id, []))
        visited: set[int] = set()
        while queue:
            current = queue.popleft()
            if current in visited:
                continue
            visited.add(current)
            level = id_to_level.get(current)
            if level is None:
                continue
            if level == target_level:
                found.add(current)
            elif level > target_level:
                queue.extend(parents.get(current, []))
        return tuple(sorted(found, key=lambda item: id_to_name.get(item, str(item)).lower()))

    return ancestors_at_level


def add_profile_counts(
    profiles: dict[str, dict[str, Counter[str]]],
    author_id: str,
    items: list[tuple[int, str, int, float]],
    ancestors_at_level,
    id_to_name: dict[int, str],
) -> None:
    for fos_id, name, level, weight in items:
        profiles[author_id]["direct"][f"L{level}:{name}"] += weight
        for target in (0, 1, 2):
            ancestors = ancestors_at_level(fos_id, target)
            for ancestor_id in ancestors:
                profiles[author_id][f"l{target}"][f"L{target}:{id_to_name.get(ancestor_id, str(ancestor_id))}"] += weight


def jsd(counter_a: Counter[str], counter_b: Counter[str]) -> float | None:
    total_a = sum(counter_a.values())
    total_b = sum(counter_b.values())
    if total_a <= 0.0 or total_b <= 0.0:
        return None
    keys = set(counter_a) | set(counter_b)
    value = 0.0
    for key in keys:
        p = counter_a.get(key, 0.0) / total_a
        q = counter_b.get(key, 0.0) / total_b
        m = 0.5 * (p + q)
        if p > 0.0:
            value += 0.5 * p * math.log(p / m)
        if q > 0.0:
            value += 0.5 * q * math.log(q / m)
    return value


def pairwise_jsd_stats(author_list: list[str], profiles: dict[str, dict[str, Counter[str]]], profile_key: str) -> tuple[float, float, int]:
    values = []
    for author_a, author_b in combinations(sorted(set(author_list)), 2):
        value = jsd(profiles[author_a][profile_key], profiles[author_b][profile_key])
        if value is not None:
            values.append(value)
    if not values:
        return 0.0, 0.0, 0
    return sum(values) / len(values), min(values), len(values)


def task_projected_labels(
    items: list[tuple[int, str, int, float]],
    target_level: int,
    ancestors_at_level,
    id_to_name: dict[int, str],
) -> list[str]:
    labels: set[str] = set()
    for fos_id, _name, _level, _weight in items:
        for ancestor_id in ancestors_at_level(fos_id, target_level):
            labels.add(id_to_name.get(ancestor_id, str(ancestor_id)))
    return sorted(labels, key=str.lower)


def main() -> None:
    args = parse_args()
    sample = read_jsonl(Path(args.sample_jsonl))
    manual = read_tsv_by_id(Path(args.manual_tsv))
    existing = read_tsv_by_id(Path(args.existing_labels_tsv))
    id_to_name, id_to_level, name_to_id = load_fos(Path(args.fos_map))
    parents = load_parents(Path(args.fos_parents))
    ancestors_at_level = make_projector(parents, id_to_level, id_to_name)

    sample_author_ids = {author_id for obj in sample for author_id in author_ids(obj)}
    profiles: dict[str, dict[str, Counter[str]]] = defaultdict(lambda: defaultdict(Counter))
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
        matched_authors = sorted(set(author_ids(obj)) & sample_author_ids)
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
    passthrough_columns = [
        "direct_l2_count",
        "direct_l3_count",
        "direct_l4_count",
        "fos_candidate",
        "fos_strict",
        "covered_label_count",
        "coverage_frac",
        "distinct_cover_authors",
        "top_author_label_share",
        "default_dispersion_pass",
        "high_precision_dispersion_pass",
        "r20_block_count",
        "coarse_block_count",
        "task_labels",
        "covered_assignments",
    ]
    for obj in sample:
        paper_id = str(obj.get("id"))
        authors = author_ids(obj)
        task_items = fos_items(obj, args.task_min_fos_weight, name_to_id, id_to_level)
        row = {
            "paper_id": paper_id,
            "title": str(obj.get("title") or ""),
            "author_count": len(authors),
            "manual_label": manual.get(paper_id, {}).get("manual_label", ""),
            "use_for_cross_domain_eval": manual.get(paper_id, {}).get("use_for_cross_domain_eval", ""),
            "level0_projected_count": len(task_projected_labels(task_items, 0, ancestors_at_level, id_to_name)),
            "level1_projected_count": len(task_projected_labels(task_items, 1, ancestors_at_level, id_to_name)),
            "level2_projected_count": len(task_projected_labels(task_items, 2, ancestors_at_level, id_to_name)),
            "level0_projected_labels": " | ".join(task_projected_labels(task_items, 0, ancestors_at_level, id_to_name)),
            "level1_projected_labels": " | ".join(task_projected_labels(task_items, 1, ancestors_at_level, id_to_name)),
            "level2_projected_labels": " | ".join(task_projected_labels(task_items, 2, ancestors_at_level, id_to_name)),
        }
        existing_row = existing.get(paper_id, {})
        for column in passthrough_columns:
            row[column] = existing_row.get(column, "")
        for key in ("direct", "l0", "l1", "l2"):
            mean, min_value, pair_count = pairwise_jsd_stats(authors, profiles, key)
            row[f"author_jsd_{key}_mean"] = mean
            row[f"author_jsd_{key}_min"] = min_value
            row[f"author_jsd_{key}_pair_count"] = pair_count
        rows.append(row)

    out_path = Path(args.out_tsv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote={out_path}")


if __name__ == "__main__":
    main()
