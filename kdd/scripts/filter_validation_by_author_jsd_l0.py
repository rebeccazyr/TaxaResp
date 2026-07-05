#!/usr/bin/env python3
"""Filter 2018 validation papers by author-count and author-history L0 JSD."""

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


DEFAULT_VALIDATION_JSONL = "outputs/temporal_task_splits_full/validation_2018_all_authors_hist_ge5.jsonl"
DEFAULT_DBLP_JSON = "data/dblp/dblp.v12.json"
DEFAULT_FOS_MAP = "../data/dblp/FieldsOfStudy.txt"
DEFAULT_FOS_PARENTS = "../data/dblp/13.FieldOfStudyChildren.nt"
DEFAULT_OUT_DIR = "outputs/cross_domain_eval_selection/author_jsd_l0_ge0p019_author_ge3"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-jsonl", default=DEFAULT_VALIDATION_JSONL)
    parser.add_argument("--dblp-json", default=DEFAULT_DBLP_JSON)
    parser.add_argument("--fos-map", default=DEFAULT_FOS_MAP)
    parser.add_argument("--fos-parents", default=DEFAULT_FOS_PARENTS)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--cutoff-year", type=int, default=2018)
    parser.add_argument("--history-min-fos-weight", type=float, default=0.4)
    parser.add_argument("--min-authors", type=int, default=3)
    parser.add_argument("--min-author-jsd-l0-mean", type=float, default=0.019)
    parser.add_argument("--sample-size", type=int, default=50)
    parser.add_argument("--progress-every", type=int, default=500000)
    return parser.parse_args()


def iter_dblp_json(path: Path) -> Iterable[dict]:
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


def author_ids(obj: dict) -> list[str]:
    return [
        str(author["id"])
        for author in obj.get("authors") or []
        if isinstance(author, dict) and author.get("id") is not None
    ]


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


def make_projector(parents: dict[int, list[int]], id_to_level: dict[int, int]) :
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
        return tuple(sorted(found))

    return ancestors_at_level


def paper_fos_items(
    obj: dict,
    min_weight: float,
    name_to_id: dict[str, int],
    id_to_level: dict[int, int],
) -> list[tuple[int, float]]:
    items = []
    seen: set[int] = set()
    for item in obj.get("fos") or []:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip().lower()
        if not name:
            continue
        try:
            weight = float(item.get("w") or 0.0)
        except (TypeError, ValueError):
            weight = 0.0
        if weight < min_weight:
            continue
        fos_id = name_to_id.get(name)
        if fos_id is None or fos_id in seen or fos_id not in id_to_level:
            continue
        seen.add(fos_id)
        items.append((fos_id, weight))
    return items


def add_history_profile(
    profile: Counter[str],
    items: list[tuple[int, float]],
    ancestors_at_level,
    id_to_name: dict[int, str],
) -> None:
    for fos_id, weight in items:
        for ancestor_id in ancestors_at_level(fos_id, 0):
            profile[id_to_name.get(ancestor_id, str(ancestor_id))] += weight


def jsd(counter_a: Counter[str], counter_b: Counter[str]) -> float | None:
    total_a = sum(counter_a.values())
    total_b = sum(counter_b.values())
    if total_a <= 0.0 or total_b <= 0.0:
        return None
    value = 0.0
    for key in set(counter_a) | set(counter_b):
        p = counter_a.get(key, 0.0) / total_a
        q = counter_b.get(key, 0.0) / total_b
        m = 0.5 * (p + q)
        if p > 0.0:
            value += 0.5 * p * math.log(p / m)
        if q > 0.0:
            value += 0.5 * q * math.log(q / m)
    return value


def paper_jsd(author_list: list[str], profiles: dict[str, Counter[str]]) -> tuple[float, float, int]:
    values = []
    for author_a, author_b in combinations(sorted(set(author_list)), 2):
        value = jsd(profiles.get(author_a, Counter()), profiles.get(author_b, Counter()))
        if value is not None:
            values.append(value)
    if not values:
        return 0.0, 0.0, 0
    return sum(values) / len(values), min(values), len(values)


def bucket_jsd(value: float) -> str:
    bins = [
        (0.0, 0.005),
        (0.005, 0.01),
        (0.01, 0.019),
        (0.019, 0.03),
        (0.03, 0.05),
        (0.05, 0.08),
        (0.08, 0.12),
        (0.12, 0.2),
        (0.2, 1.0),
    ]
    for low, high in bins:
        if low <= value < high:
            return f"[{low:g},{high:g})"
    return "other"


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    validation = read_jsonl(Path(args.validation_jsonl))
    validation_author_ids = {author_id for obj in validation for author_id in author_ids(obj)}
    print(f"validation_papers={len(validation):,} validation_authors={len(validation_author_ids):,}", flush=True)

    id_to_name, id_to_level, name_to_id = load_fos(Path(args.fos_map))
    parents = load_parents(Path(args.fos_parents))
    ancestors_at_level = make_projector(parents, id_to_level)

    profiles: dict[str, Counter[str]] = defaultdict(Counter)
    scanned = 0
    matched_history = 0
    for obj in iter_dblp_json(Path(args.dblp_json)):
        scanned += 1
        if args.progress_every and scanned % args.progress_every == 0:
            print(f"scanned={scanned:,} matched_history_papers={matched_history:,}", flush=True)
        try:
            year = int(obj.get("year"))
        except (TypeError, ValueError):
            continue
        if year >= args.cutoff_year:
            continue
        matched_authors = sorted(set(author_ids(obj)) & validation_author_ids)
        if not matched_authors:
            continue
        items = paper_fos_items(obj, args.history_min_fos_weight, name_to_id, id_to_level)
        if not items:
            continue
        matched_history += 1
        for author_id in matched_authors:
            add_history_profile(profiles[author_id], items, ancestors_at_level, id_to_name)
    print(f"scanned={scanned:,} matched_history_papers={matched_history:,}", flush=True)

    feature_rows = []
    selected_objects = []
    before_author_count_dist = Counter()
    after_author_count_dist = Counter()
    jsd_dist_all = Counter()
    jsd_dist_selected = Counter()
    before_author_appearances = 0
    after_author_appearances = 0
    before_unique_authors: set[str] = set()
    after_unique_authors: set[str] = set()
    missing_pair_count = 0
    for obj in validation:
        authors = author_ids(obj)
        author_count = len(authors)
        before_author_count_dist[author_count] += 1
        before_author_appearances += author_count
        before_unique_authors.update(authors)
        mean_jsd, min_jsd, pair_count = paper_jsd(authors, profiles)
        if pair_count == 0 and author_count >= 2:
            missing_pair_count += 1
        selected = author_count >= args.min_authors and mean_jsd >= args.min_author_jsd_l0_mean
        if selected:
            after_author_count_dist[author_count] += 1
            after_author_appearances += author_count
            after_unique_authors.update(authors)
            selected_objects.append(obj)
            jsd_dist_selected[bucket_jsd(mean_jsd)] += 1
        jsd_dist_all[bucket_jsd(mean_jsd)] += 1
        feature_rows.append(
            {
                "paper_id": str(obj.get("id")),
                "title": str(obj.get("title") or ""),
                "author_count": author_count,
                "author_jsd_l0_mean": mean_jsd,
                "author_jsd_l0_min": min_jsd,
                "author_jsd_pair_count": pair_count,
                "selected": int(selected),
            }
        )

    features_path = out_dir / "validation_2018_author_jsd_l0_features.tsv"
    with features_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=list(feature_rows[0].keys()))
        writer.writeheader()
        writer.writerows(feature_rows)

    selected_tsv_path = out_dir / "selected_papers.tsv"
    with selected_tsv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=list(feature_rows[0].keys()))
        writer.writeheader()
        writer.writerows(row for row in feature_rows if row["selected"] == 1)

    selected_jsonl_path = out_dir / "selected_papers.jsonl"
    with selected_jsonl_path.open("w", encoding="utf-8") as f:
        for obj in selected_objects:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    sample_path = out_dir / "selected_sample.tsv"
    with sample_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=list(feature_rows[0].keys()))
        writer.writeheader()
        writer.writerows([row for row in feature_rows if row["selected"] == 1][: args.sample_size])

    author_dist_path = out_dir / "author_count_distribution.tsv"
    all_author_counts = sorted(set(before_author_count_dist) | set(after_author_count_dist))
    with author_dist_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["author_count", "before_papers", "after_papers", "after_share_within_count"])
        for author_count in all_author_counts:
            before = before_author_count_dist[author_count]
            after = after_author_count_dist[author_count]
            writer.writerow([author_count, before, after, after / before if before else 0.0])

    jsd_dist_path = out_dir / "author_jsd_l0_distribution.tsv"
    all_buckets = ["[0,0.005)", "[0.005,0.01)", "[0.01,0.019)", "[0.019,0.03)", "[0.03,0.05)", "[0.05,0.08)", "[0.08,0.12)", "[0.12,0.2)", "[0.2,1)"]
    with jsd_dist_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["author_jsd_l0_mean_bucket", "all_papers", "selected_papers"])
        for bucket in all_buckets:
            writer.writerow([bucket, jsd_dist_all[bucket], jsd_dist_selected[bucket]])

    summary = {
        "validation_jsonl": args.validation_jsonl,
        "filter": {
            "min_authors": args.min_authors,
            "min_author_jsd_l0_mean": args.min_author_jsd_l0_mean,
            "history_min_fos_weight": args.history_min_fos_weight,
            "cutoff_year": args.cutoff_year,
        },
        "papers_before": len(validation),
        "papers_after": len(selected_objects),
        "paper_keep_ratio": len(selected_objects) / len(validation) if validation else 0.0,
        "author_appearances_before": before_author_appearances,
        "author_appearances_after": after_author_appearances,
        "unique_authors_before": len(before_unique_authors),
        "unique_authors_after": len(after_unique_authors),
        "mean_authors_per_paper_before": before_author_appearances / len(validation) if validation else 0.0,
        "mean_authors_per_paper_after": after_author_appearances / len(selected_objects) if selected_objects else 0.0,
        "papers_with_no_valid_jsd_pairs": missing_pair_count,
        "matched_history_papers": matched_history,
        "outputs": {
            "features_tsv": str(features_path),
            "selected_tsv": str(selected_tsv_path),
            "selected_jsonl": str(selected_jsonl_path),
            "sample_tsv": str(sample_path),
            "author_count_distribution_tsv": str(author_dist_path),
            "author_jsd_l0_distribution_tsv": str(jsd_dist_path),
        },
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
