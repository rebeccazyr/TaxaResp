#!/usr/bin/env python3
"""Select cross-domain evaluation candidates with a two-stage FoS filter.

Stage 1 uses direct FoS label counts to find technically multi-facet papers.
Stage 2 checks whether the gold authors' pre-cutoff history profiles cover
different task labels, reducing false positives where many FoS labels are just
near-synonyms inside one author's specialty.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import DefaultDict, Iterable, List


DEFAULT_VALIDATION_JSONL = "outputs/temporal_task_splits_full/validation_2018_all_authors_hist_ge5.jsonl"
DEFAULT_DBLP_JSON = "data/dblp/dblp.v12.json"
DEFAULT_FOS_MAP = "../data/dblp/FieldsOfStudy.txt"
DEFAULT_OUT_DIR = "outputs/cross_domain_eval_selection"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-jsonl", default=DEFAULT_VALIDATION_JSONL)
    parser.add_argument("--dblp-json", default=DEFAULT_DBLP_JSON)
    parser.add_argument("--fos-map", default=DEFAULT_FOS_MAP)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--cutoff-year", type=int, default=2018)
    parser.add_argument("--task-min-fos-weight", type=float, default=0.5)
    parser.add_argument("--history-min-fos-weight", type=float, default=0.4)
    parser.add_argument("--min-authors", type=int, default=2)
    parser.add_argument("--candidate-min-l2", type=int, default=5)
    parser.add_argument("--candidate-min-l3", type=int, default=3)
    parser.add_argument("--candidate-min-l4", type=int, default=2)
    parser.add_argument("--strict-min-l2", type=int, default=6)
    parser.add_argument("--strict-min-l3", type=int, default=4)
    parser.add_argument("--strict-min-l4", type=int, default=3)
    parser.add_argument("--min-covered-labels", type=int, default=3)
    parser.add_argument("--min-coverage-frac", type=float, default=0.35)
    parser.add_argument("--min-distinct-cover-authors", type=int, default=2)
    parser.add_argument("--max-top-author-label-share", type=float, default=0.75)
    parser.add_argument("--sample-size", type=int, default=40)
    parser.add_argument("--seed", type=int, default=13)
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


def load_fos_levels(path: Path) -> dict[str, int]:
    name_to_level: dict[str, int] = {}
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 6:
                continue
            name = parts[3].strip().lower()
            if not name:
                continue
            try:
                level = int(parts[5])
            except ValueError:
                continue
            name_to_level[name] = level
    return name_to_level


def direct_fos_by_level(
    obj: dict,
    name_to_level: dict[str, int],
    min_weight: float,
    allowed_levels: set[int],
) -> dict[int, list[tuple[str, float]]]:
    labels: dict[int, list[tuple[str, float]]] = defaultdict(list)
    seen: set[tuple[int, str]] = set()
    for item in obj.get("fos") or []:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        if not name:
            continue
        try:
            weight = float(item.get("w") or 0.0)
        except (TypeError, ValueError):
            weight = 0.0
        if weight < min_weight:
            continue
        level = name_to_level.get(name.lower())
        if level not in allowed_levels:
            continue
        key = (level, name.lower())
        if key in seen:
            continue
        seen.add(key)
        labels[level].append((name, weight))
    for level in labels:
        labels[level].sort(key=lambda item: (item[0].lower(), -item[1]))
    return dict(labels)


def author_ids(obj: dict) -> list[str]:
    ids: list[str] = []
    for author in obj.get("authors") or []:
        if isinstance(author, dict) and author.get("id") is not None:
            ids.append(str(author["id"]))
    return ids


def is_fos_candidate(labels_by_level: dict[int, list[tuple[str, float]]], args: argparse.Namespace) -> bool:
    return (
        len(labels_by_level.get(2, [])) >= args.candidate_min_l2
        or len(labels_by_level.get(3, [])) >= args.candidate_min_l3
        or len(labels_by_level.get(4, [])) >= args.candidate_min_l4
    )


def is_fos_strict(labels_by_level: dict[int, list[tuple[str, float]]], args: argparse.Namespace) -> bool:
    return (
        len(labels_by_level.get(2, [])) >= args.strict_min_l2
        or len(labels_by_level.get(3, [])) >= args.strict_min_l3
        or len(labels_by_level.get(4, [])) >= args.strict_min_l4
    )


def label_key(level: int, name: str) -> str:
    return f"L{level}::{name.lower()}"


def label_display(level: int, name: str) -> str:
    return f"L{level}:{name}"


def load_candidate_tasks(args: argparse.Namespace, name_to_level: dict[str, int]) -> list[dict]:
    tasks: list[dict] = []
    for obj in iter_dblp_json(Path(args.validation_jsonl)):
        authors = author_ids(obj)
        if len(authors) < args.min_authors:
            continue
        labels_by_level = direct_fos_by_level(
            obj,
            name_to_level,
            args.task_min_fos_weight,
            allowed_levels={2, 3, 4},
        )
        if not is_fos_candidate(labels_by_level, args):
            continue
        task_labels: list[dict] = []
        for level in (2, 3, 4):
            for name, weight in labels_by_level.get(level, []):
                task_labels.append(
                    {
                        "level": level,
                        "name": name,
                        "key": label_key(level, name),
                        "display": label_display(level, name),
                        "weight": weight,
                    }
                )
        tasks.append(
            {
                "paper_id": str(obj.get("id")),
                "title": str(obj.get("title") or ""),
                "author_ids": authors,
                "author_count": len(authors),
                "labels_by_level": labels_by_level,
                "task_labels": task_labels,
                "fos_strict": is_fos_strict(labels_by_level, args),
            }
        )
    return tasks


def build_author_history_profiles(
    args: argparse.Namespace,
    candidate_author_ids: set[str],
    candidate_label_keys: set[str],
    name_to_level: dict[str, int],
) -> tuple[dict[str, Counter[str]], dict[str, float]]:
    profiles: DefaultDict[str, Counter[str]] = defaultdict(Counter)
    totals: DefaultDict[str, float] = defaultdict(float)
    scanned = 0
    matched_papers = 0
    for obj in iter_dblp_json(Path(args.dblp_json)):
        scanned += 1
        if args.progress_every and scanned % args.progress_every == 0:
            print(f"scanned={scanned:,} matched_history_papers={matched_papers:,}", flush=True)
        try:
            year = int(obj.get("year"))
        except (TypeError, ValueError):
            continue
        if year >= args.cutoff_year:
            continue
        matched_authors = sorted(set(author_ids(obj)) & candidate_author_ids)
        if not matched_authors:
            continue
        labels_by_level = direct_fos_by_level(
            obj,
            name_to_level,
            args.history_min_fos_weight,
            allowed_levels={2, 3, 4},
        )
        history_labels: list[tuple[str, float]] = []
        for level in (2, 3, 4):
            for name, weight in labels_by_level.get(level, []):
                key = label_key(level, name)
                if key in candidate_label_keys:
                    history_labels.append((key, weight))
        if not history_labels:
            continue
        matched_papers += 1
        for author_id in matched_authors:
            for key, weight in history_labels:
                profiles[author_id][key] += weight
                totals[author_id] += weight
    print(f"scanned={scanned:,} matched_history_papers={matched_papers:,}", flush=True)
    return dict(profiles), dict(totals)


def normalized_author_score(
    author_id: str,
    label: str,
    profiles: dict[str, Counter[str]],
    totals: dict[str, float],
) -> float:
    total = totals.get(author_id, 0.0)
    if total <= 0.0:
        return 0.0
    return profiles.get(author_id, Counter()).get(label, 0.0) / total


def evaluate_task_dispersion(
    task: dict,
    profiles: dict[str, Counter[str]],
    totals: dict[str, float],
    args: argparse.Namespace,
) -> dict:
    assignments: list[dict] = []
    for label in task["task_labels"]:
        best_author = ""
        best_score = 0.0
        best_raw = 0.0
        for author_id in task["author_ids"]:
            raw = profiles.get(author_id, Counter()).get(label["key"], 0.0)
            score = normalized_author_score(author_id, label["key"], profiles, totals)
            if score > best_score or (score == best_score and raw > best_raw):
                best_author = author_id
                best_score = score
                best_raw = raw
        if best_score > 0.0:
            assignments.append(
                {
                    "label": label["display"],
                    "label_key": label["key"],
                    "best_author": best_author,
                    "score": best_score,
                    "raw_weight": best_raw,
                }
            )

    label_count = len(task["task_labels"])
    covered = len(assignments)
    coverage_frac = covered / label_count if label_count else 0.0
    author_counts = Counter(row["best_author"] for row in assignments)
    distinct_authors = len(author_counts)
    top_share = max(author_counts.values()) / covered if covered else 1.0
    entropy = 0.0
    if covered and distinct_authors > 1:
        for count in author_counts.values():
            p = count / covered
            entropy -= p * math.log(p)
        entropy /= math.log(distinct_authors)

    passes = (
        covered >= args.min_covered_labels
        and coverage_frac >= args.min_coverage_frac
        and distinct_authors >= args.min_distinct_cover_authors
        and top_share <= args.max_top_author_label_share
    )
    return {
        "covered_label_count": covered,
        "coverage_frac": coverage_frac,
        "distinct_cover_authors": distinct_authors,
        "top_author_label_share": top_share,
        "assignment_entropy_norm": entropy,
        "passes_dispersion": passes,
        "assignments": assignments,
    }


def write_outputs(tasks: list[dict], evaluated: list[dict], args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for task, metrics in zip(tasks, evaluated):
        assignments = metrics["assignments"]
        rows.append(
            {
                "paper_id": task["paper_id"],
                "title": task["title"],
                "author_count": task["author_count"],
                "fos_strict": int(task["fos_strict"]),
                "direct_l2_count": len(task["labels_by_level"].get(2, [])),
                "direct_l3_count": len(task["labels_by_level"].get(3, [])),
                "direct_l4_count": len(task["labels_by_level"].get(4, [])),
                "task_label_count": len(task["task_labels"]),
                "covered_label_count": metrics["covered_label_count"],
                "coverage_frac": metrics["coverage_frac"],
                "distinct_cover_authors": metrics["distinct_cover_authors"],
                "top_author_label_share": metrics["top_author_label_share"],
                "assignment_entropy_norm": metrics["assignment_entropy_norm"],
                "passes_dispersion": int(metrics["passes_dispersion"]),
                "task_labels": " | ".join(label["display"] for label in task["task_labels"]),
                "covered_assignments": " | ".join(
                    f"{row['label']}=>{row['best_author']}:{row['score']:.4f}"
                    for row in assignments
                ),
            }
        )

    full_path = out_dir / "cross_domain_candidates_with_dispersion.tsv"
    with full_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    selected = [row for row in rows if row["passes_dispersion"] == 1]
    selected_path = out_dir / "selected_cross_domain_eval_candidates.tsv"
    with selected_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), delimiter="\t")
        writer.writeheader()
        writer.writerows(selected)

    rng = random.Random(args.seed)
    sample_rows = rng.sample(selected, min(args.sample_size, len(selected))) if selected else []
    sample_path = out_dir / "selected_sample_for_manual_check.tsv"
    with sample_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), delimiter="\t")
        writer.writeheader()
        writer.writerows(sample_rows)

    summary = {
        "validation_jsonl": args.validation_jsonl,
        "candidate_papers": len(rows),
        "selected_papers": len(selected),
        "strict_candidates": sum(row["fos_strict"] for row in rows),
        "selected_strict_papers": sum(row["fos_strict"] for row in selected),
        "task_min_fos_weight": args.task_min_fos_weight,
        "history_min_fos_weight": args.history_min_fos_weight,
        "min_authors": args.min_authors,
        "candidate_rule": {
            "direct_l2_ge": args.candidate_min_l2,
            "direct_l3_ge": args.candidate_min_l3,
            "direct_l4_ge": args.candidate_min_l4,
        },
        "dispersion_rule": {
            "min_covered_labels": args.min_covered_labels,
            "min_coverage_frac": args.min_coverage_frac,
            "min_distinct_cover_authors": args.min_distinct_cover_authors,
            "max_top_author_label_share": args.max_top_author_label_share,
        },
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"full={full_path}")
    print(f"selected={selected_path}")
    print(f"sample={sample_path}")


def main() -> None:
    args = parse_args()
    name_to_level = load_fos_levels(Path(args.fos_map))
    tasks = load_candidate_tasks(args, name_to_level)
    print(f"fos_candidate_tasks={len(tasks):,}", flush=True)
    candidate_author_ids = {author_id for task in tasks for author_id in task["author_ids"]}
    candidate_label_keys = {label["key"] for task in tasks for label in task["task_labels"]}
    print(
        f"candidate_authors={len(candidate_author_ids):,} candidate_labels={len(candidate_label_keys):,}",
        flush=True,
    )
    profiles, totals = build_author_history_profiles(
        args,
        candidate_author_ids,
        candidate_label_keys,
        name_to_level,
    )
    evaluated = [evaluate_task_dispersion(task, profiles, totals, args) for task in tasks]
    write_outputs(tasks, evaluated, args)


if __name__ == "__main__":
    main()
