#!/usr/bin/env python3
"""Sample filtered 2018 validation papers and write full-FoS audit evidence."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


DEFAULT_VALIDATION_JSONL = "outputs/temporal_task_splits_full/validation_2018_all_authors_hist_ge5.jsonl"
DEFAULT_DBLP_JSON = "data/dblp/dblp.v12.json"
DEFAULT_FOS_MAP = "../data/dblp/FieldsOfStudy.txt"
DEFAULT_OUT_MD = "outputs/cross_domain_eval_selection/full_fos_sample_audit_direct_l2_ge3_author_ge3.md"
DEFAULT_OUT_JSONL = "outputs/cross_domain_eval_selection/full_fos_sample_audit_direct_l2_ge3_author_ge3.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-jsonl", default=DEFAULT_VALIDATION_JSONL)
    parser.add_argument("--dblp-json", default=DEFAULT_DBLP_JSON)
    parser.add_argument("--fos-map", default=DEFAULT_FOS_MAP)
    parser.add_argument("--out-md", default=DEFAULT_OUT_MD)
    parser.add_argument("--out-jsonl", default=DEFAULT_OUT_JSONL)
    parser.add_argument("--cutoff-year", type=int, default=2018)
    parser.add_argument("--task-min-fos-weight", type=float, default=0.5)
    parser.add_argument("--min-direct-l2", type=int, default=3)
    parser.add_argument("--min-authors", type=int, default=3)
    parser.add_argument("--sample-size", type=int, default=10)
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--history-top-k", type=int, default=12)
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


def author_rows(obj: dict) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for author in obj.get("authors") or []:
        if isinstance(author, dict) and author.get("id") is not None:
            rows.append(
                {
                    "author_id": str(author.get("id")),
                    "name": str(author.get("name") or ""),
                }
            )
    return rows


def fos_rows(obj: dict, name_to_level: dict[str, int]) -> list[dict]:
    rows = []
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
        rows.append(
            {
                "name": name,
                "weight": weight,
                "level": name_to_level.get(name.lower()),
            }
        )
    rows.sort(key=lambda row: (-row["weight"], row["level"] if row["level"] is not None else 99, row["name"].lower()))
    return rows


def direct_l2_count(obj: dict, name_to_level: dict[str, int], min_weight: float) -> int:
    labels = {
        row["name"].lower()
        for row in fos_rows(obj, name_to_level)
        if row["level"] == 2 and row["weight"] >= min_weight
    }
    return len(labels)


def abstract_text(obj: dict) -> str:
    indexed = obj.get("indexed_abstract") or {}
    inverted = indexed.get("InvertedIndex") if isinstance(indexed, dict) else None
    if not isinstance(inverted, dict):
        return ""
    positions: list[tuple[int, str]] = []
    for word, indices in inverted.items():
        if not isinstance(indices, list):
            continue
        for index in indices:
            try:
                positions.append((int(index), str(word)))
            except (TypeError, ValueError):
                continue
    return " ".join(word for _, word in sorted(positions))


def load_candidates(args: argparse.Namespace, name_to_level: dict[str, int]) -> list[dict]:
    candidates = []
    for obj in iter_dblp_json(Path(args.validation_jsonl)):
        authors = author_rows(obj)
        if len(authors) < args.min_authors:
            continue
        l2_count = direct_l2_count(obj, name_to_level, args.task_min_fos_weight)
        if l2_count < args.min_direct_l2:
            continue
        candidates.append(
            {
                "paper_id": str(obj.get("id")),
                "title": str(obj.get("title") or ""),
                "abstract": abstract_text(obj),
                "year": obj.get("year"),
                "authors": authors,
                "direct_l2_count": l2_count,
                "paper_fos": fos_rows(obj, name_to_level),
            }
        )
    return candidates


def build_author_history(
    args: argparse.Namespace,
    sampled_author_ids: set[str],
    name_to_level: dict[str, int],
) -> dict[str, dict]:
    history = {
        author_id: {
            "paper_count": 0,
            "fos": Counter(),
        }
        for author_id in sampled_author_ids
    }
    scanned = 0
    matched = 0
    for obj in iter_dblp_json(Path(args.dblp_json)):
        scanned += 1
        if args.progress_every and scanned % args.progress_every == 0:
            print(f"scanned={scanned:,} matched_history_papers={matched:,}", flush=True)
        try:
            year = int(obj.get("year"))
        except (TypeError, ValueError):
            continue
        if year >= args.cutoff_year:
            continue
        matched_authors = sorted({row["author_id"] for row in author_rows(obj)} & sampled_author_ids)
        if not matched_authors:
            continue
        rows = fos_rows(obj, name_to_level)
        if not rows:
            continue
        matched += 1
        for author_id in matched_authors:
            history[author_id]["paper_count"] += 1
            for row in rows:
                key = f"L{row['level']}:{row['name']}" if row["level"] is not None else f"L?:{row['name']}"
                history[author_id]["fos"][key] += row["weight"]
    print(f"scanned={scanned:,} matched_history_papers={matched:,}", flush=True)
    return history


def top_history_labels(history_row: dict, limit: int) -> list[dict]:
    total = sum(history_row["fos"].values())
    rows = []
    for label, weight in history_row["fos"].most_common(limit):
        rows.append({"label": label, "weight": weight, "share": weight / total if total else 0.0})
    return rows


def write_outputs(args: argparse.Namespace, sample: list[dict], history: dict[str, dict]) -> None:
    out_md = Path(args.out_md)
    out_jsonl = Path(args.out_jsonl)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    with out_jsonl.open("w", encoding="utf-8") as f:
        for paper in sample:
            row = dict(paper)
            row["author_history"] = {
                author["author_id"]: {
                    "paper_count": history[author["author_id"]]["paper_count"],
                    "top_direct_fos": top_history_labels(history[author["author_id"]], args.history_top_k),
                }
                for author in paper["authors"]
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    lines = [
        "# Full-FoS Sample Audit",
        "",
        f"Filter: direct_l2_count >= {args.min_direct_l2}, author_count >= {args.min_authors}, task FoS weight >= {args.task_min_fos_weight}",
        f"Sample size: {len(sample)}, seed: {args.seed}",
        "",
    ]
    for index, paper in enumerate(sample, 1):
        lines.extend(
            [
                f"## {index}. {paper['title']}",
                "",
                f"- paper_id: `{paper['paper_id']}`",
                f"- year: {paper['year']}",
                f"- author_count: {len(paper['authors'])}",
                f"- direct_l2_count: {paper['direct_l2_count']}",
                f"- abstract: {paper['abstract'][:900] if paper['abstract'] else '(missing)'}",
                "",
                "Paper FoS labels, all direct labels sorted by weight:",
            ]
        )
        for row in paper["paper_fos"]:
            level = row["level"] if row["level"] is not None else "?"
            lines.append(f"- L{level} {row['name']} ({row['weight']:.4f})")
        lines.extend(["", "Groundtruth author historical direct FoS, pre-2018 top labels:"])
        for author in paper["authors"]:
            h = history[author["author_id"]]
            lines.append(f"- {author['name']} (`{author['author_id']}`), history_papers={h['paper_count']}")
            for label in top_history_labels(h, args.history_top_k):
                lines.append(f"  - {label['label']} weight={label['weight']:.3f}, share={label['share']:.3f}")
        lines.extend(["", "Judgment: TODO", "Reason: TODO", ""])
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote={out_md}")
    print(f"wrote={out_jsonl}")


def main() -> None:
    args = parse_args()
    name_to_level = load_fos_levels(Path(args.fos_map))
    candidates = load_candidates(args, name_to_level)
    rng = random.Random(args.seed)
    sample = rng.sample(candidates, min(args.sample_size, len(candidates)))
    sampled_author_ids = {author["author_id"] for paper in sample for author in paper["authors"]}
    print(f"candidates={len(candidates):,} sampled_papers={len(sample)} sampled_authors={len(sampled_author_ids)}", flush=True)
    history = build_author_history(args, sampled_author_ids, name_to_level)
    write_outputs(args, sample, history)


if __name__ == "__main__":
    main()
