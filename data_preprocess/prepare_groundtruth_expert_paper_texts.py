#!/usr/bin/env python3
"""Prepare title/abstract texts for all papers by ground-truth task experts.

This is intentionally narrower than prepare_hierec_embedding_inputs.py: it only
collects the historical papers of experts who appear in the 2020+ task teams.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

from embedding_pipeline_utils import decode_indexed_abstract, iter_json_objects


DEFAULT_TASKS = "data_preprocess/teams_2020plus_with_skill_weights.csv"
DEFAULT_PROFILE_DIR = "output/expert_profile_year_bins/all_2000_2019"
DEFAULT_DBLP = "data/dblp/dblp.v12.json"
DEFAULT_OUT = "output/groundtruth_expert_paper_embeddings"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Prepare full historical paper texts for ground-truth experts"
    )
    p.add_argument("--tasks-csv", default=DEFAULT_TASKS)
    p.add_argument("--profile-dir", default=DEFAULT_PROFILE_DIR)
    p.add_argument("--dblp-json", default=DEFAULT_DBLP)
    p.add_argument("--out-dir", default=DEFAULT_OUT)
    p.add_argument("--paper-text-max-chars", type=int, default=2000)
    p.add_argument("--progress-every", type=int, default=500000)
    return p.parse_args()


def parse_member_author_ids(raw_members: str) -> list[str]:
    out = []
    for member in (raw_members or "").split("|"):
        member = member.strip()
        if member:
            out.append(member.split("_", 1)[0])
    return out


def load_groundtruth_experts(tasks_csv: Path) -> tuple[set[str], dict[str, set[str]]]:
    experts: set[str] = set()
    expert_to_tasks: dict[str, set[str]] = defaultdict(set)
    with tasks_csv.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            paper_id = str(row.get("paper_id") or "")
            for expert_id in parse_member_author_ids(row.get("members", "")):
                experts.add(expert_id)
                expert_to_tasks[expert_id].add(paper_id)
    return experts, expert_to_tasks


def collect_profile_papers(
    profile_dir: Path,
    expert_ids: set[str],
) -> tuple[set[str], dict[str, set[str]], int, list[str]]:
    paper_ids: set[str] = set()
    expert_to_papers: dict[str, set[str]] = defaultdict(set)
    node_paper_entries = 0
    missing_profiles = []
    for expert_id in sorted(expert_ids):
        path = profile_dir / f"{expert_id}_direct_fos_nodes.tsv"
        if not path.exists():
            missing_profiles.append(expert_id)
            continue
        with path.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f, delimiter="\t"):
                try:
                    details = json.loads(row.get("paper_weight_details") or "[]")
                except json.JSONDecodeError:
                    details = []
                for paper in details:
                    if not isinstance(paper, dict):
                        continue
                    paper_id = str(paper.get("paper_id") or "").strip()
                    if not paper_id:
                        continue
                    paper_ids.add(paper_id)
                    expert_to_papers[expert_id].add(paper_id)
                    node_paper_entries += 1
    return paper_ids, expert_to_papers, node_paper_entries, missing_profiles


def scan_dblp_texts(
    dblp_json: Path,
    requested_ids: set[str],
    max_chars: int,
    progress_every: int,
) -> dict[str, dict]:
    found = {}
    parsed = 0
    for obj in iter_json_objects(dblp_json):
        parsed += 1
        if progress_every > 0 and parsed % progress_every == 0:
            print(
                "paper_text_progress "
                f"parsed={parsed:,} found={len(found):,}/{len(requested_ids):,}"
            )
        paper_id = str(obj.get("id", ""))
        if paper_id not in requested_ids:
            continue
        title = str(obj.get("title") or "").strip()
        abstract = decode_indexed_abstract(obj.get("indexed_abstract"))
        text = re.sub(r"\s+", " ", f"{title}. {abstract}".strip())
        if max_chars > 0:
            text = text[:max_chars]
        found[paper_id] = {
            "id": paper_id,
            "paper_id": paper_id,
            "year": obj.get("year", ""),
            "title": title,
            "abstract": abstract,
            "text": text,
        }
        if len(found) >= len(requested_ids):
            break
    return found


def write_jsonl(path: Path, rows) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    expert_ids, expert_to_tasks = load_groundtruth_experts(Path(args.tasks_csv))
    paper_ids, expert_to_papers, node_paper_entries, missing_profiles = collect_profile_papers(
        Path(args.profile_dir),
        expert_ids,
    )
    paper_texts = scan_dblp_texts(
        Path(args.dblp_json),
        paper_ids,
        args.paper_text_max_chars,
        args.progress_every,
    )

    write_jsonl(
        out_dir / "paper_texts.jsonl",
        (paper_texts[paper_id] for paper_id in sorted(paper_texts)),
    )

    with (out_dir / "groundtruth_experts.tsv").open(
        "w", encoding="utf-8", newline=""
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["expert_id", "task_count", "historical_paper_count"],
            delimiter="\t",
        )
        writer.writeheader()
        for expert_id in sorted(expert_ids):
            writer.writerow(
                {
                    "expert_id": expert_id,
                    "task_count": len(expert_to_tasks.get(expert_id, set())),
                    "historical_paper_count": len(expert_to_papers.get(expert_id, set())),
                }
            )

    with (out_dir / "expert_papers.tsv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["expert_id", "paper_id"], delimiter="\t")
        writer.writeheader()
        for expert_id in sorted(expert_to_papers):
            for paper_id in sorted(expert_to_papers[expert_id]):
                writer.writerow({"expert_id": expert_id, "paper_id": paper_id})

    with (out_dir / "summary.tsv").open("w", encoding="utf-8") as f:
        f.write("metric\tvalue\n")
        f.write(f"groundtruth_experts\t{len(expert_ids)}\n")
        f.write(f"missing_profiles\t{len(missing_profiles)}\n")
        f.write(f"unique_historical_papers_requested\t{len(paper_ids)}\n")
        f.write(f"unique_historical_papers_loaded\t{len(paper_texts)}\n")
        f.write(f"unique_expert_paper_pairs\t{sum(len(v) for v in expert_to_papers.values())}\n")
        f.write(f"node_paper_evidence_entries\t{node_paper_entries}\n")

    print(f"out_dir={out_dir}")
    print(f"paper_texts={out_dir / 'paper_texts.jsonl'}")
    print(f"summary={out_dir / 'summary.tsv'}")


if __name__ == "__main__":
    main()
