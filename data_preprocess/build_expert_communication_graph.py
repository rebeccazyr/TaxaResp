#!/usr/bin/env python3
"""Build expert communication graph from DBLP JSON.

Graph definition:
- node: expert (from expert_id_name.tsv)
- edge: two experts have coauthored at least one paper
- edge_weight: number of coauthored papers
- edge annotation: list of coauthored paper ids
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build communication graph for experts from DBLP data"
    )
    parser.add_argument(
        "--expert-tsv",
        default="data/dblp/expert_id_name.tsv",
        help="Input TSV containing experts (expert_id, name)",
    )
    parser.add_argument(
        "--dblp-json",
        default="data/dblp/dblp.v12.json",
        help="Path to dblp.v12.json",
    )
    parser.add_argument(
        "--out-dir",
        default="output/expert_graph",
        help="Output directory",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=500000,
        help="Log progress every N parsed papers",
    )
    return parser.parse_args()


def load_experts(path: Path) -> Tuple[Dict[str, str], List[str]]:
    id_to_name: Dict[str, str] = {}
    ordered_ids: List[str] = []
    with path.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f, delimiter="\t")
        if r.fieldnames and "expert_id" in r.fieldnames:
            for row in r:
                eid = str((row.get("expert_id") or "")).strip()
                if not eid:
                    continue
                if eid in id_to_name:
                    continue
                id_to_name[eid] = str((row.get("name") or "")).strip()
                ordered_ids.append(eid)
        else:
            f.seek(0)
            first = True
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if first:
                    first = False
                    # best-effort skip header if looks like one
                    if parts and parts[0].lower() in {"expert_id", "id"}:
                        continue
                if not parts:
                    continue
                eid = parts[0].strip()
                if not eid or eid in id_to_name:
                    continue
                id_to_name[eid] = parts[1].strip() if len(parts) > 1 else ""
                ordered_ids.append(eid)
    return id_to_name, ordered_ids


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


def pair_key(a: str, b: str) -> Tuple[str, str]:
    return (a, b) if a < b else (b, a)


def main() -> None:
    args = parse_args()

    id_to_name, ordered_ids = load_experts(Path(args.expert_tsv))
    expert_set: Set[str] = set(ordered_ids)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # pair -> list of coauthored paper ids
    edge_papers: Dict[Tuple[str, str], List[str]] = defaultdict(list)

    parsed = 0
    matched_papers = 0
    matched_expert_mentions = 0
    for obj in iter_json_objects(Path(args.dblp_json)):
        parsed += 1
        if args.progress_every > 0 and parsed % args.progress_every == 0:
            print(
                f"progress parsed={parsed:,} matched_papers={matched_papers:,} "
                f"edges={len(edge_papers):,}"
            )

        paper_id = str(obj.get("id", "")).strip()
        if not paper_id:
            continue

        authors = obj.get("authors") or []
        if not isinstance(authors, list):
            continue

        paper_experts: List[str] = []
        for a in authors:
            if not isinstance(a, dict):
                continue
            aid = str(a.get("id", "")).strip()
            if aid and aid in expert_set:
                paper_experts.append(aid)

        # keep unique within paper
        if paper_experts:
            matched_expert_mentions += len(paper_experts)
            paper_experts = sorted(set(paper_experts))
        if len(paper_experts) < 2:
            continue

        matched_papers += 1
        for u, v in itertools.combinations(paper_experts, 2):
            edge_papers[pair_key(u, v)].append(paper_id)

    nodes_path = out_dir / "communication_nodes.tsv"
    with nodes_path.open("w", encoding="utf-8") as f:
        f.write("expert_id\texpert_name\n")
        for eid in ordered_ids:
            f.write(f"{eid}\t{id_to_name.get(eid, '')}\n")

    edges_path = out_dir / "communication_edges.tsv"
    with edges_path.open("w", encoding="utf-8") as f:
        f.write(
            "source_expert_id\tsource_expert_name\t"
            "target_expert_id\ttarget_expert_name\t"
            "edge_weight\tco_paper_ids\n"
        )
        for (u, v) in sorted(edge_papers.keys()):
            papers = edge_papers[(u, v)]
            # deduplicate while preserving order
            seen = set()
            uniq = []
            for pid in papers:
                if pid in seen:
                    continue
                seen.add(pid)
                uniq.append(pid)
            f.write(
                f"{u}\t{id_to_name.get(u,'')}\t{v}\t{id_to_name.get(v,'')}\t"
                f"{len(uniq)}\t{json.dumps(uniq, ensure_ascii=False)}\n"
            )

    stats_path = out_dir / "communication_stats.txt"
    with stats_path.open("w", encoding="utf-8") as f:
        f.write(f"experts_total={len(ordered_ids)}\n")
        f.write(f"parsed_papers={parsed}\n")
        f.write(f"matched_papers_with_2plus_experts={matched_papers}\n")
        f.write(f"matched_expert_mentions={matched_expert_mentions}\n")
        f.write(f"edges_total={len(edge_papers)}\n")
        f.write(f"nodes_file={nodes_path}\n")
        f.write(f"edges_file={edges_path}\n")

    print(f"experts_total={len(ordered_ids)}")
    print(f"parsed_papers={parsed}")
    print(f"matched_papers_with_2plus_experts={matched_papers}")
    print(f"edges_total={len(edge_papers)}")
    print(f"output_dir={out_dir}")


if __name__ == "__main__":
    main()
