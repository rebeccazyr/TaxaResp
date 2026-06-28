#!/usr/bin/env python3
"""Build direct FoS profiles for all experts in one pass over dblp.v12.json.

Output format per expert is aligned with build_author_direct_fos_nodes.py:
  <expert_id>_direct_fos_nodes.tsv

Key behavior:
- Only FoS that explicitly appear in papers are used (no taxonomy completion).
- FoS with weight <= 0 are excluded from all statistics.
- The DBLP JSON file is scanned once.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build all expert direct FoS profiles in one pass over DBLP JSON"
    )
    parser.add_argument(
        "--expert-tsv",
        default="data/dblp/expert_id_name.tsv",
        help="TSV containing expert ids (expects column `expert_id` or first column as id)",
    )
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
        default="output/expert_profile",
        help="Output directory for per-expert profiles",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=500000,
        help="Print progress every N parsed records",
    )
    return parser.parse_args()


def safe_float(v: object) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def load_expert_ids(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if reader.fieldnames and "expert_id" in reader.fieldnames:
            ids = [str((row.get("expert_id") or "")).strip() for row in reader]
        else:
            f.seek(0)
            ids = []
            for line in f:
                s = line.strip()
                if not s:
                    continue
                ids.append(s.split("\t")[0].strip())
    ids = [x for x in ids if x]
    # keep stable unique order
    seen: Set[str] = set()
    unique: List[str] = []
    for x in ids:
        if x in seen:
            continue
        seen.add(x)
        unique.append(x)
    return unique


def load_fos_map(path: Path) -> Tuple[Dict[str, str], Dict[str, str]]:
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


def main() -> None:
    args = parse_args()

    expert_ids = load_expert_ids(Path(args.expert_tsv))
    expert_set = set(expert_ids)
    name_to_id, id_to_name = load_fos_map(Path(args.fos_map))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Per expert:
    # profiles[expert_id][fos_key] = {
    #   fos_id, fos_name, paper_weights: {paper_id: {paper_id, year, weight, title}}
    # }
    profiles: Dict[str, Dict[str, dict]] = defaultdict(dict)
    papers_seen: Dict[str, Set[str]] = defaultdict(set)

    parsed = 0
    matched_papers = 0
    for obj in iter_json_objects(Path(args.dblp_json)):
        parsed += 1
        if args.progress_every > 0 and parsed % args.progress_every == 0:
            print(f"progress parsed={parsed:,} matched_papers={matched_papers:,}")

        authors = obj.get("authors") or []
        if not isinstance(authors, list):
            continue

        matched_experts: Set[str] = set()
        for a in authors:
            if not isinstance(a, dict):
                continue
            aid = str(a.get("id", ""))
            if aid in expert_set:
                matched_experts.add(aid)
        if not matched_experts:
            continue

        paper_id = str(obj.get("id", ""))
        if not paper_id:
            continue
        matched_papers += 1

        year = obj.get("year", "")
        title = obj.get("title", "")

        fos_items = obj.get("fos") or []
        if not isinstance(fos_items, list):
            fos_items = []

        # Precompute valid FoS for this paper once, reuse for all matched experts.
        valid_fos = []
        for item in fos_items:
            if not isinstance(item, dict):
                continue
            fos_name = str(item.get("name", "")).strip()
            if not fos_name:
                continue
            w = safe_float(item.get("w", 0.0))
            # User requirement: do not count zero/non-positive weights.
            if w <= 0.0:
                continue
            mapped_id = name_to_id.get(fos_name.lower(), "")
            key = mapped_id if mapped_id else f"__NAME__:{fos_name.lower()}"
            resolved_name = id_to_name.get(mapped_id, fos_name) if mapped_id else fos_name
            valid_fos.append((key, mapped_id, resolved_name, round(w, 5)))

        for expert_id in matched_experts:
            papers_seen[expert_id].add(paper_id)

            edict = profiles[expert_id]
            for key, mapped_id, resolved_name, weight in valid_fos:
                if key not in edict:
                    edict[key] = {
                        "fos_id": mapped_id,
                        "fos_name": resolved_name,
                        "paper_weights": {},
                    }

                prev = edict[key]["paper_weights"].get(paper_id)
                if prev is None or weight > prev["weight"]:
                    edict[key]["paper_weights"][paper_id] = {
                        "paper_id": paper_id,
                        "weight": weight,
                    }

    # Write per-expert output
    summary_path = out_dir / "_summary.tsv"
    with summary_path.open("w", encoding="utf-8") as sf:
        sf.write("expert_id\tpapers\tdirect_fos_nodes\toutput_file\n")

        for expert_id in expert_ids:
            fos_rows = profiles.get(expert_id, {})
            rows_out: List[dict] = []
            for rec in fos_rows.values():
                details = [rec["paper_weights"][pid] for pid in sorted(rec["paper_weights"])]
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

            rows_out.sort(
                key=lambda r: (
                    -r["direct_weight_sum"],
                    -r["direct_paper_count"],
                    r["fos_name"].lower(),
                )
            )

            out_path = out_dir / f"{expert_id}_direct_fos_nodes.tsv"
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

            sf.write(
                f"{expert_id}\t{len(papers_seen.get(expert_id, set()))}\t"
                f"{len(rows_out)}\t{out_path}\n"
            )

    print(f"experts={len(expert_ids)}")
    print(f"parsed_records={parsed}")
    print(f"matched_papers={matched_papers}")
    print(f"output_dir={out_dir}")
    print(f"summary={summary_path}")


if __name__ == "__main__":
    main()
