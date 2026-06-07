#!/usr/bin/env python3
"""Prepare server-side inputs for HieRec embedding team formation.

Outputs under --out-dir:
- paper_texts.jsonl: paper_id/title/abstract/text for expert evidence and tasks.
- expert_node_evidence.jsonl: direct expert FoS nodes and evidence paper ids.
- node_texts.jsonl: taxonomy node labels to embed.
- task_nodes.jsonl: task taxonomy nodes and subtree skill context.
- task_node_prompts.jsonl: prompts for LLM task-node requirement generation.
- summary.tsv: counts.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

from embedding_pipeline_utils import (
    ancestor_cache_builder,
    build_task_subtree_vectors,
    decode_indexed_abstract,
    iter_json_objects,
    load_child_to_parents,
    load_fos_map,
    load_tasks,
    read_profile_evidence,
    write_jsonl,
)


DEFAULT_TASKS = "data_preprocess/teams_2020plus_with_skill_weights.csv"
DEFAULT_PROFILE_DIR = "output/expert_profile_year_bins/all_2000_2019"
DEFAULT_EXPERTS = "data/dblp/expert_id_name.tsv"
DEFAULT_FOS_MAP = "data/dblp/FieldsOfStudy.txt"
DEFAULT_FOS_CHILDREN = "data/dblp/13.FieldOfStudyChildren.nt"
DEFAULT_DBLP = "data/dblp/dblp.v12.json"
DEFAULT_OUT = "output/hierec_embedding_server_inputs"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare HieRec embedding input caches")
    p.add_argument("--tasks-csv", default=DEFAULT_TASKS)
    p.add_argument("--profile-dir", default=DEFAULT_PROFILE_DIR)
    p.add_argument("--expert-tsv", default=DEFAULT_EXPERTS)
    p.add_argument("--fos-map", default=DEFAULT_FOS_MAP)
    p.add_argument("--fos-children", default=DEFAULT_FOS_CHILDREN)
    p.add_argument("--dblp-json", default=DEFAULT_DBLP)
    p.add_argument("--out-dir", default=DEFAULT_OUT)
    p.add_argument("--max-experts", type=int, default=0, help="0 means all experts")
    p.add_argument("--max-profile-nodes", type=int, default=120)
    p.add_argument("--max-evidence-papers-per-node", type=int, default=5)
    p.add_argument("--ancestor-depth", type=int, default=5)
    p.add_argument("--paper-text-max-chars", type=int, default=2000)
    p.add_argument("--progress-every", type=int, default=500000)
    return p.parse_args()


def load_expert_names(path: Path) -> dict:
    out = {}
    with path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            expert_id = (row.get("expert_id") or "").strip()
            if expert_id:
                out[expert_id] = row.get("name") or expert_id
    return out


def skill_list_text(subtree_vec: dict, id_to_name: dict) -> str:
    return "; ".join(
        sorted(
            f"{id_to_name.get(skill_id, skill_id)}:{weight:.4f}"
            for skill_id, weight in subtree_vec.items()
        )
    )


def all_task_skill_text(task: dict) -> str:
    return "; ".join(f"{name}:{weight:.4f}" for _, weight, name in task["direct"])


def build_task_node_prompt(task: dict, node_row: dict) -> str:
    return (
        "You are analyzing a research paper as a team-formation task.\n"
        "Given the paper title/abstract and one node in the task taxonomy, "
        "describe what capability/work this task requires under that taxonomy node. "
        "Use the paper title/abstract as the main source of context; use the node "
        "name and subtree skills only to locate the relevant aspect of the task.\n\n"
        f"Paper id: {task['paper_id']}\n"
        f"Task paper title/abstract:\n{node_row.get('task_paper_text', '')}\n\n"
        f"Taxonomy node: {node_row['node_name']} (id={node_row['node_id']})\n"
        f"Subtree target skills under this node: {node_row['subtree_skills']}\n"
        f"All target skills for the task: {node_row['all_task_skills']}\n\n"
        "Return strict JSON with fields: paper_id, node_id, requirement, "
        "key_capabilities, evidence_from_abstract. The requirement should be one "
        "concise paragraph about this specific task, not generic field knowledge."
    )


def scan_paper_texts(
    dblp_json: Path,
    requested_ids: set,
    max_chars: int,
    progress_every: int,
) -> dict:
    found = {}
    parsed = 0
    for obj in iter_json_objects(dblp_json):
        parsed += 1
        if progress_every > 0 and parsed % progress_every == 0:
            print(f"paper_text_progress parsed={parsed:,} found={len(found):,}/{len(requested_ids):,}")
        paper_id = str(obj.get("id", ""))
        if paper_id not in requested_ids:
            continue
        title = str(obj.get("title") or "").strip()
        abstract = decode_indexed_abstract(obj.get("indexed_abstract"))
        text = re.sub(r"\s+", " ", f"{title}. {abstract}".strip())
        if max_chars > 0:
            text = text[:max_chars]
        found[paper_id] = {
            "paper_id": paper_id,
            "year": obj.get("year", ""),
            "title": title,
            "abstract": abstract,
            "text": text,
        }
        if len(found) >= len(requested_ids):
            break
    return found


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    expert_names = load_expert_names(Path(args.expert_tsv))
    name_to_id, id_to_name, id_to_level = load_fos_map(Path(args.fos_map))
    child_to_parents = load_child_to_parents(Path(args.fos_children))
    ancestors = ancestor_cache_builder(child_to_parents, args.ancestor_depth)
    tasks = [
        t
        for t in load_tasks(Path(args.tasks_csv), name_to_id, id_to_name)
        if t["direct"] and t["team_size"] > 0
    ]

    requested_paper_ids = {str(t["paper_id"]) for t in tasks if t.get("paper_id")}
    node_ids = set()
    expert_evidence_rows = []

    profile_files = [
        p
        for p in sorted(Path(args.profile_dir).glob("*_direct_fos_nodes.tsv"))
        if not p.name.startswith("_")
    ]
    if args.max_experts > 0:
        profile_files = profile_files[: args.max_experts]

    for profile_path in profile_files:
        expert_id = profile_path.name.replace("_direct_fos_nodes.tsv", "")
        direct_rows = read_profile_evidence(
            profile_path,
            args.max_profile_nodes,
            args.max_evidence_papers_per_node,
        )
        total_direct_weight = sum(r["direct_weight_sum"] for r in direct_rows)
        for row in direct_rows:
            node_ids.add(row["node_id"])
            for parent_id, _ in ancestors(row["node_id"]):
                node_ids.add(parent_id)
            for paper in row["papers"]:
                requested_paper_ids.add(str(paper["paper_id"]))
            expert_evidence_rows.append(
                {
                    "expert_id": expert_id,
                    "expert_name": expert_names.get(expert_id, expert_id),
                    "node_id": row["node_id"],
                    "node_name": row["node_name"],
                    "direct_weight_sum": row["direct_weight_sum"],
                    "expert_total_direct_weight": total_direct_weight,
                    "papers": row["papers"],
                }
            )

    task_node_rows = []
    for task in tasks:
        subtrees = build_task_subtree_vectors(task, ancestors)
        for node_id, subtree_vec in subtrees.items():
            node_ids.add(node_id)
            row = {
                "paper_id": task["paper_id"],
                "team_size": task["team_size"],
                "members": task["members"],
                "node_id": node_id,
                "node_name": id_to_name.get(node_id, node_id),
                "node_level": id_to_level.get(node_id, ""),
                "subtree_skill_count": len(subtree_vec),
                "node_importance": sum(subtree_vec.values()),
                "subtree_skills": skill_list_text(subtree_vec, id_to_name),
                "all_task_skills": all_task_skill_text(task),
            }
            task_node_rows.append(row)

    paper_texts = scan_paper_texts(
        Path(args.dblp_json),
        requested_paper_ids,
        args.paper_text_max_chars,
        args.progress_every,
    )
    print(f"paper_texts_loaded={len(paper_texts)}/{len(requested_paper_ids)}")

    for row in task_node_rows:
        row["task_paper_text"] = paper_texts.get(str(row["paper_id"]), {}).get("text", "")
        task = next(t for t in tasks if t["paper_id"] == row["paper_id"])
        row["prompt"] = build_task_node_prompt(task, row)

    node_rows = [
        {
            "id": node_id,
            "node_id": node_id,
            "node_name": id_to_name.get(node_id, node_id),
            "node_level": id_to_level.get(node_id, ""),
            "text": id_to_name.get(node_id, node_id),
        }
        for node_id in sorted(node_ids)
    ]

    paper_rows = [paper_texts[pid] for pid in sorted(paper_texts)]
    for row in paper_rows:
        row["id"] = row["paper_id"]

    write_jsonl(out_dir / "paper_texts.jsonl", paper_rows)
    write_jsonl(out_dir / "expert_node_evidence.jsonl", expert_evidence_rows)
    write_jsonl(out_dir / "node_texts.jsonl", node_rows)
    write_jsonl(out_dir / "task_nodes.jsonl", task_node_rows)
    write_jsonl(out_dir / "task_node_prompts.jsonl", task_node_rows)

    with (out_dir / "summary.tsv").open("w", encoding="utf-8") as f:
        f.write("metric\tvalue\n")
        f.write(f"tasks\t{len(tasks)}\n")
        f.write(f"task_nodes\t{len(task_node_rows)}\n")
        f.write(f"experts\t{len(profile_files)}\n")
        f.write(f"expert_direct_nodes\t{len(expert_evidence_rows)}\n")
        f.write(f"taxonomy_nodes\t{len(node_rows)}\n")
        f.write(f"requested_papers\t{len(requested_paper_ids)}\n")
        f.write(f"loaded_papers\t{len(paper_rows)}\n")

    print(f"out_dir={out_dir}")
    print(f"task_node_prompts={out_dir / 'task_node_prompts.jsonl'}")
    print(f"paper_texts={out_dir / 'paper_texts.jsonl'}")
    print(f"summary={out_dir / 'summary.tsv'}")


if __name__ == "__main__":
    main()
