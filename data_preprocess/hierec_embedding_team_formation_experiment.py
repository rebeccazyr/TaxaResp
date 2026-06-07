#!/usr/bin/env python3
"""HieRec-style embedding team formation experiment.

This is a local prototype for the fixed-budget setting:
1. Build expert-node embeddings from historical FoS evidence papers.
2. Build task-node embeddings from LLM-generated node requirements. The script
   can export task-node prompts, but does not call an LLM/API itself.
3. Assign one best expert to every task taxonomy node.
4. Deduplicate assigned experts, rank by node match score, and keep exactly the
   ground-truth team size when possible.

No external LLM/API is called here. TF-IDF + SVD is used as the local embedding
stand-in so task requirements and expert evidence are generated in the same
vector space.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from build_hierec_expert_node_embeddings import (  # noqa: E402
    build_node_embeddings_for_expert,
    load_requested_paper_texts,
    read_profile,
)
from taxonomy_team_formation_experiment import (  # noqa: E402
    DEFAULT_EXPERTS,
    DEFAULT_FOS_CHILDREN,
    DEFAULT_FOS_MAP,
    DEFAULT_PROFILE_DIR,
    DEFAULT_TASKS,
    ancestor_cache_builder,
    build_task_subtree_vectors,
    load_child_to_parents,
    load_expert_names,
    load_fos_map,
    load_tasks,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run fixed-budget HieRec-style embedding team formation"
    )
    p.add_argument("--tasks-csv", default=DEFAULT_TASKS)
    p.add_argument("--profile-dir", default=DEFAULT_PROFILE_DIR)
    p.add_argument("--expert-tsv", default=DEFAULT_EXPERTS)
    p.add_argument("--fos-map", default=DEFAULT_FOS_MAP)
    p.add_argument("--fos-children", default=DEFAULT_FOS_CHILDREN)
    p.add_argument("--dblp-json", default="data/dblp/dblp.v12.json")
    p.add_argument(
        "--out-dir",
        default="output/hierec_embedding_team_formation_experiment",
    )
    p.add_argument("--max-experts", type=int, default=20)
    p.add_argument("--max-profile-nodes", type=int, default=40)
    p.add_argument("--ancestor-depth", type=int, default=5)
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--max-evidence-papers-per-node", type=int, default=3)
    p.add_argument("--paper-text-max-chars", type=int, default=1200)
    p.add_argument("--progress-every", type=int, default=500000)
    p.add_argument(
        "--export-task-node-prompts",
        default="",
        help=(
            "Write JSONL prompts for LLM task-node requirement generation and exit. "
            "Each row contains paper_id, node_id, node context, and prompt."
        ),
    )
    p.add_argument(
        "--task-node-requirements-jsonl",
        default="",
        help=(
            "JSONL containing LLM outputs with paper_id, node_id, and requirement. "
            "Required for evaluation unless --allow-template-task-text is set."
        ),
    )
    p.add_argument(
        "--allow-template-task-text",
        action="store_true",
        help="Fallback to a non-LLM skill-list template when no requirement is available",
    )
    p.add_argument("--top-k-output", type=int, default=20)
    return p.parse_args()


def l2(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom <= 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def skill_list_text(subtree_vec: Dict[str, float], id_to_name: Dict[str, str]) -> str:
    return "; ".join(
        sorted(
            (
                f"{id_to_name.get(skill_id, skill_id)}:{weight:.4f}"
                for skill_id, weight in subtree_vec.items()
            ),
            key=str.lower,
        )
    )


def all_task_skill_text(task: dict) -> str:
    return "; ".join(
        f"{name}:{weight:.4f}" for _, weight, name in task.get("direct", [])
    )


def template_task_node_text(
    task: dict,
    node_id: str,
    subtree_vec: Dict[str, float],
    id_to_name: Dict[str, str],
) -> str:
    return " ".join(
        [
        f"Task taxonomy node: {id_to_name.get(node_id, node_id)}.",
            "Subtree target skills: " + skill_list_text(subtree_vec, id_to_name) + ".",
            "All task target skills: " + all_task_skill_text(task) + ".",
        ]
    )


def build_task_node_prompt(
    task: dict,
    node_id: str,
    subtree_vec: Dict[str, float],
    id_to_name: Dict[str, str],
    task_paper_text: str,
) -> str:
    node_name = id_to_name.get(node_id, node_id)
    subtree_skills = skill_list_text(subtree_vec, id_to_name)
    all_skills = all_task_skill_text(task)
    return (
        "You are analyzing a research paper as a team-formation task.\n"
        "Given the paper abstract and one node in the task taxonomy, describe "
        "what capability/work this task requires under that taxonomy node. "
        "Use the paper abstract as the main source of context; use the node "
        "name and subtree skills only to locate the relevant aspect of the task.\n\n"
        f"Paper id: {task.get('paper_id', '')}\n"
        f"Task paper title/abstract:\n{task_paper_text}\n\n"
        f"Taxonomy node: {node_name} (id={node_id})\n"
        f"Subtree target skills under this node: {subtree_skills}\n"
        f"All target skills for the task: {all_skills}\n\n"
        "Return strict JSON with these fields:\n"
        "{\n"
        '  "paper_id": "...",\n'
        '  "node_id": "...",\n'
        '  "requirement": "One concise paragraph describing what this task needs under this node.",\n'
        '  "key_capabilities": ["short capability phrase", "..."],\n'
        '  "evidence_from_abstract": ["short phrase from the abstract or title", "..."]\n'
        "}\n"
        "Do not describe generic field knowledge. Describe only what this specific task needs."
    )


def load_task_node_requirements(path: Path) -> Dict[Tuple[str, str], str]:
    requirements: Dict[Tuple[str, str], str] = {}
    if not path:
        return requirements
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            paper_id = str(obj.get("paper_id", "")).strip()
            node_id = str(obj.get("node_id", "")).strip()
            requirement = str(
                obj.get("requirement")
                or obj.get("task_node_requirement")
                or obj.get("text")
                or ""
            ).strip()
            if paper_id and node_id and requirement:
                requirements[(paper_id, node_id)] = requirement
    return requirements


def mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


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
        for t in load_tasks(Path(args.tasks_csv), name_to_id, id_to_name, 0.0)
        if t["direct"] and t["team_size"] > 0
    ]

    profile_files = [
        p
        for p in sorted(Path(args.profile_dir).glob("*_direct_fos_nodes.tsv"))
        if not p.name.startswith("_")
    ][: args.max_experts]

    expert_direct: Dict[str, List[Tuple[str, str, float]]] = {}
    expert_evidence: Dict[str, Dict[str, List[dict]]] = {}
    all_node_ids = set()
    requested_paper_ids = set()

    for path in profile_files:
        expert_id = path.name.replace("_direct_fos_nodes.tsv", "")
        if expert_id not in expert_names:
            continue
        direct_items, direct_evidence = read_profile(
            path, args.max_profile_nodes, args.max_evidence_papers_per_node
        )
        expert_direct[expert_id] = direct_items
        expert_evidence[expert_id] = direct_evidence
        if not args.export_task_node_prompts:
            for papers in direct_evidence.values():
                for paper in papers:
                    requested_paper_ids.add(str(paper.get("paper_id")))
        for fos_id, _, _ in direct_items:
            for node_id, _ in ancestors(fos_id):
                all_node_ids.add(node_id)

    task_subtrees_by_paper: Dict[str, Dict[str, Dict[str, float]]] = {}
    task_node_texts: Dict[Tuple[str, str], str] = {}
    task_paper_ids = {str(t["paper_id"]) for t in tasks if t.get("paper_id")}
    if args.export_task_node_prompts:
        requested_paper_ids.update(task_paper_ids)

    paper_texts = load_requested_paper_texts(
        Path(args.dblp_json),
        requested_paper_ids,
        args.paper_text_max_chars,
        args.progress_every,
    )
    print(f"paper_texts_loaded={len(paper_texts)}/{len(requested_paper_ids)}")

    requirements = (
        load_task_node_requirements(Path(args.task_node_requirements_jsonl))
        if args.task_node_requirements_jsonl
        else {}
    )
    prompt_rows = []
    missing_requirement_count = 0

    for task in tasks:
        subtrees = build_task_subtree_vectors(task, ancestors)
        task_subtrees_by_paper[task["paper_id"]] = subtrees
        task_paper_text = paper_texts.get(str(task["paper_id"]), "")
        for node_id, subtree_vec in subtrees.items():
            all_node_ids.add(node_id)
            key = (task["paper_id"], node_id)
            if args.export_task_node_prompts:
                prompt_rows.append(
                    {
                        "paper_id": task["paper_id"],
                        "node_id": node_id,
                        "node_name": id_to_name.get(node_id, node_id),
                        "node_level": id_to_level.get(node_id, ""),
                        "subtree_skill_count": len(subtree_vec),
                        "subtree_skills": skill_list_text(subtree_vec, id_to_name),
                        "all_task_skills": all_task_skill_text(task),
                        "task_paper_text": task_paper_text,
                        "prompt": build_task_node_prompt(
                            task, node_id, subtree_vec, id_to_name, task_paper_text
                        ),
                    }
                )
                continue

            if key in requirements:
                task_node_texts[key] = requirements[key]
            elif args.allow_template_task_text:
                task_node_texts[key] = template_task_node_text(
                    task, node_id, subtree_vec, id_to_name
                )
            else:
                missing_requirement_count += 1

    if args.export_task_node_prompts:
        prompt_path = Path(args.export_task_node_prompts)
        prompt_path.parent.mkdir(parents=True, exist_ok=True)
        with prompt_path.open("w", encoding="utf-8") as f:
            for row in prompt_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"task_node_prompts={prompt_path}")
        print(f"prompt_rows={len(prompt_rows)}")
        return

    if missing_requirement_count:
        raise SystemExit(
            "Missing LLM task-node requirements for "
            f"{missing_requirement_count} nodes. Provide --task-node-requirements-jsonl "
            "or explicitly pass --allow-template-task-text for the old non-LLM fallback."
        )

    if not task_node_texts:
        raise SystemExit(
            "No task-node requirement texts available. First run with "
            "--export-task-node-prompts, call an LLM on those prompts, then pass "
            "--task-node-requirements-jsonl."
        )

    node_ids = sorted(all_node_ids)
    node_texts = [id_to_name.get(node_id, node_id).replace("_", " ") for node_id in node_ids]
    expert_paper_ids = sorted(pid for pid in requested_paper_ids if pid in paper_texts)
    task_text_keys = sorted(task_node_texts)

    corpus = (
        node_texts
        + [paper_texts[pid] for pid in expert_paper_ids]
        + [task_node_texts[key] for key in task_text_keys]
    )
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), analyzer="word", min_df=1)
    tfidf = vectorizer.fit_transform(corpus)
    dim = min(args.dim, max(2, min(tfidf.shape) - 1))
    dense = normalize(TruncatedSVD(n_components=dim, random_state=0).fit_transform(tfidf))

    semantic = {node_id: dense[i] for i, node_id in enumerate(node_ids)}
    paper_offset = len(node_ids)
    paper_vectors = {
        pid: dense[paper_offset + i] for i, pid in enumerate(expert_paper_ids)
    }
    task_offset = paper_offset + len(expert_paper_ids)
    task_text_vectors = {
        key: dense[task_offset + i] for i, key in enumerate(task_text_keys)
    }

    expert_node_vectors: Dict[str, List[Tuple[str, np.ndarray, dict]]] = defaultdict(list)
    for expert_id, direct_items in expert_direct.items():
        embeddings = build_node_embeddings_for_expert(
            direct_items,
            expert_evidence.get(expert_id, {}),
            ancestors,
            child_to_parents,
            semantic,
            paper_vectors,
        )
        for node_id, rec in embeddings.items():
            expert_node_vectors[node_id].append((expert_id, rec["embedding"], rec))

    metric_rows = []
    assignment_rows = []
    prediction_rows = []
    macro_p = []
    macro_r = []
    micro_hits = 0
    micro_selected = 0
    micro_positives = 0
    assigned_node_counts = []
    unique_assigned_counts = []
    selected_counts = []

    for task in tasks:
        positives = set(task["members"])
        node_assignments = []
        for node_id, subtree_vec in task_subtrees_by_paper[task["paper_id"]].items():
            candidates = expert_node_vectors.get(node_id, [])
            if not candidates:
                continue
            task_vec = l2(semantic[node_id] + task_text_vectors[(task["paper_id"], node_id)])
            best = None
            for expert_id, expert_vec, rec in candidates:
                score = cosine(task_vec, expert_vec)
                if best is None or score > best[1]:
                    best = (expert_id, score, rec)
            if best is None:
                continue
            expert_id, score, rec = best
            node_importance = sum(subtree_vec.values())
            weighted_score = score * node_importance
            node_assignments.append((node_id, expert_id, score, weighted_score, node_importance))
            assignment_rows.append(
                {
                    "paper_id": task["paper_id"],
                    "node_id": node_id,
                    "node_name": id_to_name.get(node_id, node_id),
                    "node_level": id_to_level.get(node_id, ""),
                    "subtree_skill_count": len(subtree_vec),
                    "node_importance": f"{node_importance:.6f}",
                    "expert_id": expert_id,
                    "expert_name": expert_names.get(expert_id, expert_id),
                    "score": f"{score:.6f}",
                    "weighted_score": f"{weighted_score:.6f}",
                    "is_actual_member": "1" if expert_id in positives else "0",
                }
            )

        assigned_node_counts.append(len(node_assignments))
        best_by_expert: Dict[str, Tuple[float, str, float]] = {}
        for node_id, expert_id, score, weighted_score, _ in node_assignments:
            prev = best_by_expert.get(expert_id)
            if prev is None or weighted_score > prev[0]:
                best_by_expert[expert_id] = (weighted_score, node_id, score)
        ranked = sorted(best_by_expert.items(), key=lambda x: x[1][0], reverse=True)
        unique_assigned_counts.append(len(ranked))
        selected = ranked[: max(1, task["team_size"])]
        selected_ids = [expert_id for expert_id, _ in selected]
        selected_counts.append(len(selected_ids))

        hits = len(positives.intersection(selected_ids))
        precision = hits / len(selected_ids) if selected_ids else 0.0
        recall = hits / len(positives) if positives else 0.0
        macro_p.append(precision)
        macro_r.append(recall)
        micro_hits += hits
        micro_selected += len(selected_ids)
        micro_positives += len(positives)

        for rank, (expert_id, (weighted_score, node_id, raw_score)) in enumerate(
            selected[: args.top_k_output], start=1
        ):
            prediction_rows.append(
                {
                    "paper_id": task["paper_id"],
                    "rank": rank,
                    "expert_id": expert_id,
                    "expert_name": expert_names.get(expert_id, expert_id),
                    "score": f"{weighted_score:.6f}",
                    "raw_node_score": f"{raw_score:.6f}",
                    "best_node_id": node_id,
                    "best_node_name": id_to_name.get(node_id, node_id),
                    "is_actual_member": "1" if expert_id in positives else "0",
                }
            )

    metric_rows.append(
        {
            "method": "hierec_embedding_node_assign_then_team_size_cut",
            "tasks": len(tasks),
            "experts": len(expert_direct),
            "embedding_dim": dim,
            "task_requirement_source": (
                "llm_jsonl" if args.task_node_requirements_jsonl else "template_fallback"
            ),
            "avg_assigned_nodes": f"{mean(assigned_node_counts):.6f}",
            "avg_unique_assigned_experts": f"{mean(unique_assigned_counts):.6f}",
            "avg_selected_experts": f"{mean(selected_counts):.6f}",
            "macro_precision_at_team_size": f"{mean(macro_p):.6f}",
            "macro_recall_at_team_size": f"{mean(macro_r):.6f}",
            "micro_precision_at_team_size": f"{(micro_hits / micro_selected) if micro_selected else 0.0:.6f}",
            "micro_recall_at_team_size": f"{(micro_hits / micro_positives) if micro_positives else 0.0:.6f}",
            "requested_paper_texts": len(requested_paper_ids),
            "loaded_paper_texts": len(paper_texts),
        }
    )

    metrics_path = out_dir / "metrics_summary.tsv"
    with metrics_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(metric_rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(metric_rows)

    assignments_path = out_dir / "node_assignments.tsv"
    with assignments_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "paper_id",
            "node_id",
            "node_name",
            "node_level",
            "subtree_skill_count",
            "node_importance",
            "expert_id",
            "expert_name",
            "score",
            "weighted_score",
            "is_actual_member",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(assignment_rows)

    predictions_path = out_dir / "predictions_team_size.tsv"
    with predictions_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "paper_id",
            "rank",
            "expert_id",
            "expert_name",
            "score",
            "raw_node_score",
            "best_node_id",
            "best_node_name",
            "is_actual_member",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(prediction_rows)

    print(f"tasks={len(tasks)}")
    print(f"experts={len(expert_direct)}")
    print(f"embedding_dim={dim}")
    print(f"metrics={metrics_path}")
    print(f"assignments={assignments_path}")
    print(f"predictions={predictions_path}")
    print(
        "macro_p={:.4f}% macro_r={:.4f}%".format(
            100 * mean(macro_p), 100 * mean(macro_r)
        )
    )


if __name__ == "__main__":
    main()
