#!/usr/bin/env python3
"""Local web demo for task-node role descriptions and expert assignment."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
STATIC_DIR = Path(__file__).resolve().parent / "static"
DEFAULT_EVIDENCE_JSONL = ROOT / "output/hierec_embedding_server_inputs/expert_node_evidence.jsonl"
DEFAULT_NODE_JSONL = ROOT / "output/hierec_embedding_server_inputs/node_texts.jsonl"
DEFAULT_FOS_MAP = ROOT / "data/dblp/FieldsOfStudy.txt"
DEFAULT_FOS_CHILDREN = ROOT / "data/dblp/13.FieldOfStudyChildren.nt"
DEFAULT_PROFILE_DIR = ROOT / "output/expert_profile"
DEFAULT_EXPERT_NAMES = ROOT / "data/dblp/expert_id_name.tsv"
DEFAULT_TASKS_CSV = ROOT / "data_preprocess/teams_2020plus_with_skill_weights.csv"
DEFAULT_METHOD_DIR = ROOT / "output/embedding_bfs_unique_assignment_no_label"
DEFAULT_REQUIREMENT_TEXTS = (
    ROOT / "output/all_expert_paper_embeddings/task_node_requirement_texts_strict_v2.jsonl"
)
DEFAULT_TASK_PROMPTS = ROOT / "output/all_expert_paper_embeddings/task_node_prompts_expertise.jsonl"

MODEL = "openai/gpt-oss-120b"
TOGETHER_CHAT_URL = "https://api.together.xyz/v1/chat/completions"
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "with",
}


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def tokenize(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if len(token) > 2 and token not in STOPWORDS
    }


def load_node_catalog(path: Path, fallback_path: Path) -> dict[str, dict[str, Any]]:
    nodes: dict[str, dict[str, Any]] = {}
    if path.exists():
        for row in read_jsonl(path):
            node_id = str(row.get("node_id") or row.get("id") or "").strip()
            node_name = str(row.get("node_name") or row.get("text") or node_id).strip()
            if not node_id or not node_name:
                continue
            nodes[node_id] = {
                "node_id": node_id,
                "node_name": node_name,
                "node_level": row.get("node_level", ""),
                "tokens": tokenize(node_name),
            }

    if nodes or not fallback_path.exists():
        return nodes

    with fallback_path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                continue
            node_id, _, _, display_name = parts[:4]
            node_id = node_id.strip()
            display_name = display_name.strip()
            if not node_id or not display_name:
                continue
            nodes[node_id] = {
                "node_id": node_id,
                "node_name": display_name,
                "node_level": parts[5] if len(parts) > 5 else "",
                "tokens": tokenize(display_name),
            }
    return nodes


def load_expert_index(path: Path) -> dict[str, list[dict[str, Any]]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing expert evidence JSONL: {path}")

    by_node: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in read_jsonl(path):
        node_id = str(row.get("node_id") or "").strip()
        expert_id = str(row.get("expert_id") or "").strip()
        if not node_id or not expert_id:
            continue
        by_node[node_id].append(
            {
                "expert_id": expert_id,
                "expert_name": str(row.get("expert_name") or expert_id),
                "node_id": node_id,
                "node_name": str(row.get("node_name") or node_id),
                "direct_weight_sum": float(row.get("direct_weight_sum") or 0.0),
                "expert_total_direct_weight": float(row.get("expert_total_direct_weight") or 0.0),
                "papers": row.get("papers") or [],
            }
        )

    for node_id, experts in by_node.items():
        experts.sort(key=lambda x: x["direct_weight_sum"], reverse=True)
        by_node[node_id] = experts[:25]
    return dict(by_node)


def load_expert_names(path: Path) -> dict[str, str]:
    names = {}
    if not path.exists():
        return names
    with path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            expert_id = str(row.get("expert_id") or "").strip()
            if expert_id:
                names[expert_id] = str(row.get("name") or expert_id)
    return names


def load_fos_name_map(path: Path) -> dict[str, dict[str, str]]:
    out = {}
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                continue
            node_id = parts[0].strip()
            raw_name = parts[2].strip()
            display_name = parts[3].strip() or raw_name
            if not node_id or not raw_name:
                continue
            keys = {
                raw_name.lower(),
                raw_name.lower().replace(" ", "_"),
                display_name.lower(),
                display_name.lower().replace(" ", "_"),
            }
            for key in keys:
                out[key] = {"node_id": node_id, "node_name": display_name}
    return out


def load_fos_parent_map(path: Path) -> dict[str, str]:
    parents = {}
    if not path.exists():
        return parents
    pattern = re.compile(r"entity/(\d+)> .*hasParent.*entity/(\d+)>")
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            match = pattern.search(line)
            if match:
                parents[match.group(1)] = match.group(2)
    return parents


def load_profile_index(
    profile_dir: Path,
    expert_names_path: Path,
    max_experts: int,
    max_nodes_per_expert: int,
) -> dict[str, list[dict[str, Any]]]:
    if not profile_dir.exists():
        raise FileNotFoundError(f"Missing expert profile directory: {profile_dir}")

    expert_names = load_expert_names(expert_names_path)
    by_node: dict[str, list[dict[str, Any]]] = defaultdict(list)
    profile_files = sorted(profile_dir.glob("*_direct_fos_nodes.tsv"))
    if max_experts > 0:
        profile_files = profile_files[:max_experts]

    for profile_path in profile_files:
        expert_id = profile_path.name.replace("_direct_fos_nodes.tsv", "")
        expert_name = expert_names.get(expert_id, expert_id)
        with profile_path.open("r", encoding="utf-8") as f:
            for idx, row in enumerate(csv.DictReader(f, delimiter="\t")):
                if max_nodes_per_expert > 0 and idx >= max_nodes_per_expert:
                    break
                node_id = str(row.get("fos_id") or "").strip()
                node_name = str(row.get("fos_name") or node_id).strip()
                if not node_id:
                    continue
                papers = []
                details = str(row.get("paper_weight_details") or "").strip()
                if details:
                    try:
                        papers = json.loads(details)[:3]
                    except json.JSONDecodeError:
                        papers = []
                by_node[node_id].append(
                    {
                        "expert_id": expert_id,
                        "expert_name": expert_name,
                        "node_id": node_id,
                        "node_name": node_name,
                        "direct_weight_sum": float(row.get("direct_weight_sum") or 0.0),
                        "expert_total_direct_weight": 0.0,
                        "papers": papers,
                    }
                )

    for node_id, experts in by_node.items():
        experts.sort(key=lambda x: x["direct_weight_sum"], reverse=True)
        by_node[node_id] = experts[:25]
    return dict(by_node)


def read_profile_nodes(profile_dir: Path, expert_id: str, limit: int = 18) -> list[dict[str, Any]]:
    path = profile_dir / f"{expert_id}_direct_fos_nodes.tsv"
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            node_id = str(row.get("fos_id") or "").strip()
            if not node_id:
                continue
            rows.append(
                {
                    "node_id": node_id,
                    "node_name": str(row.get("fos_name") or node_id),
                    "direct_weight_sum": float(row.get("direct_weight_sum") or 0.0),
                    "direct_paper_count": int(float(row.get("direct_paper_count") or 0)),
                }
            )
    rows.sort(key=lambda x: x["direct_weight_sum"], reverse=True)
    return rows[:limit]


class MethodState:
    variant = "embedding_bfs_unique_assign_each_node_then_top_team_size_by_weighted_score"

    def __init__(
        self,
        demo_state: DemoState,
        tasks_csv: Path,
        method_dir: Path,
        profile_dir: Path,
        requirement_texts: Path,
        task_prompts: Path,
    ):
        self.demo_state = demo_state
        self.tasks_csv = tasks_csv
        self.method_dir = method_dir
        self.profile_dir = profile_dir
        self.requirement_texts = requirement_texts
        self.task_titles = self.load_task_titles(task_prompts)
        self.fos_by_name = load_fos_name_map(DEFAULT_FOS_MAP)
        self.fos_parent = load_fos_parent_map(DEFAULT_FOS_CHILDREN)
        self.metrics = self.load_metrics(method_dir / "metrics_summary.tsv")
        self.tasks = self.load_tasks(tasks_csv)
        self.predictions = self.load_predictions(method_dir / "predictions_team_size.tsv")
        self.assignments = self.load_assignments(method_dir / "node_assignments.tsv")
        self.paper_ids = [pid for pid in self.predictions if pid in self.tasks]
        self.idf = self.build_visual_idf(demo_state.experts_by_node)

    def load_task_titles(self, path: Path) -> dict[str, str]:
        titles = {}
        if not path.exists():
            return titles
        for row in read_jsonl(path):
            paper_id = str(row.get("paper_id") or "")
            if not paper_id or paper_id in titles:
                continue
            text = str(row.get("task_paper_text") or "").strip()
            if not text:
                continue
            titles[paper_id] = text.split(". ", 1)[0].strip()
        return titles

    def load_metrics(self, path: Path) -> dict[str, Any]:
        with path.open("r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f, delimiter="\t"))
        for row in rows:
            if row.get("method") == self.variant or row.get("variant") == self.variant:
                return row
        return rows[0] if rows else {}

    def load_predictions(self, path: Path) -> dict[str, list[dict[str, Any]]]:
        out: dict[str, list[dict[str, Any]]] = defaultdict(list)
        with path.open("r", encoding="utf-8") as f:
            for row in csv.DictReader(f, delimiter="\t"):
                if row.get("method") != self.variant:
                    continue
                paper_id = str(row.get("paper_id") or "")
                row["rank"] = int(row.get("rank") or 0)
                row["score"] = float(row.get("score") or 0.0)
                row["is_actual_member"] = str(row.get("is_actual_member") or "0") == "1"
                out[paper_id].append(row)
        for rows in out.values():
            rows.sort(key=lambda x: x["rank"])
        return dict(out)

    def load_assignments(self, path: Path) -> dict[str, list[dict[str, Any]]]:
        out: dict[str, list[dict[str, Any]]] = defaultdict(list)
        with path.open("r", encoding="utf-8") as f:
            for row in csv.DictReader(f, delimiter="\t"):
                paper_id = str(row.get("paper_id") or "")
                if not paper_id:
                    continue
                row["node_importance"] = float(row.get("node_importance") or 0.0)
                row["node_log_sum"] = float(row.get("node_log_sum") or 0.0)
                row["subtree_skill_count"] = int(float(row.get("subtree_skill_count") or 0))
                row["similarity"] = float(row.get("similarity") or row.get("score") or 0.0)
                row["weighted_score"] = float(row.get("weighted_score") or 0.0)
                row["log_sum_score"] = float(row.get("log_sum_score") or 0.0)
                row["is_actual_member"] = str(row.get("is_actual_member") or "0") == "1"
                out[paper_id].append(row)
        return dict(out)

    def load_tasks(self, path: Path) -> dict[str, dict[str, Any]]:
        out = {}
        with path.open("r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                paper_id = str(row.get("paper_id") or "").strip()
                if not paper_id:
                    continue
                skills = str(row.get("skills") or "").split("|")
                weights = str(row.get("skill_weights") or "").split("|")
                skill_rows = []
                for skill, weight in zip(skills, weights):
                    clean = skill.strip()
                    if not clean:
                        continue
                    weight_value = float(weight or 0.0)
                    fos = self.fos_by_name.get(clean.lower()) or self.fos_by_name.get(
                        clean.lower().replace("_", " ")
                    )
                    skill_rows.append(
                        {
                            "skill": clean.replace("_", " "),
                            "weight": weight_value,
                            "node_id": fos["node_id"] if fos else "",
                            "node_name": fos["node_name"] if fos else clean.replace("_", " "),
                        }
                    )
                members = []
                for member in str(row.get("members") or "").split("|"):
                    if not member:
                        continue
                    parts = member.split("_", 1)
                    members.append(
                        {
                            "expert_id": parts[0],
                            "expert_name": parts[1].replace("_", " ") if len(parts) > 1 else member,
                        }
                    )
                out[paper_id] = {
                    "paper_id": paper_id,
                    "year": row.get("year", ""),
                    "team_size": int(float(row.get("team_size") or 0)),
                    "skills": skill_rows,
                    "members": members,
                }
        return out

    def build_visual_idf(self, experts_by_node: dict[str, list[dict[str, Any]]]) -> dict[str, float]:
        expert_ids = set()
        for experts in experts_by_node.values():
            for expert in experts:
                expert_ids.add(expert["expert_id"])
        n = max(1, len(expert_ids))
        return {
            node_id: math.log((n + 1) / (len(experts) + 1)) + 1.0
            for node_id, experts in experts_by_node.items()
        }

    def lookup_requirement_texts(self, paper_id: str) -> dict[str, dict[str, Any]]:
        out = {}
        if not self.requirement_texts.exists():
            return out
        for row in read_jsonl(self.requirement_texts):
            if str(row.get("paper_id")) != paper_id:
                continue
            node_id = str(row.get("node_id") or "")
            out[node_id] = row
        return out

    def nearest_visible_parent(self, node_id: str, visible_node_ids: set[str]) -> str:
        seen = set()
        parent_id = self.fos_parent.get(node_id, "")
        while parent_id and parent_id not in seen:
            if parent_id in visible_node_ids:
                return parent_id
            seen.add(parent_id)
            parent_id = self.fos_parent.get(parent_id, "")
        return ""

    def overview(self) -> dict[str, Any]:
        tasks = []
        for paper_id in self.paper_ids[:80]:
            task = self.tasks[paper_id]
            predictions = self.predictions.get(paper_id, [])
            hit_count = sum(1 for p in predictions if p["is_actual_member"])
            assigned_count = len(self.assignments.get(paper_id, []))
            title = self.task_titles.get(paper_id, f"Paper {paper_id}")
            tasks.append(
                {
                    "paper_id": paper_id,
                    "title": title,
                    "year": task["year"],
                    "team_size": task["team_size"],
                    "assigned_nodes": assigned_count,
                    "hits": hit_count,
                    "label": title,
                }
            )
        return {
            "variant": self.variant,
            "metrics": self.metrics,
            "tasks": tasks,
            "total_tasks": len(self.paper_ids),
            "sources": {
                "metrics": str(self.method_dir / "metrics_summary.tsv"),
                "predictions": str(self.method_dir / "predictions_team_size.tsv"),
                "assignments": str(self.method_dir / "node_assignments.tsv"),
                "tasks": str(self.tasks_csv),
                "profiles": str(self.profile_dir),
                "role_texts": str(self.requirement_texts),
                "task_prompts": str(DEFAULT_TASK_PROMPTS),
            },
        }

    def task_detail(self, paper_id: str) -> dict[str, Any]:
        if paper_id not in self.tasks:
            raise ValueError(f"Unknown paper_id: {paper_id}")
        task = self.tasks[paper_id]
        actual_member_ids = {member["expert_id"] for member in task["members"]}
        actual_member_names = {
            member["expert_id"]: member["expert_name"] for member in task["members"]
        }
        predictions = self.predictions.get(paper_id, [])
        if not predictions:
            raise ValueError(f"No precomputed predictions for paper_id: {paper_id}")
        requirement_by_node = self.lookup_requirement_texts(paper_id)

        skill_nodes = []
        for skill in task["skills"]:
            node_id = skill["node_id"]
            idf = self.idf.get(node_id, 1.0)
            coeff = skill["weight"] * idf
            text_row = requirement_by_node.get(node_id, {})
            skill_nodes.append(
                {
                    **skill,
                    "idf": round(idf, 4),
                    "task_coeff": round(coeff, 4),
                    "embedding_id": f"{paper_id}::{node_id}" if node_id else "",
                    "role_text": str(text_row.get("text") or "")[:420],
                }
            )

        assignment_rows = []
        for idx, row in enumerate(self.assignments.get(paper_id, []), start=1):
            text_row = requirement_by_node.get(str(row.get("node_id") or ""), {})
            assignment_rows.append(
                {
                    "bfs_rank": idx,
                    "node_id": str(row.get("node_id") or ""),
                    "node_name": str(row.get("node_name") or ""),
                    "node_level": row.get("node_level", ""),
                    "node_importance": round(float(row.get("node_importance") or 0.0), 6),
                    "node_log_sum": round(float(row.get("node_log_sum") or 0.0), 6),
                    "subtree_skill_count": int(row.get("subtree_skill_count") or 0),
                    "expert_id": str(row.get("expert_id") or ""),
                    "expert_name": str(row.get("expert_name") or "").replace("_", " "),
                    "similarity": round(float(row.get("similarity") or 0.0), 6),
                    "weighted_score": round(float(row.get("weighted_score") or 0.0), 6),
                    "log_sum_score": round(float(row.get("log_sum_score") or 0.0), 6),
                    "is_actual_member": str(row.get("expert_id") or "") in actual_member_ids,
                    "embedding_id": f"{paper_id}::{row.get('node_id')}",
                    "role_text": str(text_row.get("text") or "")[:420],
                }
            )
        visible_node_ids = {row["node_id"] for row in assignment_rows if row["node_id"]}
        for row in assignment_rows:
            row["parent_id"] = self.fos_parent.get(row["node_id"], "")
            row["tree_parent_id"] = self.nearest_visible_parent(row["node_id"], visible_node_ids)

        top_predictions = predictions
        selected = top_predictions
        hits = sum(1 for p in selected if p["is_actual_member"])
        all_assignment_hit_ids = sorted(
            {
                row["expert_id"]
                for row in assignment_rows
                if row["expert_id"] and row["expert_id"] in actual_member_ids
            }
        )
        all_assignment_hit_members = [
            {
                "expert_id": expert_id,
                "expert_name": actual_member_names.get(expert_id, expert_id),
            }
            for expert_id in all_assignment_hit_ids
        ]
        expert_details = []
        assignment_by_expert = {row["expert_id"]: row for row in assignment_rows}
        for pred in top_predictions:
            profile_nodes = read_profile_nodes(self.profile_dir, str(pred["expert_id"]))
            components = []
            for node in profile_nodes[:12]:
                idf = self.idf.get(node["node_id"], 1.0)
                coeff = math.log1p(node["direct_weight_sum"]) * idf
                components.append(
                    {
                        **node,
                        "idf": round(idf, 4),
                        "expert_coeff": round(coeff, 4),
                        "embedding_id": f"{pred['expert_id']}::{node['node_id']}",
                        "exact_task_node": any(
                            node["node_id"] == assignment["node_id"] for assignment in assignment_rows
                        ),
                    }
                )
            expert_details.append(
                {
                    **pred,
                    "assigned_node": assignment_by_expert.get(str(pred["expert_id"]), {}),
                    "components": components,
                }
            )

        return {
            "variant": self.variant,
            "paper_id": paper_id,
            "title": self.task_titles.get(paper_id, f"Paper {paper_id}"),
            "task": task,
            "skill_nodes": skill_nodes,
            "direct_nodes": assignment_rows,
            "predictions": top_predictions,
            "selected": selected,
            "expert_details": expert_details,
            "local_metrics": {
                "team_size": task["team_size"],
                "hits_at_team_size": hits,
                "precision_at_team_size": round(hits / max(1, len(selected)), 4),
                "recall_at_team_size": round(hits / max(1, len(task["members"])), 4),
                "all_assignment_selected_experts": len(assignment_rows),
                "all_assignment_hits": len(all_assignment_hit_ids),
                "all_assignment_precision": round(
                    len(all_assignment_hit_ids) / max(1, len(assignment_rows)), 4
                ),
                "all_assignment_recall": round(
                    len(all_assignment_hit_ids) / max(1, len(task["members"])), 4
                ),
                "all_assignment_hit_members": all_assignment_hit_members,
            },
            "formula": {
                "task_vector": "For each task taxonomy node, use the existing no-label task-node embedding.",
                "expert_vector": "For the same taxonomy node, compare against existing no-label expert-node embeddings.",
                "score": "similarity = cosine(task_node_embedding, expert_node_embedding); weighted_score = similarity * node_importance",
                "note": "Nodes are visited in BFS order. Each node assigns the best same-node expert not already used for this task. The final team is the top team_size assigned experts by weighted_score.",
            },
        }


class DemoState:
    def __init__(
        self,
        source: str,
        evidence_jsonl: Path,
        node_jsonl: Path,
        profile_dir: Path,
        expert_names: Path,
        max_experts: int,
        max_nodes_per_expert: int,
    ):
        if source == "jsonl":
            self.experts_by_node = load_expert_index(evidence_jsonl)
        else:
            self.experts_by_node = load_profile_index(
                profile_dir,
                expert_names,
                max_experts,
                max_nodes_per_expert,
            )
        all_nodes = load_node_catalog(node_jsonl, DEFAULT_FOS_MAP)
        self.nodes = {
            node_id: all_nodes.get(
                node_id,
                {
                    "node_id": node_id,
                    "node_name": experts[0]["node_name"],
                    "node_level": "",
                    "tokens": tokenize(experts[0]["node_name"]),
                },
            )
            for node_id, experts in self.experts_by_node.items()
        }
        for node_id, node in self.nodes.items():
            node["global_weight"] = sum(x["direct_weight_sum"] for x in self.experts_by_node[node_id])

    def rank_candidates(self, title: str, abstract: str, limit: int) -> list[dict[str, Any]]:
        text = f"{title}\n{abstract}".strip()
        text_lower = text.lower()
        task_tokens = tokenize(text)
        ranked = []

        for node in self.nodes.values():
            node_tokens = node["tokens"]
            overlap = task_tokens & node_tokens
            phrase_hit = 1 if node["node_name"].lower() in text_lower else 0
            if not overlap and not phrase_hit:
                continue
            level = str(node.get("node_level") or "")
            level_penalty = 0.65 if level in {"0", "1"} else 1.0
            score = (
                len(overlap) * 4.0
                + phrase_hit * 12.0
                + math.log1p(float(node.get("global_weight") or 0.0)) * 0.15
            ) * level_penalty
            ranked.append((score, node))

        if len(ranked) < min(limit, 12):
            existing = {node["node_id"] for _, node in ranked}
            fallback = sorted(
                (n for n in self.nodes.values() if n["node_id"] not in existing),
                key=lambda x: float(x.get("global_weight") or 0.0),
                reverse=True,
            )
            ranked.extend((0.1, node) for node in fallback[: limit])

        ranked.sort(key=lambda x: x[0], reverse=True)
        return [
            {
                "node_id": node["node_id"],
                "node_name": node["node_name"],
                "node_level": node.get("node_level", ""),
                "local_score": round(score, 4),
            }
            for score, node in ranked[:limit]
        ]

    def assign_experts(self, node_id: str, limit: int) -> list[dict[str, Any]]:
        out = []
        for expert in self.experts_by_node.get(node_id, [])[:limit]:
            papers = expert.get("papers") or []
            out.append(
                {
                    "expert_id": expert["expert_id"],
                    "expert_name": expert["expert_name"].replace("_", " "),
                    "score": round(expert["direct_weight_sum"], 4),
                    "evidence_papers": papers[:3],
                }
            )
        return out


def extract_json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def build_messages(title: str, abstract: str, candidates: list[dict[str, Any]], max_nodes: int):
    candidate_text = "\n".join(
        f"- node_id={c['node_id']} | node_name={c['node_name']} | level={c.get('node_level', '')}"
        for c in candidates
    )
    return [
        {
            "role": "system",
            "content": (
                "You analyze a paper abstract for expert team formation. Select the "
                "most relevant taxonomy nodes from the supplied candidate list and "
                "write node-specific expert role descriptions. Return only JSON."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Paper title:\n{title or '(not provided)'}\n\n"
                f"Paper abstract:\n{abstract}\n\n"
                f"Candidate taxonomy nodes:\n{candidate_text}\n\n"
                f"Select 3 to {max_nodes} relevant nodes. Use only node_id values from "
                "the candidate list. Return strict JSON in this shape:\n"
                '{"nodes":[{"node_id":"...","node_name":"...","role_description":"...",'
                '"key_capabilities":["..."],"evidence_from_abstract":["..."]}]}'
            ),
        },
    ]


def call_together(api_key: str, messages: list[dict[str, str]], temperature: float) -> dict[str, Any]:
    try:
        from together import Together  # type: ignore

        client = Together(api_key=api_key)
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=temperature,
        )
        content = response.choices[0].message.content
        return extract_json_object(content)
    except ImportError:
        payload = {
            "model": MODEL,
            "messages": messages,
            "temperature": temperature,
        }
        req = urllib.request.Request(
            TOGETHER_CHAT_URL,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=90) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        content = data["choices"][0]["message"]["content"]
        return extract_json_object(content)


def normalize_llm_nodes(
    llm_obj: dict[str, Any],
    candidates: list[dict[str, Any]],
    state: DemoState,
    experts_per_node: int,
) -> list[dict[str, Any]]:
    candidate_by_id = {str(c["node_id"]): c for c in candidates}
    out = []
    seen = set()
    for item in llm_obj.get("nodes") or []:
        node_id = str(item.get("node_id") or "").strip()
        if not node_id or node_id in seen or node_id not in candidate_by_id:
            continue
        seen.add(node_id)
        candidate = candidate_by_id[node_id]
        role_description = str(item.get("role_description") or item.get("requirement") or "").strip()
        if not role_description:
            role_description = f"Provide expertise in {candidate['node_name']} for this paper."
        out.append(
            {
                "node_id": node_id,
                "node_name": candidate["node_name"],
                "node_level": candidate.get("node_level", ""),
                "role_description": role_description,
                "key_capabilities": [
                    str(x) for x in item.get("key_capabilities", []) if str(x).strip()
                ],
                "evidence_from_abstract": [
                    str(x) for x in item.get("evidence_from_abstract", []) if str(x).strip()
                ],
                "assigned_experts": state.assign_experts(node_id, experts_per_node),
            }
        )
    return out


class Handler(SimpleHTTPRequestHandler):
    state: DemoState
    method_state: MethodState

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(STATIC_DIR), **kwargs)

    def log_message(self, fmt: str, *args):
        sys.stderr.write("[%s] %s\n" % (time.strftime("%H:%M:%S"), fmt % args))

    def send_json(self, status: int, obj: dict[str, Any]):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/api/method/overview":
            self.send_json(200, self.method_state.overview())
            return
        if self.path.startswith("/api/method/task"):
            try:
                query = self.path.split("?", 1)[1] if "?" in self.path else ""
                params = {}
                for part in query.split("&"):
                    if "=" not in part:
                        continue
                    key, value = part.split("=", 1)
                    params[key] = urllib.parse.unquote(value)
                paper_id = params.get("paper_id") or self.method_state.paper_ids[0]
                self.send_json(200, self.method_state.task_detail(paper_id))
            except Exception as exc:
                self.send_json(500, {"error": str(exc)})
            return
        if self.path == "/api/metadata":
            self.send_json(
                200,
                {
                    "model": MODEL,
                    "nodes": len(self.state.nodes),
                    "expert_node_pairs": sum(len(v) for v in self.state.experts_by_node.values()),
                    "has_api_key": bool(os.environ.get("TOGETHER_API_KEY")),
                },
            )
            return
        if self.path == "/":
            self.path = "/index.html"
        super().do_GET()

    def do_POST(self):
        if self.path != "/api/analyze":
            self.send_json(404, {"error": "Unknown endpoint"})
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            title = str(payload.get("title") or "").strip()
            abstract = str(payload.get("abstract") or "").strip()
            if len(abstract) < 40:
                self.send_json(400, {"error": "Please provide a longer paper abstract."})
                return

            max_nodes = max(3, min(int(payload.get("max_nodes") or 6), 10))
            candidate_count = max(12, min(int(payload.get("candidate_count") or 35), 80))
            experts_per_node = max(1, min(int(payload.get("experts_per_node") or 3), 8))
            temperature = float(payload.get("temperature") or 0.0)

            candidates = self.state.rank_candidates(title, abstract, candidate_count)
            api_key = os.environ.get("TOGETHER_API_KEY", "")
            if not api_key:
                self.send_json(
                    500,
                    {"error": "Missing TOGETHER_API_KEY in the server environment."},
                )
                return

            messages = build_messages(title, abstract, candidates, max_nodes)
            llm_obj = call_together(api_key, messages, temperature)
            nodes = normalize_llm_nodes(llm_obj, candidates, self.state, experts_per_node)
            if not nodes:
                self.send_json(
                    502,
                    {"error": "LLM did not return any valid candidate node assignments."},
                )
                return

            self.send_json(
                200,
                {
                    "model": MODEL,
                    "candidate_count": len(candidates),
                    "nodes": nodes[:max_nodes],
                },
            )
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            self.send_json(502, {"error": f"Together API HTTP {exc.code}: {detail}"})
        except Exception as exc:
            self.send_json(500, {"error": str(exc)})


def parse_args():
    p = argparse.ArgumentParser(description="Run the local role-assignment web demo")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument(
        "--source",
        choices=("profiles", "jsonl"),
        default="profiles",
        help="profiles starts quickly; jsonl loads the full HieRec evidence JSONL.",
    )
    p.add_argument("--evidence-jsonl", default=str(DEFAULT_EVIDENCE_JSONL))
    p.add_argument("--node-jsonl", default=str(DEFAULT_NODE_JSONL))
    p.add_argument("--profile-dir", default=str(DEFAULT_PROFILE_DIR))
    p.add_argument("--expert-names", default=str(DEFAULT_EXPERT_NAMES))
    p.add_argument("--tasks-csv", default=str(DEFAULT_TASKS_CSV))
    p.add_argument("--method-dir", default=str(DEFAULT_METHOD_DIR))
    p.add_argument("--requirement-texts", default=str(DEFAULT_REQUIREMENT_TEXTS))
    p.add_argument("--task-prompts", default=str(DEFAULT_TASK_PROMPTS))
    p.add_argument("--max-experts", type=int, default=1200)
    p.add_argument("--max-nodes-per-expert", type=int, default=80)
    return p.parse_args()


def main():
    args = parse_args()
    Handler.state = DemoState(
        args.source,
        Path(args.evidence_jsonl),
        Path(args.node_jsonl),
        Path(args.profile_dir),
        Path(args.expert_names),
        args.max_experts,
        args.max_nodes_per_expert,
    )
    Handler.method_state = MethodState(
        Handler.state,
        Path(args.tasks_csv),
        Path(args.method_dir),
        Path(args.profile_dir),
        Path(args.requirement_texts),
        Path(args.task_prompts),
    )
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Role assignment demo: http://{args.host}:{args.port}")
    print(f"Loaded nodes={len(Handler.state.nodes)}")
    print(f"Loaded method tasks={len(Handler.method_state.paper_ids)}")
    print(f"Together API key present={bool(os.environ.get('TOGETHER_API_KEY'))}")
    server.serve_forever()


if __name__ == "__main__":
    main()
