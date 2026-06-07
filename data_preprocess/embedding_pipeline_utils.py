#!/usr/bin/env python3
"""Shared helpers for the HieRec embedding team-formation pipeline."""

from __future__ import annotations

import csv
import json
import math
import re
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


def safe_float(v: object, default: float = 0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def norm_name(s: str) -> str:
    s = s.lower().replace("_", " ")
    s = s.replace("–", "-").replace("—", "-").replace("‑", "-")
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def iter_json_objects(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            s = raw.strip()
            if not s:
                continue
            if s.startswith(","):
                s = s[1:].strip()
            if s.endswith(","):
                s = s[:-1].strip()
            if not s.startswith("{"):
                continue
            try:
                yield json.loads(s)
            except json.JSONDecodeError:
                continue


def decode_indexed_abstract(indexed: object) -> str:
    if not isinstance(indexed, dict):
        return ""
    inv = indexed.get("InvertedIndex")
    if not isinstance(inv, dict):
        return ""
    length = int(indexed.get("IndexLength") or 0)
    if length <= 0:
        max_pos = -1
        for positions in inv.values():
            if isinstance(positions, list) and positions:
                max_pos = max(max_pos, max(int(p) for p in positions))
        length = max_pos + 1
    words = [""] * length
    for word, positions in inv.items():
        if not isinstance(positions, list):
            continue
        for pos in positions:
            try:
                i = int(pos)
            except (TypeError, ValueError):
                continue
            if 0 <= i < length:
                words[i] = str(word)
    return " ".join(w for w in words if w).strip()


def load_fos_map(path: Path) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, int]]:
    name_to_id: Dict[str, str] = {}
    id_to_name: Dict[str, str] = {}
    id_to_level: Dict[str, int] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 6:
                continue
            fos_id = parts[0].strip()
            norm = parts[2].strip()
            display = parts[3].strip()
            if not fos_id:
                continue
            id_to_name[fos_id] = display or norm or fos_id
            id_to_level[fos_id] = int(parts[5]) if parts[5].isdigit() else -1
            for name in (norm, display):
                key = norm_name(name)
                if key:
                    name_to_id[key] = fos_id
    return name_to_id, id_to_name, id_to_level


def load_child_to_parents(path: Path) -> Dict[str, List[str]]:
    pat = re.compile(
        r"<https://makg.org/entity/(\d+)>\s+"
        r"<https://makg.org/property/hasParent>\s+"
        r"<https://makg.org/entity/(\d+)>\s+\."
    )
    out: Dict[str, List[str]] = defaultdict(list)
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            m = pat.match(line.strip())
            if m:
                out[m.group(1)].append(m.group(2))
    return dict(out)


def ancestor_cache_builder(child_to_parents: Dict[str, List[str]], max_depth: int):
    cache: Dict[str, List[Tuple[str, int]]] = {}

    def ancestors(seed: str) -> List[Tuple[str, int]]:
        if seed in cache:
            return cache[seed]
        out = [(seed, 0)]
        q = deque([(seed, 0)])
        seen = {seed}
        while q:
            node, dist = q.popleft()
            if dist >= max_depth:
                continue
            for parent in child_to_parents.get(node, []):
                if parent in seen:
                    continue
                seen.add(parent)
                out.append((parent, dist + 1))
                q.append((parent, dist + 1))
        cache[seed] = out
        return out

    return ancestors


def parse_task_members(members: str) -> List[str]:
    out = []
    for member in (members or "").split("|"):
        member = member.strip()
        if member:
            out.append(member.split("_", 1)[0])
    return out


def load_tasks(
    path: Path,
    name_to_id: Dict[str, str],
    id_to_name: Dict[str, str],
) -> List[dict]:
    tasks: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            skills = (row.get("skills") or "").split("|")
            weights = (row.get("skill_weights") or "").split("|")
            direct: List[Tuple[str, float, str]] = []
            missing: List[str] = []
            for idx, raw_skill in enumerate(skills):
                skill = raw_skill.strip()
                if not skill:
                    continue
                fos_id = name_to_id.get(norm_name(skill))
                if not fos_id:
                    missing.append(skill)
                    continue
                weight = safe_float(weights[idx], 0.0) if idx < len(weights) else 0.0
                if weight > 0:
                    direct.append((fos_id, weight, id_to_name.get(fos_id, skill)))
            tasks.append(
                {
                    "paper_id": row.get("paper_id", ""),
                    "row_idx": row.get("row_idx", ""),
                    "year": row.get("year", ""),
                    "team_size": int(float(row.get("team_size") or 0)),
                    "members": parse_task_members(row.get("members", "")),
                    "member_labels": row.get("members", ""),
                    "direct": direct,
                    "missing_skills": missing,
                    "raw_skills": row.get("skills", ""),
                }
            )
    return tasks


def build_task_subtree_vectors(task: dict, ancestors) -> Dict[str, Dict[str, float]]:
    subtree_vectors: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for leaf_id, weight, _ in task["direct"]:
        for node, _ in ancestors(leaf_id):
            subtree_vectors[node][leaf_id] += weight
    return {node: dict(vec) for node, vec in subtree_vectors.items()}


def read_profile_evidence(
    path: Path,
    max_profile_nodes: int,
    max_evidence_papers_per_node: int,
) -> List[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            node_id = (row.get("fos_id") or "").strip()
            if not node_id:
                continue
            weight_sum = safe_float(row.get("direct_weight_sum"), 0.0)
            if weight_sum <= 0:
                continue
            try:
                papers = json.loads(row.get("paper_weight_details") or "[]")
            except json.JSONDecodeError:
                papers = []
            papers = [p for p in papers if isinstance(p, dict) and p.get("paper_id")]
            papers.sort(
                key=lambda p: (
                    -safe_float(p.get("weight")),
                    -int(p.get("year") or 0) if str(p.get("year") or "").isdigit() else 0,
                    str(p.get("paper_id")),
                )
            )
            rows.append(
                {
                    "node_id": node_id,
                    "node_name": row.get("fos_name") or node_id,
                    "direct_weight_sum": weight_sum,
                    "papers": papers[:max_evidence_papers_per_node],
                }
            )
    rows.sort(key=lambda r: r["direct_weight_sum"], reverse=True)
    return rows[:max_profile_nodes] if max_profile_nodes > 0 else rows


def softmax(items: Sequence[Tuple[str, float]]) -> Dict[str, float]:
    if not items:
        return {}
    vals = np.array([math.log1p(max(w, 0.0)) for _, w in items], dtype=np.float32)
    vals -= vals.max()
    probs = np.exp(vals)
    probs /= probs.sum()
    return {key: float(prob) for (key, _), prob in zip(items, probs)}


def l2(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def read_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def load_embedding_table(ids_path: Path, npy_path: Path) -> Dict[str, np.ndarray]:
    ids = []
    with ids_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            ids.append(row["id"])
    arr = np.load(npy_path)
    if len(ids) != arr.shape[0]:
        raise ValueError(f"ids/embedding row mismatch: {len(ids)} vs {arr.shape[0]}")
    return {id_: arr[i] for i, id_ in enumerate(ids)}


def save_embedding_table(
    ids_path: Path,
    npy_path: Path,
    ids: Sequence[str],
    embeddings: np.ndarray,
    extra: Dict[str, dict] | None = None,
) -> None:
    ids_path.parent.mkdir(parents=True, exist_ok=True)
    npy_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(npy_path, embeddings.astype(np.float32))
    extra = extra or {}
    keys = ["id"]
    extra_keys = sorted({k for rec in extra.values() for k in rec})
    with ids_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys + extra_keys, delimiter="\t")
        writer.writeheader()
        for id_ in ids:
            row = {"id": id_}
            row.update(extra.get(id_, {}))
            writer.writerow(row)
