"""Citation-graph proximity helpers.

The graph is expected to use Cartesian-normalized author-author citation
weights. Proximity is normalized by the candidate author's weighted degree by
default, so prolific or broadly connected authors do not look close to every
team solely because their total citation connectivity is high.
"""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, Mapping

Adjacency = Dict[str, Dict[str, float]]


def load_undirected_adjacency(path: Path) -> Adjacency:
    adjacency: Adjacency = defaultdict(dict)
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            a = str(row["author_id_a"])
            b = str(row["author_id_b"])
            weight = float(row["weight"])
            if weight <= 0.0:
                continue
            adjacency[a][b] = adjacency[a].get(b, 0.0) + weight
            adjacency[b][a] = adjacency[b].get(a, 0.0) + weight
    return dict(adjacency)


def weighted_degrees(adjacency: Mapping[str, Mapping[str, float]]) -> Dict[str, float]:
    return {author_id: sum(neighbors.values()) for author_id, neighbors in adjacency.items()}


def raw_team_proximity(
    author_id: str,
    team_author_ids: Iterable[str],
    adjacency: Mapping[str, Mapping[str, float]],
    *,
    exclude_self: bool = True,
) -> float:
    neighbors = adjacency.get(str(author_id), {})
    total = 0.0
    seen = set()
    for team_author_id in team_author_ids:
        team_author_id = str(team_author_id)
        if not team_author_id or team_author_id in seen:
            continue
        seen.add(team_author_id)
        if exclude_self and team_author_id == str(author_id):
            continue
        total += float(neighbors.get(team_author_id, 0.0))
    return total


def normalize_proximity(raw_value: float, degree: float, mode: str = "degree") -> float:
    if mode == "none":
        return raw_value
    if degree <= 0.0:
        return 0.0
    if mode == "degree":
        return raw_value / degree
    if mode == "sqrt_degree":
        return raw_value / math.sqrt(degree)
    raise ValueError(f"unknown degree normalization mode: {mode}")


def team_proximity(
    author_id: str,
    team_author_ids: Iterable[str],
    adjacency: Mapping[str, Mapping[str, float]],
    degrees: Mapping[str, float],
    *,
    normalization: str = "degree",
    exclude_self: bool = True,
) -> tuple[float, float, float]:
    raw_value = raw_team_proximity(
        author_id,
        team_author_ids,
        adjacency,
        exclude_self=exclude_self,
    )
    degree = float(degrees.get(str(author_id), 0.0))
    return raw_value, degree, normalize_proximity(raw_value, degree, normalization)
