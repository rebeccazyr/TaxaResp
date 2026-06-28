#!/usr/bin/env python3
"""Build a personal FoS taxonomy for a given author id from DBLP-style JSON.

The script constructs a FoS subgraph for all papers an author participated in,
then links it to ROOT_FOS using edges from an existing minimal connected tree.
It also records per-node direct paper-level FoS weights.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build personal FoS taxonomy by author id")
    parser.add_argument("--author-id", required=True, help="Author id in DBLP JSON")
    parser.add_argument(
        "--dblp-json",
        default="data/dblp/dblp.v12.json",
        help="Path to DBLP JSON file",
    )
    parser.add_argument(
        "--fos-map",
        default="data/dblp/FieldsOfStudy.txt",
        help="Path to FieldsOfStudy.txt for fos_name -> fos_id mapping",
    )
    parser.add_argument(
        "--fos-children",
        default="data/dblp/13.FieldOfStudyChildren.nt",
        help="Path to child-parent FoS edges (.nt)",
    )
    parser.add_argument(
        "--root-tree-tsv",
        default="output/entity_13_minimal_connected_tree.tsv",
        help="Path to taxonomy tree TSV containing ROOT_FOS edges",
    )
    parser.add_argument(
        "--out-dir",
        default="output/author_papers",
        help="Output directory",
    )
    return parser.parse_args()


def load_fos_map(path: Path) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Return (name_to_id, id_to_name)."""
    name_to_id: Dict[str, str] = {}
    id_to_name: Dict[str, str] = {}

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                continue
            fos_id = parts[0].strip()
            normalized_name = parts[2].strip()
            display_name = parts[3].strip()
            if not fos_id:
                continue

            preferred_name = display_name or normalized_name or fos_id
            id_to_name[fos_id] = preferred_name

            if normalized_name:
                name_to_id[normalized_name.lower()] = fos_id
            if display_name:
                name_to_id[display_name.lower()] = fos_id

    return name_to_id, id_to_name


def load_fos_edges(path: Path) -> Tuple[Dict[str, Set[str]], Dict[str, Set[str]]]:
    """Return (child_to_parents, parent_to_children)."""
    child_to_parents: Dict[str, Set[str]] = defaultdict(set)
    parent_to_children: Dict[str, Set[str]] = defaultdict(set)
    pattern = re.compile(
        r"<https://makg.org/entity/(\d+)>\s+"
        r"<https://makg.org/property/hasParent>\s+"
        r"<https://makg.org/entity/(\d+)>\s+\."
    )

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            match = pattern.match(line.strip())
            if not match:
                continue
            child_id, parent_id = match.group(1), match.group(2)
            child_to_parents[child_id].add(parent_id)
            parent_to_children[parent_id].add(child_id)

    return child_to_parents, parent_to_children


def load_root_children(path: Path) -> List[str]:
    """Load ROOT_FOS direct children from a tree TSV file."""
    root_children: List[str] = []
    if not path.exists():
        return root_children

    with path.open("r", encoding="utf-8") as f:
        header_skipped = False
        for line in f:
            if not header_skipped:
                header_skipped = True
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                continue
            parent_id, child_id = parts[0].strip(), parts[2].strip()
            if parent_id == "ROOT_FOS" and child_id:
                root_children.append(child_id)

    # Deduplicate while keeping order
    seen: Set[str] = set()
    unique = []
    for x in root_children:
        if x in seen:
            continue
        seen.add(x)
        unique.append(x)
    return unique


def iter_json_objects(path: Path) -> Iterable[dict]:
    """Yield JSON objects from a DBLP-style line-oriented file."""
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


def safe_float(v: object) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def main() -> None:
    args = parse_args()

    author_id = str(args.author_id)
    dblp_json_path = Path(args.dblp_json)
    fos_map_path = Path(args.fos_map)
    fos_children_path = Path(args.fos_children)
    root_tree_path = Path(args.root_tree_tsv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    name_to_id, id_to_name = load_fos_map(fos_map_path)
    child_to_parents, _ = load_fos_edges(fos_children_path)
    preferred_root_children = load_root_children(root_tree_path)

    papers: Dict[str, dict] = {}
    paper_fos_rows: List[dict] = []
    node_direct_weights: Dict[str, Dict[str, float]] = defaultdict(dict)
    unmapped_names: Set[str] = set()

    for obj in iter_json_objects(dblp_json_path):
        authors = obj.get("authors") or []
        if not isinstance(authors, list):
            continue

        matched = False
        for a in authors:
            if str((a or {}).get("id", "")) == author_id:
                matched = True
                break
        if not matched:
            continue

        paper_id = str(obj.get("id", ""))
        if not paper_id:
            continue

        paper_meta = {
            "paper_id": paper_id,
            "year": obj.get("year", ""),
            "title": obj.get("title", ""),
            "venue": ((obj.get("venue") or {}).get("raw", "") if isinstance(obj.get("venue"), dict) else ""),
            "doi": obj.get("doi", ""),
        }
        papers[paper_id] = paper_meta

        fos_items = obj.get("fos") or []
        if not isinstance(fos_items, list):
            continue

        for item in fos_items:
            if not isinstance(item, dict):
                continue
            fos_name = str(item.get("name", "")).strip()
            if not fos_name:
                continue
            weight = safe_float(item.get("w", 0.0))
            # User requirement: zero/non-positive weights are excluded from stats.
            if weight <= 0.0:
                continue
            fos_id = name_to_id.get(fos_name.lower(), "")

            paper_fos_rows.append(
                {
                    "paper_id": paper_id,
                    "year": paper_meta["year"],
                    "title": paper_meta["title"],
                    "fos_name": fos_name,
                    "w": weight,
                    "fos_id": fos_id,
                }
            )

            if not fos_id:
                unmapped_names.add(fos_name)
                continue

            # In case the same fos appears multiple times for one paper, keep max weight.
            prev = node_direct_weights[fos_id].get(paper_id)
            if prev is None or weight > prev:
                node_direct_weights[fos_id][paper_id] = weight

    seed_nodes = set(node_direct_weights.keys())

    # Build ancestor closure edges (parent -> child).
    taxonomy_edges: Set[Tuple[str, str]] = set()
    taxonomy_nodes: Set[str] = set(seed_nodes)
    queue: deque[str] = deque(seed_nodes)
    visited: Set[str] = set(seed_nodes)

    while queue:
        child = queue.popleft()
        for parent in child_to_parents.get(child, set()):
            taxonomy_edges.add((parent, child))
            taxonomy_nodes.add(parent)
            taxonomy_nodes.add(child)
            if parent not in visited:
                visited.add(parent)
                queue.append(parent)

    # Roots are nodes with no parent inside this induced subgraph.
    roots = []
    for node in taxonomy_nodes:
        parents_in_subgraph = [p for p in child_to_parents.get(node, set()) if p in taxonomy_nodes]
        if not parents_in_subgraph:
            roots.append(node)

    def node_sort_key(node_id: str) -> Tuple[str, str]:
        return (id_to_name.get(node_id, node_id).lower(), node_id)

    roots = sorted(set(roots), key=node_sort_key)
    edge_rows = sorted(taxonomy_edges, key=lambda e: (node_sort_key(e[0]), node_sort_key(e[1])))

    preferred_in_graph = [x for x in preferred_root_children if x in taxonomy_nodes]
    root_connect = preferred_in_graph[:]
    for r in roots:
        if r not in root_connect:
            root_connect.append(r)

    # Prepare node metadata.
    node_rows = []
    for node_id in sorted(taxonomy_nodes, key=node_sort_key):
        direct = node_direct_weights.get(node_id, {})
        direct_items = []
        weight_sum = 0.0
        for pid, w in sorted(direct.items(), key=lambda x: x[0]):
            direct_items.append(
                {
                    "paper_id": pid,
                    "weight": round(w, 5),
                }
            )
            weight_sum += w

        count = len(direct_items)
        avg = (weight_sum / count) if count else 0.0
        node_rows.append(
            {
                "fos_id": node_id,
                "fos_name": id_to_name.get(node_id, node_id),
                "in_author_papers": 1 if count else 0,
                "direct_paper_count": count,
                "direct_weight_sum": weight_sum,
                "direct_weight_avg": avg,
                "paper_weight_details": json.dumps(direct_items, ensure_ascii=False),
            }
        )

    def edge_line(parent_id: str, child_id: str) -> str:
        child_direct = node_direct_weights.get(child_id, {})
        child_count = len(child_direct)
        child_sum = sum(child_direct.values())
        child_avg = (child_sum / child_count) if child_count else 0.0
        child_details = []
        for pid, w in sorted(child_direct.items(), key=lambda x: x[0]):
            child_details.append(
                {
                    "paper_id": pid,
                    "weight": round(w, 5),
                }
            )

        sum_str = f"{child_sum:.5f}" if child_count else ""
        avg_str = f"{child_avg:.5f}" if child_count else ""
        cnt_str = str(child_count) if child_count else ""
        flag = "1" if child_count else "0"
        details_str = json.dumps(child_details, ensure_ascii=False) if child_count else "[]"

        return (
            f"{parent_id}\t{id_to_name.get(parent_id, parent_id)}\t"
            f"{child_id}\t{id_to_name.get(child_id, child_id)}\t"
            f"{sum_str}\t{cnt_str}\t{avg_str}\t{flag}\t{details_str}\n"
        )

    # Output files
    prefix = out_dir / f"{author_id}"

    paper_fos_file = prefix.with_name(f"{author_id}_paper_fos.tsv")
    with paper_fos_file.open("w", encoding="utf-8") as f:
        f.write("paper_id\tyear\ttitle\tfos_name\tw\tfos_id\n")
        for row in sorted(paper_fos_rows, key=lambda r: (r["paper_id"], r["fos_name"].lower())):
            f.write(
                f"{row['paper_id']}\t{row['year']}\t{row['title']}\t"
                f"{row['fos_name']}\t{row['w']:.5f}\t{row['fos_id']}\n"
            )

    node_file = prefix.with_name(f"{author_id}_personal_taxonomy_nodes.tsv")
    with node_file.open("w", encoding="utf-8") as f:
        f.write(
            "fos_id\tfos_name\tin_author_papers\tdirect_paper_count\t"
            "direct_weight_sum\tdirect_weight_avg\tpaper_weight_details\n"
        )
        for row in node_rows:
            f.write(
                f"{row['fos_id']}\t{row['fos_name']}\t{row['in_author_papers']}\t"
                f"{row['direct_paper_count']}\t{row['direct_weight_sum']:.5f}\t"
                f"{row['direct_weight_avg']:.5f}\t{row['paper_weight_details']}\n"
            )

    edges_file = prefix.with_name(f"{author_id}_personal_taxonomy_edges.tsv")
    with edges_file.open("w", encoding="utf-8") as f:
        f.write(
            "parent_id\tparent_name\tchild_id\tchild_name\t"
            "child_weight_sum\tchild_paper_count\tchild_weight_avg\t"
            "is_author_fos\tchild_paper_weight_details\n"
        )
        for parent_id, child_id in edge_rows:
            f.write(edge_line(parent_id, child_id))

    connected_edges_file = prefix.with_name(f"{author_id}_personal_taxonomy_connected_edges.tsv")
    with connected_edges_file.open("w", encoding="utf-8") as f:
        f.write(
            "parent_id\tparent_name\tchild_id\tchild_name\t"
            "child_weight_sum\tchild_paper_count\tchild_weight_avg\t"
            "is_author_fos\tchild_paper_weight_details\n"
        )
        for child_id in root_connect:
            f.write(
                "ROOT_FOS\tField of study\t"
                f"{child_id}\t{id_to_name.get(child_id, child_id)}\t\t\t\t0\t[]\n"
            )
        for parent_id, child_id in edge_rows:
            f.write(edge_line(parent_id, child_id))

    # Tree text (DAG can repeat nodes under different parents, which is expected).
    adjacency: Dict[str, List[str]] = defaultdict(list)
    for child_id in root_connect:
        adjacency["ROOT_FOS"].append(child_id)
    for parent_id, child_id in edge_rows:
        adjacency[parent_id].append(child_id)
    for k in list(adjacency.keys()):
        adjacency[k] = sorted(set(adjacency[k]), key=node_sort_key)

    def node_label(node_id: str) -> str:
        direct = node_direct_weights.get(node_id, {})
        name = id_to_name.get(node_id, node_id)
        if not direct:
            return f"{name} ({node_id})"
        details = "; ".join([f"{pid}:{w:.5f}" for pid, w in sorted(direct.items(), key=lambda x: x[0])])
        return f"{name} ({node_id}, papers=[{details}])"

    tree_lines = ["Field of study (ROOT_FOS)"]
    visited_on_path: Set[Tuple[str, str]] = set()

    def dfs(parent_id: str, depth: int) -> None:
        for child_id in adjacency.get(parent_id, []):
            tree_lines.append("  " * depth + f"- {node_label(child_id)}")
            edge_key = (parent_id, child_id)
            if edge_key in visited_on_path:
                continue
            visited_on_path.add(edge_key)
            dfs(child_id, depth + 1)

    dfs("ROOT_FOS", 1)

    tree_file = prefix.with_name(f"{author_id}_personal_taxonomy_connected_tree.txt")
    with tree_file.open("w", encoding="utf-8") as f:
        f.write("\n".join(tree_lines) + "\n")

    stats_file = prefix.with_name(f"{author_id}_personal_taxonomy_stats.txt")
    mapped_rows = sum(1 for r in paper_fos_rows if r["fos_id"])
    with stats_file.open("w", encoding="utf-8") as f:
        f.write(f"author_id={author_id}\n")
        f.write(f"input_papers={len(papers)}\n")
        f.write(f"papers_with_fos_found={len({r['paper_id'] for r in paper_fos_rows})}\n")
        f.write(f"unique_fos={len({(r['fos_name'].lower()) for r in paper_fos_rows})}\n")
        f.write(f"mapped_fos_rows={mapped_rows}\n")
        f.write(f"unmapped_fos_rows={len(paper_fos_rows) - mapped_rows}\n")
        f.write(f"taxonomy_seed_nodes={len(seed_nodes)}\n")
        f.write(f"taxonomy_total_nodes={len(taxonomy_nodes)}\n")
        f.write(f"taxonomy_edges={len(edge_rows)}\n")
        f.write(f"taxonomy_roots={len(roots)}\n")
        f.write(f"root_connected_nodes={len(root_connect)}\n")
        if unmapped_names:
            f.write("unmapped_fos_examples=" + " | ".join(sorted(unmapped_names)) + "\n")
        else:
            f.write("unmapped_fos_examples=\n")

    print("Generated:")
    print(f"- {paper_fos_file}")
    print(f"- {node_file}")
    print(f"- {edges_file}")
    print(f"- {connected_edges_file}")
    print(f"- {tree_file}")
    print(f"- {stats_file}")


if __name__ == "__main__":
    main()
