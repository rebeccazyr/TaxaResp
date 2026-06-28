#!/usr/bin/env python3
"""Complete direct FoS nodes into connected taxonomy (ROOT -> leaves).

Input:
- per-expert direct nodes TSV, e.g. <expert_id>_direct_fos_nodes.tsv
  (from build_author_direct_fos_nodes.py / build_all_expert_profiles_onepass.py)

Output per expert:
- <expert_id>_personal_taxonomy_nodes.tsv
- <expert_id>_personal_taxonomy_edges.tsv
- <expert_id>_personal_taxonomy_connected_edges.tsv
- <expert_id>_personal_taxonomy_connected_tree.txt
- <expert_id>_personal_taxonomy_stats.txt
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, List, Set, Tuple


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Complete direct FoS nodes into connected taxonomy without rescanning DBLP"
    )
    p.add_argument(
        "--direct-nodes-dir",
        default="output/expert_profile",
        help="Directory containing *_direct_fos_nodes.tsv",
    )
    p.add_argument(
        "--direct-nodes-file",
        default="",
        help="Optional single direct nodes file to process",
    )
    p.add_argument(
        "--fos-map",
        default="data/dblp/FieldsOfStudy.txt",
        help="Path to FieldsOfStudy.txt",
    )
    p.add_argument(
        "--fos-children",
        default="data/dblp/13.FieldOfStudyChildren.nt",
        help="Path to FoS child-parent edges file",
    )
    p.add_argument(
        "--root-tree-tsv",
        default="output/entity_13_minimal_connected_tree.tsv",
        help="Path to TSV containing ROOT_FOS edges",
    )
    p.add_argument(
        "--out-dir",
        default="output/expert_profile_taxonomy",
        help="Output directory",
    )
    p.add_argument(
        "--progress-every",
        type=int,
        default=500,
        help="Print progress every N experts",
    )
    return p.parse_args()


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


def load_fos_edges(path: Path) -> Dict[str, Set[str]]:
    child_to_parents: Dict[str, Set[str]] = defaultdict(set)
    pat = re.compile(
        r"<https://makg.org/entity/(\d+)>\s+"
        r"<https://makg.org/property/hasParent>\s+"
        r"<https://makg.org/entity/(\d+)>\s+\."
    )
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            m = pat.match(line.strip())
            if not m:
                continue
            child, parent = m.group(1), m.group(2)
            child_to_parents[child].add(parent)
    return child_to_parents


def load_root_children(path: Path) -> List[str]:
    roots: List[str] = []
    if not path.exists():
        return roots
    with path.open("r", encoding="utf-8") as f:
        first = True
        for line in f:
            if first:
                first = False
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                continue
            parent_id, child_id = parts[0].strip(), parts[2].strip()
            if parent_id == "ROOT_FOS" and child_id:
                roots.append(child_id)
    seen = set()
    out = []
    for x in roots:
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


def parse_direct_nodes(path: Path) -> Dict[str, dict]:
    """Return seed nodes keyed by fos_id."""
    seeds: Dict[str, dict] = {}
    with path.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f, delimiter="\t")
        for row in r:
            fos_id = (row.get("fos_id") or "").strip()
            if not fos_id:
                continue
            try:
                cnt = int(float(row.get("direct_paper_count", "0") or 0))
            except Exception:
                cnt = 0
            if cnt <= 0:
                continue
            seeds[fos_id] = {
                "fos_id": fos_id,
                "fos_name": row.get("fos_name", ""),
                "direct_paper_count": cnt,
                "direct_weight_sum": float(row.get("direct_weight_sum", "0") or 0),
                "direct_weight_avg": float(row.get("direct_weight_avg", "0") or 0),
                "paper_weight_details": row.get("paper_weight_details", "[]"),
            }
    return seeds


def build_for_one(
    direct_file: Path,
    out_dir: Path,
    id_to_name: Dict[str, str],
    child_to_parents: Dict[str, Set[str]],
    preferred_root_children: List[str],
) -> None:
    expert_id = direct_file.name.replace("_direct_fos_nodes.tsv", "")
    seeds = parse_direct_nodes(direct_file)
    seed_nodes = set(seeds.keys())

    edges: Set[Tuple[str, str]] = set()
    nodes: Set[str] = set(seed_nodes)
    q = deque(seed_nodes)
    seen = set(seed_nodes)

    while q:
        child = q.popleft()
        for parent in child_to_parents.get(child, set()):
            edges.add((parent, child))
            nodes.add(parent)
            nodes.add(child)
            if parent not in seen:
                seen.add(parent)
                q.append(parent)

    def node_key(nid: str) -> Tuple[str, str]:
        return (id_to_name.get(nid, nid).lower(), nid)

    roots = []
    for n in nodes:
        pin = [p for p in child_to_parents.get(n, set()) if p in nodes]
        if not pin:
            roots.append(n)
    roots = sorted(set(roots), key=node_key)
    edge_rows = sorted(edges, key=lambda e: (node_key(e[0]), node_key(e[1])))

    preferred_in_graph = [x for x in preferred_root_children if x in nodes]
    root_connect = preferred_in_graph[:]
    for r in roots:
        if r not in root_connect:
            root_connect.append(r)

    # nodes.tsv
    nodes_file = out_dir / f"{expert_id}_personal_taxonomy_nodes.tsv"
    with nodes_file.open("w", encoding="utf-8") as f:
        f.write(
            "fos_id\tfos_name\tin_author_papers\tdirect_paper_count\t"
            "direct_weight_sum\tdirect_weight_avg\tpaper_weight_details\n"
        )
        for nid in sorted(nodes, key=node_key):
            if nid in seeds:
                s = seeds[nid]
                f.write(
                    f"{nid}\t{id_to_name.get(nid, s['fos_name'])}\t1\t{s['direct_paper_count']}\t"
                    f"{s['direct_weight_sum']:.5f}\t{s['direct_weight_avg']:.5f}\t{s['paper_weight_details']}\n"
                )
            else:
                f.write(
                    f"{nid}\t{id_to_name.get(nid, nid)}\t0\t0\t\t\t\n"
                )

    def edge_line(parent_id: str, child_id: str) -> str:
        s = seeds.get(child_id)
        if s:
            sum_str = f"{s['direct_weight_sum']:.5f}"
            cnt_str = str(s["direct_paper_count"])
            avg_str = f"{s['direct_weight_avg']:.5f}"
            flag = "1"
            details = s["paper_weight_details"]
        else:
            sum_str = ""
            cnt_str = ""
            avg_str = ""
            flag = "0"
            details = "[]"
        return (
            f"{parent_id}\t{id_to_name.get(parent_id, parent_id)}\t"
            f"{child_id}\t{id_to_name.get(child_id, child_id)}\t"
            f"{sum_str}\t{cnt_str}\t{avg_str}\t{flag}\t{details}\n"
        )

    edges_header = (
        "parent_id\tparent_name\tchild_id\tchild_name\t"
        "child_weight_sum\tchild_paper_count\tchild_weight_avg\t"
        "is_author_fos\tchild_paper_weight_details\n"
    )

    edges_file = out_dir / f"{expert_id}_personal_taxonomy_edges.tsv"
    with edges_file.open("w", encoding="utf-8") as f:
        f.write(edges_header)
        for p, c in edge_rows:
            f.write(edge_line(p, c))

    connected_file = out_dir / f"{expert_id}_personal_taxonomy_connected_edges.tsv"
    with connected_file.open("w", encoding="utf-8") as f:
        f.write(edges_header)
        for c in root_connect:
            f.write(
                f"ROOT_FOS\tField of study\t{c}\t{id_to_name.get(c, c)}\t\t\t\t0\t[]\n"
            )
        for p, c in edge_rows:
            f.write(edge_line(p, c))

    # tree txt
    adj: Dict[str, List[str]] = defaultdict(list)
    for c in root_connect:
        adj["ROOT_FOS"].append(c)
    for p, c in edge_rows:
        adj[p].append(c)
    for k in list(adj.keys()):
        adj[k] = sorted(set(adj[k]), key=node_key)

    def node_label(nid: str) -> str:
        s = seeds.get(nid)
        if not s:
            return f"{id_to_name.get(nid, nid)} ({nid})"
        try:
            items = json.loads(s["paper_weight_details"])
        except Exception:
            items = []
        pieces = []
        for d in items:
            pid = d.get("paper_id", "")
            w = d.get("weight", "")
            pieces.append(f"{pid}:{w}")
        return f"{id_to_name.get(nid, nid)} ({nid}, papers=[{'; '.join(pieces)}])"

    lines = ["Field of study (ROOT_FOS)"]

    def dfs(node: str, depth: int) -> None:
        for c in adj.get(node, []):
            lines.append("  " * depth + f"- {node_label(c)}")
            dfs(c, depth + 1)

    dfs("ROOT_FOS", 1)
    tree_file = out_dir / f"{expert_id}_personal_taxonomy_connected_tree.txt"
    tree_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

    stats_file = out_dir / f"{expert_id}_personal_taxonomy_stats.txt"
    with stats_file.open("w", encoding="utf-8") as f:
        f.write(f"expert_id={expert_id}\n")
        f.write(f"seed_nodes={len(seed_nodes)}\n")
        f.write(f"taxonomy_total_nodes={len(nodes)}\n")
        f.write(f"taxonomy_edges={len(edge_rows)}\n")
        f.write(f"taxonomy_roots={len(roots)}\n")
        f.write(f"root_connected_nodes={len(root_connect)}\n")


def main() -> None:
    args = parse_args()

    _, id_to_name = load_fos_map(Path(args.fos_map))
    child_to_parents = load_fos_edges(Path(args.fos_children))
    preferred_root_children = load_root_children(Path(args.root_tree_tsv))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.direct_nodes_file:
        files = [Path(args.direct_nodes_file)]
    else:
        ddir = Path(args.direct_nodes_dir)
        files = sorted(ddir.glob("*_direct_fos_nodes.tsv"))

    total = len(files)
    for i, f in enumerate(files, start=1):
        build_for_one(
            direct_file=f,
            out_dir=out_dir,
            id_to_name=id_to_name,
            child_to_parents=child_to_parents,
            preferred_root_children=preferred_root_children,
        )
        if args.progress_every > 0 and i % args.progress_every == 0:
            print(f"progress experts={i}/{total}")

    print(f"done experts={total}")
    print(f"output_dir={out_dir}")


if __name__ == "__main__":
    main()
