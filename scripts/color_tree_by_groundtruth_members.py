#!/usr/bin/env python3
"""Coloring experiment: color each induced-taxonomy-tree node by the groundtruth
team member whose representation is most cosine-similar to that node's frozen
role-description embedding, then analyze the resulting colored tree.

Answers three questions on the level1 cross-domain test set:
  (a) Are a member's nodes one contiguous piece on the tree, or many pieces?
  (b) How aligned/cohesive is each color block (intra- vs inter-member cosine)?
  (c) Do color boundaries fall on branch points (>=2 children) or mid-chain?

Note on internal nodes: in this pipeline EVERY task node (leaf and internal)
has its own role-description embedding, so internal nodes do NOT need mean/max
aggregation. We therefore color the full tree with native per-node embeddings.
A leaf-only restriction is also reported, per the requested first-version view.
"""
from __future__ import annotations

import argparse
import sys
import statistics
from collections import defaultdict, Counter, deque
from pathlib import Path

import numpy as np

sys.path.insert(0, "scripts")
import evaluate_tree_knapsack_dp_regions as T

ROOT = Path(".")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task-nodes-jsonl", default="output/hierec_embedding_server_inputs/task_nodes.jsonl")
    p.add_argument("--test-ids", default="output/cross_domain_filter_direct_fos_by_level/level1_direct_fos_cross_domain_paper_ids.txt")
    p.add_argument("--fos-children", default="data/dblp/13.FieldOfStudyChildren.nt")
    p.add_argument("--node-ids", default="output/all_expert_paper_embeddings/task_node_embedding_ids_strict_v2_no_label.tsv")
    p.add_argument("--node-embeddings", default="output/all_expert_paper_embeddings/task_node_embeddings_strict_v2_no_label.npy")
    p.add_argument("--member-ids", default="output/virtual_root_role_descriptions/expert_mean_paper_embedding_ids.tsv")
    p.add_argument("--member-embeddings", default="output/virtual_root_role_descriptions/expert_mean_paper_embeddings.npy")
    p.add_argument("--out-dir", default="output/tree_coloring_groundtruth")
    p.add_argument("--print-trees", type=int, default=5, help="How many representative papers to print as text trees.")
    return p.parse_args()


def load_id_index(path: Path, key_col: int = 0) -> dict:
    idx = {}
    with path.open() as f:
        header = f.readline()
        for i, line in enumerate(f):
            parts = line.rstrip("\n").split("\t")
            idx[parts[key_col]] = i
    return idx


def l2norm(mat: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(mat, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return mat / n


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    test_ids = [x.strip() for x in Path(args.test_ids).read_text().splitlines() if x.strip()]
    test_set = set(test_ids)

    rows_by_paper, info, order = T.load_tasks(Path(args.task_nodes_jsonl))
    child_to_parents = T.load_child_to_parents(Path(args.fos_children))

    node_idx = load_id_index(Path(args.node_ids))      # "paper::node" -> row
    member_idx = load_id_index(Path(args.member_ids))  # expert_id -> row
    node_emb = l2norm(np.load(args.node_embeddings).astype(np.float32))
    member_emb = l2norm(np.load(args.member_embeddings).astype(np.float32))

    # ---- per-paper coloring & analysis ----
    # aggregates
    agg = {
        "leaf": {"single": 0, "multi": 0, "members": 0, "piece_counts": Counter(),
                 "intra": [], "inter": [], "piece_dist": []},
        "all":  {"single": 0, "multi": 0, "members": 0, "piece_counts": Counter(),
                 "intra": [], "inter": [], "piece_dist": []},
    }
    boundary = {"branch": 0, "chain": 0}          # full-tree boundary edges
    per_paper_records = []  # for choosing representative trees & dumps

    for pid in test_ids:
        rows = rows_by_paper.get(pid)
        if not rows:
            continue
        members = info[pid]["members"]
        members = [m for m in members if m in member_idx]
        if not members:
            continue
        mvecs = member_emb[[member_idx[m] for m in members]]  # (M,d)

        children = T.build_children(rows, child_to_parents)
        node_ids = [str(r["node_id"]) for r in rows]
        row_by_node = {str(r["node_id"]): r for r in rows}
        # parent of each node (real-node tree; VIRTUAL_ROOT as connector)
        parent_of = {}
        for par, chs in children.items():
            for c in chs:
                parent_of[c] = par

        leaves = {n for n in node_ids if len(children.get(n, [])) == 0}

        # color each node that has an embedding
        color = {}     # node_id -> member_id
        topsim = {}    # node_id -> best cosine
        node_member_sim = {}  # node_id -> full sim vector over members
        for n in node_ids:
            key = f"{pid}::{n}"
            ri = node_idx.get(key)
            if ri is None:
                continue
            z = node_emb[ri]
            sims = mvecs @ z  # (M,)
            best = int(np.argmax(sims))
            color[n] = members[best]
            topsim[n] = float(sims[best])
            node_member_sim[n] = sims

        # ---- build undirected full-tree graph (incl virtual root) for distances ----
        adj = defaultdict(list)
        for c, par in parent_of.items():
            adj[c].append(par)
            adj[par].append(c)

        def tree_dist(a, b):
            if a == b:
                return 0
            seen = {a}
            q = deque([(a, 0)])
            while q:
                x, d = q.popleft()
                for y in adj[x]:
                    if y not in seen:
                        if y == b:
                            return d + 1
                        seen.add(y)
                        q.append((y, d + 1))
            return -1

        # ---- (a) connectivity per member, for leaf-only and all-nodes scopes ----
        scope_nodes = {"leaf": [n for n in color if n in leaves],
                       "all": list(color.keys())}
        paper_member_pieces = {"leaf": {}, "all": {}}
        for scope, nodes in scope_nodes.items():
            cset = set(nodes)
            by_member = defaultdict(set)
            for n in nodes:
                by_member[color[n]].add(n)
            for m, mnodes in by_member.items():
                # connected components using parent-child edges among same-color nodes
                # (edge exists if both endpoints are this member's nodes in scope)
                comp_id = {}
                cid = 0
                for start in mnodes:
                    if start in comp_id:
                        continue
                    cid += 1
                    stack = [start]
                    comp_id[start] = cid
                    while stack:
                        x = stack.pop()
                        neigh = []
                        p = parent_of.get(x)
                        if p in mnodes:
                            neigh.append(p)
                        for ch in children.get(x, []):
                            if ch in mnodes:
                                neigh.append(ch)
                        for y in neigh:
                            if y not in comp_id:
                                comp_id[y] = cid
                                stack.append(y)
                ncomp = len(set(comp_id.values()))
                paper_member_pieces[scope][m] = ncomp
                agg[scope]["members"] += 1
                agg[scope]["piece_counts"][ncomp] += 1
                if ncomp == 1:
                    agg[scope]["single"] += 1
                else:
                    agg[scope]["multi"] += 1
                    # pairwise distance between component representatives
                    comps = defaultdict(list)
                    for n, c in comp_id.items():
                        comps[c].append(n)
                    reps = [v[0] for v in comps.values()]
                    dists = []
                    for i in range(len(reps)):
                        for j in range(i + 1, len(reps)):
                            d = tree_dist(reps[i], reps[j])
                            if d > 0:
                                dists.append(d)
                    if dists:
                        agg[scope]["piece_dist"].append(min(dists))

                # ---- (b) cohesion: intra vs inter member cosine for this member's block ----
                mi = members.index(m)
                for n in mnodes:
                    sims = node_member_sim[n]
                    agg[scope]["intra"].append(float(sims[mi]))
                    others = [float(sims[k]) for k in range(len(members)) if k != mi]
                    if others:
                        agg[scope]["inter"].append(sum(others) / len(others))

        # ---- (c) boundary edges (full tree, colored nodes only) ----
        for c, par in parent_of.items():
            if c in color and par in color and color[c] != color[par]:
                if len(children.get(par, [])) >= 2:
                    boundary["branch"] += 1
                else:
                    boundary["chain"] += 1

        per_paper_records.append({
            "pid": pid,
            "n_nodes": len(color),
            "n_leaves": len(scope_nodes["leaf"]),
            "n_members": len(members),
            "max_pieces_leaf": max(paper_member_pieces["leaf"].values()) if paper_member_pieces["leaf"] else 0,
            "max_pieces_all": max(paper_member_pieces["all"].values()) if paper_member_pieces["all"] else 0,
            "children": children, "color": color, "topsim": topsim,
            "row_by_node": row_by_node, "members": members,
            "pieces_all": paper_member_pieces["all"],
        })

    # ---------- report ----------
    def pct(a, b):
        return 100.0 * a / b if b else 0.0

    lines = []
    def emit(s=""):
        print(s)
        lines.append(s)

    emit("=" * 70)
    emit(f"COLORING EXPERIMENT  (level1 cross-domain test set, papers={len(per_paper_records)})")
    emit("=" * 70)
    for scope in ("leaf", "all"):
        a = agg[scope]
        tot = a["members"]
        emit(f"\n### scope = {scope.upper()}  (member-color blocks = {tot})")
        emit(f"(a) SINGLE-piece members : {a['single']:>5}  ({pct(a['single'],tot):.1f}%)")
        emit(f"    MULTI-piece  members : {a['multi']:>5}  ({pct(a['multi'],tot):.1f}%)")
        emit(f"    piece-count histogram: " +
             ", ".join(f"{k}:{a['piece_counts'][k]}" for k in sorted(a['piece_counts'])))
        if a["piece_dist"]:
            emit(f"    multi-piece nearest-gap hops: mean={statistics.mean(a['piece_dist']):.2f} "
                 f"median={statistics.median(a['piece_dist'])} max={max(a['piece_dist'])}")
        if a["intra"]:
            emit(f"(b) intra-member cos: mean={statistics.mean(a['intra']):.4f}   "
                 f"inter-member cos: mean={statistics.mean(a['inter']):.4f}   "
                 f"gap={statistics.mean(a['intra'])-statistics.mean(a['inter']):.4f}")

    tb = boundary["branch"] + boundary["chain"]
    emit(f"\n(c) BOUNDARY edges (full tree): total={tb}  "
         f"at branch-point(>=2 ch)={boundary['branch']} ({pct(boundary['branch'],tb):.1f}%)  "
         f"mid-chain={boundary['chain']} ({pct(boundary['chain'],tb):.1f}%)")

    # ---------- representative text trees ----------
    emit("\n" + "=" * 70)
    emit("REPRESENTATIVE COLORED TREES")
    emit("=" * 70)
    recs = per_paper_records
    chosen = []
    by_frag = sorted(recs, key=lambda r: -r["max_pieces_all"])
    chosen += by_frag[:max(1, args.print_trees - 2)]            # most fragmented
    cleanest = sorted(recs, key=lambda r: (r["max_pieces_all"], -r["n_nodes"]))
    for r in cleanest:
        if r not in chosen:
            chosen.append(r)
        if len(chosen) >= args.print_trees:
            break

    # stable short color labels per paper
    for r in chosen:
        emit(f"\n--- paper {r['pid']}  nodes={r['n_nodes']} members={r['n_members']} "
             f"max_pieces(all)={r['max_pieces_all']} ---")
        mem_label = {m: f"M{i+1}" for i, m in enumerate(r["members"])}
        emit("    members: " + ", ".join(f"{mem_label[m]}={m}" for m in r["members"]))
        children = r["children"]; color = r["color"]; topsim = r["topsim"]
        row_by_node = r["row_by_node"]

        def walk(node, depth):
            for c in children.get(node, []):
                name = row_by_node.get(c, {}).get("node_name", c)
                col = color.get(c)
                tag = f"[{mem_label.get(col,'?')} {topsim.get(c,0):.2f}]" if col else "[--]"
                leaf = "*" if len(children.get(c, [])) == 0 else " "
                emit(f"    {'  '*depth}{leaf}{name} {tag}")
                walk(c, depth + 1)

        walk(T.VIRTUAL_ROOT, 0)

    (out_dir / "coloring_report.txt").write_text("\n".join(lines), encoding="utf-8")
    emit(f"\n[written] {out_dir/'coloring_report.txt'}")


if __name__ == "__main__":
    main()
