#!/usr/bin/env python3
"""Coloring experiment, SAME-NODE alignment variant.

color(v) = argmax over this paper's groundtruth members of
           cos( task_node_embedding[paper::v] , expert_node_embedding[member::v] )

where expert_node_embedding[member::v] is the member's history papers accumulated
ON node v. A member can only color node v if it actually has an expert-node
embedding there (i.e. linked evidence papers to v); otherwise it does not compete
for v. This differs from the global expert-mean variant
(`color_tree_by_groundtruth_members.py`).

Analyzes the colored induced taxonomy tree on the level1 cross-domain test set:
  (a) single contiguous piece vs many pieces per member
  (b) intra- vs inter-member same-node cosine cohesion
  (c) color boundaries at branch points vs mid-chain
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


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task-nodes-jsonl", default="output/hierec_embedding_server_inputs/task_nodes.jsonl")
    p.add_argument("--test-ids", default="output/cross_domain_filter_direct_fos_by_level/level1_direct_fos_cross_domain_paper_ids.txt")
    p.add_argument("--fos-children", default="data/dblp/13.FieldOfStudyChildren.nt")
    p.add_argument("--node-ids", default="output/all_expert_paper_embeddings/task_node_embedding_ids_strict_v2_no_label.tsv")
    p.add_argument("--node-embeddings", default="output/all_expert_paper_embeddings/task_node_embeddings_strict_v2_no_label.npy")
    p.add_argument("--expert-node-ids", default="output/all_expert_paper_embeddings/expert_node_embedding_ids_no_label.tsv")
    p.add_argument("--expert-node-embeddings", default="output/all_expert_paper_embeddings/expert_node_embeddings_no_label.npy")
    p.add_argument("--out-dir", default="output/tree_coloring_expert_node")
    p.add_argument("--print-trees", type=int, default=5)
    return p.parse_args()


def l2norm(mat: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(mat, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return mat / n


def load_task_node_index(path: Path) -> dict:
    idx = {}
    with path.open() as f:
        f.readline()
        for i, line in enumerate(f):
            idx[line.split("\t", 1)[0]] = i
    return idx


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    test_ids = [x.strip() for x in Path(args.test_ids).read_text().splitlines() if x.strip()]

    rows_by_paper, info, order = T.load_tasks(Path(args.task_nodes_jsonl))
    child_to_parents = T.load_child_to_parents(Path(args.fos_children))

    # ---- collect needed expert-node keys: member::node for test papers ----
    needed = set()                       # "member::node"
    paper_node_set = {}                  # pid -> set(node_id)
    for pid in test_ids:
        rows = rows_by_paper.get(pid)
        if not rows:
            continue
        members = info[pid]["members"]
        nodes = {str(r["node_id"]) for r in rows}
        paper_node_set[pid] = nodes
        for m in members:
            for n in nodes:
                needed.add(f"{m}::{n}")
    print(f"[info] need {len(needed)} (member,node) expert-node lookups", flush=True)

    # ---- single pass over expert-node ids to find rows for needed keys ----
    en_row = {}
    with Path(args.expert_node_ids).open() as f:
        f.readline()
        for i, line in enumerate(f):
            key = line.split("\t", 1)[0]
            if key in needed:
                en_row[key] = i
    print(f"[info] found {len(en_row)} / {len(needed)} keys present as expert-node embeddings", flush=True)

    # ---- load needed expert-node vectors (gather rows from mmap) ----
    en_mat = np.load(args.expert_node_embeddings, mmap_mode="r")
    keys = list(en_row.keys())
    rows_idx = np.array([en_row[k] for k in keys], dtype=np.int64)
    order_sort = np.argsort(rows_idx)
    gathered = np.empty((len(keys), en_mat.shape[1]), dtype=np.float32)
    for pos in order_sort:                                  # ascending row access on mmap
        gathered[pos] = en_mat[rows_idx[pos]]
    gathered = l2norm(gathered)
    en_vec = {k: gathered[i] for i, k in enumerate(keys)}   # member::node -> unit vec

    # ---- task node embeddings ----
    node_idx = load_task_node_index(Path(args.node_ids))
    node_emb = l2norm(np.load(args.node_embeddings).astype(np.float32))

    # ---- analysis ----
    agg = {s: {"single": 0, "multi": 0, "members": 0, "piece_counts": Counter(),
               "intra": [], "inter": [], "piece_dist": []} for s in ("leaf", "all")}
    boundary = {"branch": 0, "chain": 0}
    uncolored_nodes = 0
    colored_nodes = 0
    member_node_coverage = []   # fraction of (member,node) pairs that had an expert-node emb
    per_paper_records = []

    for pid in test_ids:
        rows = rows_by_paper.get(pid)
        if not rows:
            continue
        members = info[pid]["members"]
        children = T.build_children(rows, child_to_parents)
        node_ids = [str(r["node_id"]) for r in rows]
        row_by_node = {str(r["node_id"]): r for r in rows}
        parent_of = {}
        for par, chs in children.items():
            for c in chs:
                parent_of[c] = par
        leaves = {n for n in node_ids if len(children.get(n, [])) == 0}

        # member index restricted to those with ANY expert-node emb in this paper
        present_pairs = 0
        total_pairs = 0
        color = {}
        topsim = {}
        node_member_sim = {}     # node -> {member: sim}
        for n in node_ids:
            tkey = f"{pid}::{n}"
            ti = node_idx.get(tkey)
            if ti is None:
                continue
            z = node_emb[ti]
            sims = {}
            for m in members:
                total_pairs += 1
                ev = en_vec.get(f"{m}::{n}")
                if ev is None:
                    continue
                present_pairs += 1
                sims[m] = float(ev @ z)
            if not sims:
                uncolored_nodes += 1
                continue
            colored_nodes += 1
            best = max(sims, key=sims.get)
            color[n] = best
            topsim[n] = sims[best]
            node_member_sim[n] = sims
        if total_pairs:
            member_node_coverage.append(present_pairs / total_pairs)

        # undirected full-tree graph for distances
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

        scope_nodes = {"leaf": [n for n in color if n in leaves],
                       "all": list(color.keys())}
        paper_member_pieces = {"leaf": {}, "all": {}}
        for scope, nodes in scope_nodes.items():
            by_member = defaultdict(set)
            for n in nodes:
                by_member[color[n]].add(n)
            for m, mnodes in by_member.items():
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
                a = agg[scope]
                a["members"] += 1
                a["piece_counts"][ncomp] += 1
                if ncomp == 1:
                    a["single"] += 1
                else:
                    a["multi"] += 1
                    comps = defaultdict(list)
                    for n, c in comp_id.items():
                        comps[c].append(n)
                    reps = [v[0] for v in comps.values()]
                    dists = []
                    for i in range(len(reps)):
                        for j in range(i + 1, len(reps)):
                            dd = tree_dist(reps[i], reps[j])
                            if dd > 0:
                                dists.append(dd)
                    if dists:
                        a["piece_dist"].append(min(dists))
                for n in mnodes:
                    sims = node_member_sim[n]
                    a["intra"].append(sims[m])
                    others = [v for k, v in sims.items() if k != m]
                    if others:
                        a["inter"].append(sum(others) / len(others))

        for c, par in parent_of.items():
            if c in color and par in color and color[c] != color[par]:
                if len(children.get(par, [])) >= 2:
                    boundary["branch"] += 1
                else:
                    boundary["chain"] += 1

        per_paper_records.append({
            "pid": pid, "n_nodes": len(color),
            "n_leaves": len(scope_nodes["leaf"]), "n_members": len(members),
            "max_pieces_all": max(paper_member_pieces["all"].values()) if paper_member_pieces["all"] else 0,
            "children": children, "color": color, "topsim": topsim,
            "row_by_node": row_by_node, "members": members,
        })

    # ---------- report ----------
    def pct(a, b):
        return 100.0 * a / b if b else 0.0

    lines = []
    def emit(s=""):
        print(s)
        lines.append(s)

    emit("=" * 72)
    emit(f"COLORING EXPERIMENT — SAME-NODE expert-node alignment "
         f"(level1 cross-domain test, papers={len(per_paper_records)})")
    emit("=" * 72)
    emit(f"colored nodes={colored_nodes}  uncolored (no member has expert-node emb)={uncolored_nodes}  "
         f"({pct(uncolored_nodes, colored_nodes+uncolored_nodes):.1f}% uncolored)")
    if member_node_coverage:
        emit(f"avg (member,node) pairs with an expert-node emb = "
             f"{100*statistics.mean(member_node_coverage):.1f}%  "
             f"(a member competes for a node only where it has evidence)")
    for scope in ("leaf", "all"):
        a = agg[scope]
        tot = a["members"]
        emit(f"\n### scope = {scope.upper()}  (member-color blocks = {tot})")
        emit(f"(a) SINGLE-piece members : {a['single']:>5}  ({pct(a['single'],tot):.1f}%)")
        emit(f"    MULTI-piece  members : {a['multi']:>5}  ({pct(a['multi'],tot):.1f}%)")
        emit("    piece-count histogram: " +
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

    emit("\n" + "=" * 72)
    emit("REPRESENTATIVE COLORED TREES")
    emit("=" * 72)
    recs = per_paper_records
    chosen = []
    for r in sorted(recs, key=lambda r: -r["max_pieces_all"])[:max(1, args.print_trees - 2)]:
        chosen.append(r)
    for r in sorted(recs, key=lambda r: (r["max_pieces_all"], -r["n_nodes"])):
        if r not in chosen:
            chosen.append(r)
        if len(chosen) >= args.print_trees:
            break

    for r in chosen:
        emit(f"\n--- paper {r['pid']}  nodes={r['n_nodes']} members={r['n_members']} "
             f"max_pieces(all)={r['max_pieces_all']} ---")
        mem_label = {m: f"M{i+1}" for i, m in enumerate(r["members"])}
        emit("    members: " + ", ".join(f"{mem_label[m]}={m}" for m in r["members"]))
        children = r["children"]; color = r["color"]; topsim = r["topsim"]; row_by_node = r["row_by_node"]

        def walk(node, depth):
            for c in children.get(node, []):
                name = row_by_node.get(c, {}).get("node_name", c)
                col = color.get(c)
                tag = f"[{mem_label.get(col,'?')} {topsim.get(c,0):.2f}]" if col else "[uncolored]"
                leaf = "*" if len(children.get(c, [])) == 0 else " "
                emit(f"    {'  '*depth}{leaf}{name} {tag}")
                walk(c, depth + 1)

        walk(T.VIRTUAL_ROOT, 0)

    (out_dir / "coloring_report.txt").write_text("\n".join(lines), encoding="utf-8")
    emit(f"\n[written] {out_dir/'coloring_report.txt'}")


if __name__ == "__main__":
    main()
