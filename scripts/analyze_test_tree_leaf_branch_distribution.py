#!/usr/bin/env python3
"""Per-paper distribution of (a) leaf count and (b) branch-point count
in the induced taxonomy tree, for the level1 cross-domain test set."""
import sys, json, statistics
from collections import defaultdict, Counter
from pathlib import Path

sys.path.insert(0, "scripts")
import evaluate_tree_knapsack_dp_regions as T

ROOT = Path(".")
task_nodes = ROOT / "output/hierec_embedding_server_inputs/task_nodes.jsonl"
test_ids_path = ROOT / "output/cross_domain_filter_direct_fos_by_level/level1_direct_fos_cross_domain_paper_ids.txt"
fos_children = ROOT / "data/dblp/13.FieldOfStudyChildren.nt"

test_ids = [x.strip() for x in test_ids_path.read_text().splitlines() if x.strip()]
test_set = set(test_ids)

rows_by_paper, info, order = T.load_tasks(task_nodes)
child_to_parents = T.load_child_to_parents(fos_children)

leaf_counts = {}
branch_counts_incl_root = {}
branch_counts_excl_root = {}
node_counts = {}
for pid in test_ids:
    rows = rows_by_paper[pid]
    node_ids = {str(r["node_id"]) for r in rows}
    children = T.build_children(rows, child_to_parents)
    # leaves = task nodes that are no one's chosen parent (no children)
    leaves = [n for n in node_ids if len(children.get(n, [])) == 0]
    # branch points = nodes with >=2 children
    branch_incl = [p for p, ch in children.items() if len(ch) >= 2]
    branch_excl = [p for p in branch_incl if p != T.VIRTUAL_ROOT]
    leaf_counts[pid] = len(leaves)
    branch_counts_incl_root[pid] = len(branch_incl)
    branch_counts_excl_root[pid] = len(branch_excl)
    node_counts[pid] = len(node_ids)

def summarize(name, d):
    vals = [d[p] for p in test_ids]
    print(f"\n=== {name} (n={len(vals)} papers) ===")
    print(f"  mean={statistics.mean(vals):.2f}  median={statistics.median(vals)}  "
          f"min={min(vals)}  max={max(vals)}  stdev={statistics.pstdev(vals):.2f}")
    c = Counter(vals)
    print("  value : #papers : histogram")
    for v in sorted(c):
        print(f"   {v:>4} : {c[v]:>5}   {'#'*c[v]}")

summarize("Total task-tree nodes per paper", node_counts)
summarize("DIRECT LEAF count per paper (tree leaves)", leaf_counts)
summarize("BRANCH POINTS per paper (>=2 children, incl. virtual root)", branch_counts_incl_root)
summarize("BRANCH POINTS per paper (>=2 children, EXCL virtual root)", branch_counts_excl_root)

# team size for reference
team_sizes = [info[p]["team_size"] for p in test_ids]
print(f"\n=== team_size (reference) === mean={statistics.mean(team_sizes):.2f} median={statistics.median(team_sizes)}")
