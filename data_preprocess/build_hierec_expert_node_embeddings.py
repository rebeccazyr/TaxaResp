#!/usr/bin/env python3
"""Build simple HieRec-style expert node embeddings on FoS taxonomy.

This is a local, deterministic prototype inspired by HieRec:
- direct FoS nodes play the role of fine-grained clicked subtopics;
- ancestor FoS nodes play the role of coarser topics;
- each expert-node representation is node semantic embedding plus an
  attention-weighted aggregation of child expert-node representations.

The prototype uses TF-IDF + SVD text vectors for node semantics instead of a
trained text encoder. It is intended to validate the hierarchy/aggregation
shape before replacing node semantics with LLM-generated descriptions and a
real embedding model.

When --use-paper-text is enabled, direct FoS nodes also attend over the
expert's historical paper title/abstract evidence for that node.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build HieRec-style hierarchical expert node embeddings"
    )
    p.add_argument(
        "--profile-dir",
        default="output/expert_profile_year_bins/all_2000_2019",
        help="Directory containing *_direct_fos_nodes.tsv expert profiles",
    )
    p.add_argument(
        "--expert-tsv",
        default="data/dblp/expert_id_name.tsv",
        help="TSV containing expert_id and name",
    )
    p.add_argument(
        "--fos-map",
        default="data/dblp/FieldsOfStudy.txt",
        help="FoS id/name/level map",
    )
    p.add_argument(
        "--fos-children",
        default="data/dblp/13.FieldOfStudyChildren.nt",
        help="FoS child-parent edge file",
    )
    p.add_argument("--out-dir", default="output/hierec_expert_node_embeddings")
    p.add_argument("--max-experts", type=int, default=100)
    p.add_argument("--max-profile-nodes", type=int, default=80)
    p.add_argument("--ancestor-depth", type=int, default=5)
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--sample-similarities", type=int, default=5)
    p.add_argument(
        "--use-paper-text",
        action="store_true",
        help="Use historical paper title + abstract evidence for direct FoS nodes",
    )
    p.add_argument(
        "--dblp-json",
        default="data/dblp/dblp.v12.json",
        help="DBLP JSON file used when --use-paper-text is enabled",
    )
    p.add_argument(
        "--max-evidence-papers-per-node",
        type=int,
        default=5,
        help="Maximum historical papers to keep as evidence for each direct FoS node",
    )
    p.add_argument(
        "--paper-text-max-chars",
        type=int,
        default=1200,
        help="Maximum title+abstract characters used per historical paper",
    )
    p.add_argument(
        "--progress-every",
        type=int,
        default=500000,
        help="Print DBLP scan progress every N records when loading paper text",
    )
    return p.parse_args()


def load_expert_names(path: Path) -> Dict[str, str]:
    out = {}
    with path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            expert_id = (row.get("expert_id") or "").strip()
            if expert_id:
                out[expert_id] = row.get("name") or expert_id
    return out


def load_fos_map(path: Path) -> Tuple[Dict[str, str], Dict[str, int]]:
    id_to_name: Dict[str, str] = {}
    id_to_level: Dict[str, int] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 6:
                continue
            fos_id = parts[0].strip()
            if not fos_id:
                continue
            id_to_name[fos_id] = parts[3].strip() or parts[2].strip() or fos_id
            id_to_level[fos_id] = int(parts[5]) if parts[5].isdigit() else -1
    return id_to_name, id_to_level


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


def safe_float(v: object) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def read_profile(
    path: Path, limit: int, max_evidence_papers_per_node: int
) -> Tuple[List[Tuple[str, str, float]], Dict[str, List[dict]]]:
    rows = []
    evidence: Dict[str, List[dict]] = {}
    with path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            fos_id = (row.get("fos_id") or "").strip()
            if not fos_id:
                continue
            weight = safe_float(row.get("direct_weight_sum"))
            if weight <= 0:
                continue
            paper_details = []
            try:
                paper_details = json.loads(row.get("paper_weight_details") or "[]")
            except json.JSONDecodeError:
                paper_details = []
            paper_details = [
                p for p in paper_details if isinstance(p, dict) and p.get("paper_id")
            ]
            paper_details.sort(
                key=lambda p: (
                    -safe_float(p.get("weight")),
                    -int(p.get("year") or 0) if str(p.get("year") or "").isdigit() else 0,
                    str(p.get("paper_id")),
                )
            )
            rows.append((fos_id, row.get("fos_name") or fos_id, weight))
            evidence[fos_id] = paper_details[:max_evidence_papers_per_node]
    rows.sort(key=lambda x: x[2], reverse=True)
    rows = rows[:limit]
    keep = {fos_id for fos_id, _, _ in rows}
    return rows, {fos_id: evidence.get(fos_id, []) for fos_id in keep}


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


def load_requested_paper_texts(
    path: Path,
    paper_ids: set,
    max_chars: int,
    progress_every: int,
) -> Dict[str, str]:
    if not paper_ids:
        return {}
    out: Dict[str, str] = {}
    parsed = 0
    for obj in iter_json_objects(path):
        parsed += 1
        if progress_every > 0 and parsed % progress_every == 0:
            print(f"paper_text_progress parsed={parsed:,} found={len(out):,}/{len(paper_ids):,}")
        paper_id = str(obj.get("id", ""))
        if paper_id not in paper_ids:
            continue
        title = str(obj.get("title") or "").strip()
        abstract = decode_indexed_abstract(obj.get("indexed_abstract"))
        text = f"{title}. {abstract}".strip()
        text = re.sub(r"\s+", " ", text)
        if max_chars > 0:
            text = text[:max_chars]
        if text:
            out[paper_id] = text
        if len(out) >= len(paper_ids):
            break
    return out


def softmax_weights(items: Sequence[Tuple[str, float]]) -> Dict[str, float]:
    if not items:
        return {}
    vals = np.array([math.log1p(max(w, 0.0)) for _, w in items], dtype=float)
    vals = vals - vals.max()
    probs = np.exp(vals)
    probs = probs / probs.sum()
    return {node: float(prob) for (node, _), prob in zip(items, probs)}


def l2(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def build_node_embeddings_for_expert(
    direct_items: List[Tuple[str, str, float]],
    direct_evidence: Dict[str, List[dict]],
    ancestors,
    child_to_parents: Dict[str, List[str]],
    semantic: Dict[str, np.ndarray],
    paper_vectors: Dict[str, np.ndarray] | None = None,
) -> Dict[str, dict]:
    paper_vectors = paper_vectors or {}
    direct_weight = {fos_id: weight for fos_id, _, weight in direct_items}
    nodes = set()
    for fos_id, _, _ in direct_items:
        for node, _ in ancestors(fos_id):
            nodes.add(node)

    parent_to_children: Dict[str, set] = defaultdict(set)
    for child in nodes:
        for parent in child_to_parents.get(child, []):
            if parent in nodes:
                parent_to_children[parent].add(child)

    subtree_weight: Dict[str, float] = defaultdict(float)
    direct_leaf_count: Dict[str, int] = defaultdict(int)
    for leaf, weight in direct_weight.items():
        for node, _ in ancestors(leaf):
            if node in nodes:
                subtree_weight[node] += weight
                direct_leaf_count[node] += 1

    # Bottom-up: nodes with fewer descendants tend to be more fine-grained.
    reps: Dict[str, np.ndarray] = {}
    meta: Dict[str, dict] = {}
    remaining = set(nodes)
    ordered_nodes: List[str] = []
    while remaining:
        ready = sorted(
            [
                node
                for node in remaining
                if all(child not in remaining for child in parent_to_children.get(node, set()))
            ],
            key=lambda n: (len(parent_to_children.get(n, set())), n),
        )
        if not ready:
            # The FoS graph can be noisy. Break cycles deterministically rather
            # than failing the entire prototype.
            ready = [sorted(remaining)[0]]
        ordered_nodes.extend(ready)
        remaining.difference_update(ready)

    for node in ordered_nodes:
        children = [c for c in parent_to_children.get(node, set()) if c in reps]
        if children:
            attn = softmax_weights([(c, subtree_weight[c]) for c in children])
            child_agg = sum(attn[c] * reps[c] for c in children)
        else:
            attn = {}
            child_agg = np.zeros_like(next(iter(semantic.values())))

        evidence_items = [
            (str(p.get("paper_id")), safe_float(p.get("weight")))
            for p in direct_evidence.get(node, [])
            if str(p.get("paper_id")) in paper_vectors
        ]
        if evidence_items:
            paper_attn = softmax_weights(evidence_items)
            paper_agg = sum(paper_attn[pid] * paper_vectors[pid] for pid, _ in evidence_items)
        else:
            paper_attn = {}
            paper_agg = np.zeros_like(next(iter(semantic.values())))

        # HieRec-style: node representation = aggregated evidence + node semantic embedding.
        rep = l2(semantic[node] + child_agg + paper_agg)
        reps[node] = rep
        meta[node] = {
            "embedding": rep,
            "subtree_weight_sum": subtree_weight[node],
            "direct_leaf_count": direct_leaf_count[node],
            "child_count": len(children),
            "attention_children": attn,
            "evidence_paper_count": len(evidence_items),
            "attention_papers": paper_attn,
            "is_direct_node": node in direct_weight,
        }
    return meta


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    expert_names = load_expert_names(Path(args.expert_tsv))
    id_to_name, id_to_level = load_fos_map(Path(args.fos_map))
    child_to_parents = load_child_to_parents(Path(args.fos_children))
    ancestors = ancestor_cache_builder(child_to_parents, args.ancestor_depth)

    profile_files = sorted(
        p for p in Path(args.profile_dir).glob("*_direct_fos_nodes.tsv")
        if not p.name.startswith("_")
    )[: args.max_experts]

    expert_direct: Dict[str, List[Tuple[str, str, float]]] = {}
    expert_evidence: Dict[str, Dict[str, List[dict]]] = {}
    all_nodes = set()
    requested_paper_ids = set()
    for path in profile_files:
        expert_id = path.name.replace("_direct_fos_nodes.tsv", "")
        direct_items, direct_evidence = read_profile(
            path, args.max_profile_nodes, args.max_evidence_papers_per_node
        )
        expert_direct[expert_id] = direct_items
        expert_evidence[expert_id] = direct_evidence
        if args.use_paper_text:
            for papers in direct_evidence.values():
                for paper in papers:
                    requested_paper_ids.add(str(paper.get("paper_id")))
        for fos_id, _, _ in direct_items:
            for node, _ in ancestors(fos_id):
                all_nodes.add(node)

    node_ids = sorted(all_nodes)
    node_texts = [id_to_name.get(node, node).replace("_", " ") for node in node_ids]
    paper_texts: Dict[str, str] = {}
    if args.use_paper_text:
        paper_texts = load_requested_paper_texts(
            Path(args.dblp_json),
            requested_paper_ids,
            args.paper_text_max_chars,
            args.progress_every,
        )
        print(f"paper_texts_loaded={len(paper_texts)}/{len(requested_paper_ids)}")

    paper_ids = sorted(paper_texts)
    corpus = node_texts + [paper_texts[pid] for pid in paper_ids]
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), analyzer="word", min_df=1)
    tfidf = vectorizer.fit_transform(corpus)
    dim = min(args.dim, max(2, min(tfidf.shape) - 1))
    svd = TruncatedSVD(n_components=dim, random_state=0)
    dense = normalize(svd.fit_transform(tfidf))
    semantic = {node: dense[i] for i, node in enumerate(node_ids)}
    paper_vectors = {
        pid: dense[len(node_ids) + i] for i, pid in enumerate(paper_ids)
    }

    rows = []
    for expert_id, direct_items in expert_direct.items():
        embeddings = build_node_embeddings_for_expert(
            direct_items,
            expert_evidence.get(expert_id, {}),
            ancestors,
            child_to_parents,
            semantic,
            paper_vectors,
        )
        for node, rec in embeddings.items():
            top_children = sorted(
                rec["attention_children"].items(), key=lambda x: x[1], reverse=True
            )[:5]
            top_papers = sorted(
                rec["attention_papers"].items(), key=lambda x: x[1], reverse=True
            )[:5]
            rows.append(
                {
                    "expert_id": expert_id,
                    "expert_name": expert_names.get(expert_id, expert_id),
                    "node_id": node,
                    "node_name": id_to_name.get(node, node),
                    "node_level": id_to_level.get(node, ""),
                    "is_direct_node": int(rec["is_direct_node"]),
                    "subtree_weight_sum": f"{rec['subtree_weight_sum']:.6f}",
                    "direct_leaf_count": rec["direct_leaf_count"],
                    "child_count": rec["child_count"],
                    "evidence_paper_count": rec["evidence_paper_count"],
                    "top_attention_children": json.dumps(
                        [
                            {
                                "node_id": c,
                                "node_name": id_to_name.get(c, c),
                                "attention": round(w, 6),
                            }
                            for c, w in top_children
                        ],
                        ensure_ascii=False,
                    ),
                    "top_evidence_papers": json.dumps(
                        [
                            {
                                "paper_id": pid,
                                "attention": round(w, 6),
                                "text": paper_texts.get(pid, "")[:160],
                            }
                            for pid, w in top_papers
                        ],
                        ensure_ascii=False,
                    ),
                    "embedding": json.dumps(
                        [round(float(x), 6) for x in rec["embedding"]],
                        ensure_ascii=False,
                    ),
                }
            )

    out_path = out_dir / "sample_expert_node_embeddings.tsv"
    with out_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "expert_id",
            "expert_name",
            "node_id",
            "node_name",
            "node_level",
            "is_direct_node",
            "subtree_weight_sum",
            "direct_leaf_count",
            "child_count",
            "evidence_paper_count",
            "top_attention_children",
            "top_evidence_papers",
            "embedding",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    summary_path = out_dir / "sample_summary.tsv"
    with summary_path.open("w", encoding="utf-8") as f:
        f.write("metric\tvalue\n")
        f.write(f"experts\t{len(expert_direct)}\n")
        f.write(f"unique_taxonomy_nodes\t{len(node_ids)}\n")
        f.write(f"expert_node_embeddings\t{len(rows)}\n")
        f.write(f"embedding_dim\t{dim}\n")
        f.write(f"max_profile_nodes\t{args.max_profile_nodes}\n")
        f.write(f"requested_paper_texts\t{len(requested_paper_ids)}\n")
        f.write(f"loaded_paper_texts\t{len(paper_texts)}\n")
        if args.use_paper_text:
            f.write("node_embedding_method\tTFIDF_SVD_node_semantic_plus_direct_paper_attention_plus_child_attention\n")
        else:
            f.write("node_embedding_method\tTFIDF_SVD_node_semantic_plus_child_attention\n")

    print(f"experts={len(expert_direct)}")
    print(f"unique_taxonomy_nodes={len(node_ids)}")
    print(f"expert_node_embeddings={len(rows)}")
    print(f"embedding_dim={dim}")
    print(f"output={out_path}")
    print(f"summary={summary_path}")


if __name__ == "__main__":
    main()
