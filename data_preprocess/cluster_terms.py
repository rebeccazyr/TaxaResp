#!/usr/bin/env python3
"""Cluster similar terms within each semantic category.

This script reads the ``*_terms.txt`` files created by ``export_term_sets.py``
and groups near-duplicate phrases using a simple character n-gram TF-IDF
representation with cosine similarity. Terms whose cosine similarity exceeds the
specified threshold are placed in the same cluster. For each category the script
writes two artifacts:

1. ``{category}_term_clusters.csv``: every cluster ID, its representative term,
   size, and the list of member terms.
2. ``{category}_terms_clustered.txt``: only the representative term from each
   cluster (useful for downstream steps such as definition generation).

Example:
$ python data_preprocess/cluster_terms.py \
    --terms-dir data_preprocess \
    --output-dir data_preprocess/clustered_terms \
    --similarity-threshold 0.7
"""

from __future__ import annotations

import argparse
import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.neighbors import NearestNeighbors
except ImportError as exc:  # pragma: no cover - optional dependency
    raise SystemExit(
        "scikit-learn is required for clustering. Install it via 'pip install scikit-learn'."
    ) from exc

from scipy import sparse

import numpy as np

CATEGORIES = ("data", "task", "method", "domain")
DEFAULT_TERMS_DIR = Path(__file__).parent
DEFAULT_OUTPUT_DIR = DEFAULT_TERMS_DIR / "clustered_terms"


@dataclass
class Cluster:
    id: int
    members: List[str]

    @property
    def representative(self) -> str:
        # Choose the shortest term; tie-breaker is case-insensitive order.
        return min(self.members, key=lambda term: (len(term), term.lower()))

    @property
    def size(self) -> int:
        return len(self.members)


class UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, item: int) -> int:
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])
        return self.parent[item]

    def union(self, a: int, b: int) -> None:
        root_a = self.find(a)
        root_b = self.find(b)
        if root_a == root_b:
            return
        if self.rank[root_a] < self.rank[root_b]:
            root_a, root_b = root_b, root_a
        self.parent[root_b] = root_a
        if self.rank[root_a] == self.rank[root_b]:
            self.rank[root_a] += 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cluster similar terms and emit deduplicated representative lists.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--terms-dir",
        type=Path,
        default=DEFAULT_TERMS_DIR,
        help="Directory containing data_terms.txt, method_terms.txt, etc.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Where clustered outputs (CSVs + representative TXT files) will be written.",
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.95,
        help="Cosine similarity threshold for linking two terms into the same cluster.",
    )
    parser.add_argument(
        "--min-cluster-size",
        type=int,
        default=1,
        help="Only export clusters whose size is >= this value.",
    )
    parser.add_argument(
        "--ngram-min",
        type=int,
        default=3,
        help="Minimum n-gram size for the TF-IDF character analyzer.",
    )
    parser.add_argument(
        "--ngram-max",
        type=int,
        default=5,
        help="Maximum n-gram size for the TF-IDF character analyzer.",
    )
    parser.add_argument(
        "--embedding-model",
        default="",
        help=(
            "Optional sentence-transformer checkpoint (e.g., 'allenai/specter2_base') "
            "used for a secondary semantic clustering pass."
        ),
    )
    parser.add_argument(
        "--embedding-threshold",
        type=float,
        default=0.8,
        help="Cosine similarity threshold for the embedding-based clustering stage.",
    )
    parser.add_argument(
        "--embedding-batch-size",
        type=int,
        default=16,
        help="Batch size for encoding terms with the embedding model.",
    )
    parser.add_argument(
        "--embedding-device",
        default="",
        help="Optional device string passed to the sentence-transformer (e.g., 'cpu', 'cuda').",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress logging.",
    )
    return parser.parse_args()


def read_terms(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"Could not find term file: {path}")
    terms = []
    seen = set()
    with path.open(encoding="utf-8") as handle:
        for raw in handle:
            term = raw.strip()
            if term and term not in seen:
                seen.add(term)
                terms.append(term)
    return terms


def vectorize_terms(terms: Sequence[str], *, ngram_min: int, ngram_max: int):
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(ngram_min, ngram_max))
    matrix = vectorizer.fit_transform(terms)
    return matrix


def build_similarity_graph(matrix, threshold: float) -> UnionFind:
    term_count = matrix.shape[0]
    uf = UnionFind(term_count)
    if term_count <= 1:
        return uf
    sim_matrix = cosine_similarity(matrix, dense_output=False)
    if not sparse.isspmatrix(sim_matrix):
        sim_matrix = sparse.csr_matrix(sim_matrix)
    sim_matrix.setdiag(0)
    sim_matrix.eliminate_zeros()
    coo = sim_matrix.tocoo()
    for i, j, value in zip(coo.row, coo.col, coo.data):
        if value >= threshold:
            uf.union(i, j)
    return uf


def extract_clusters(terms: Sequence[str], uf: UnionFind) -> List[Cluster]:
    groups: Dict[int, List[str]] = {}
    for idx, term in enumerate(terms):
        root = uf.find(idx)
        groups.setdefault(root, []).append(term)
    ordered_groups = sorted(groups.items(), key=lambda item: item[0])
    clusters = [
        Cluster(id=index, members=sorted(members, key=lambda t: t.lower()))
        for index, (_root, members) in enumerate(ordered_groups)
    ]
    return clusters


def semantic_recluster(
    clusters: Sequence[Cluster],
    *,
    model_name: str,
    threshold: float,
    batch_size: int,
    device: Optional[str] = None,
) -> List[Cluster]:
    if not clusters:
        return list(clusters)
    if not (0.0 < threshold <= 1.0):
        raise ValueError("embedding-threshold must be within (0, 1].")
    if batch_size <= 0:
        raise ValueError("embedding-batch-size must be >= 1.")
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise SystemExit(
            "sentence-transformers is required for --embedding-model."
            " Install it via 'pip install sentence-transformers'."
        ) from exc

    encoder = SentenceTransformer(model_name, device=device or None)
    representatives = [cluster.representative for cluster in clusters]
    embeddings = encoder.encode(
        representatives,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    uf = build_embedding_similarity_graph(embeddings, threshold)
    return merge_clusters_from_union(clusters, uf)


def build_embedding_similarity_graph(embeddings: np.ndarray, threshold: float) -> UnionFind:
    term_count = embeddings.shape[0]
    uf = UnionFind(term_count)
    if term_count <= 1:
        return uf
    # radius_neighbors_graph returns distances where cosine distance = 1 - cosine similarity
    radius = max(0.0, 1.0 - threshold) + 1e-6
    neighbors = NearestNeighbors(metric="cosine", algorithm="brute")
    neighbors.fit(embeddings)
    graph = neighbors.radius_neighbors_graph(
        embeddings, radius=radius, mode="distance"
    ).tocoo()
    for i, j, dist in zip(graph.row, graph.col, graph.data):
        if i >= j:
            continue
        cosine_sim = 1.0 - dist
        if cosine_sim >= threshold:
            uf.union(i, j)
    return uf


def merge_clusters_from_union(clusters: Sequence[Cluster], uf: UnionFind) -> List[Cluster]:
    grouped: Dict[int, List[int]] = {}
    for idx in range(len(clusters)):
        root = uf.find(idx)
        grouped.setdefault(root, []).append(idx)
    ordered = sorted(grouped.values(), key=lambda ids: min(clusters[i].id for i in ids))
    merged: List[Cluster] = []
    for new_id, member_indices in enumerate(ordered):
        members = sorted(
            {term for idx in member_indices for term in clusters[idx].members},
            key=lambda t: t.lower(),
        )
        merged.append(Cluster(id=new_id, members=members))
    return merged


def write_outputs(
    category: str,
    clusters: Iterable[Cluster],
    output_dir: Path,
    *,
    min_cluster_size: int,
) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    cluster_csv = output_dir / f"{category}_term_clusters.csv"
    rep_txt = output_dir / f"{category}_terms_clustered.txt"

    clusters = [cluster for cluster in clusters if cluster.size >= min_cluster_size]
    clusters.sort(key=lambda c: (-c.size, c.representative.lower()))

    with cluster_csv.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["cluster_id", "representative", "size", "members"])
        for cluster in clusters:
            writer.writerow(
                [cluster.id, cluster.representative, cluster.size, ";".join(cluster.members)]
            )

    with rep_txt.open("w", encoding="utf-8") as txtfile:
        for cluster in clusters:
            txtfile.write(f"{cluster.representative}\n")

    return len(clusters)


def main() -> None:
    args = parse_args()
    log_level = logging.WARNING if args.quiet else logging.INFO
    logging.basicConfig(level=log_level, format="[%(levelname)s] %(message)s")
    for category in CATEGORIES:
        term_path = args.terms_dir / f"{category}_terms.txt"
        terms = read_terms(term_path)
        if not terms:
            logging.info("Skipping %s (no terms)", category)
            continue
        logging.info("Processing %s: %s terms", category, len(terms))
        matrix = vectorize_terms(terms, ngram_min=args.ngram_min, ngram_max=args.ngram_max)
        uf = build_similarity_graph(matrix, args.similarity_threshold)
        clusters = extract_clusters(terms, uf)
        if args.embedding_model:
            logging.info(
                "Applying embedding clustering (%s, threshold=%.2f) to %s representatives",
                args.embedding_model,
                args.embedding_threshold,
                len(clusters),
            )
            clusters = semantic_recluster(
                clusters,
                model_name=args.embedding_model,
                threshold=args.embedding_threshold,
                batch_size=args.embedding_batch_size,
                device=args.embedding_device or None,
            )
        kept = write_outputs(
            category,
            clusters,
            args.output_dir,
            min_cluster_size=args.min_cluster_size,
        )
        logging.info(
            "Finished %s: %s clusters saved to %s",
            category,
            kept,
            args.output_dir,
        )


if __name__ == "__main__":
    main()
