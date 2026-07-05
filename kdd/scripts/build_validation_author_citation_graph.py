#!/usr/bin/env python3
"""Build a pre-cutoff citation graph for authors in a validation task set.

Nodes are all author ids observed in the validation JSONL. Edges are induced by
historical DBLP paper citations before the cutoff year: if historical paper p
has validation-set author A and cites historical paper q with validation-set
author B, the directed edge A -> B receives a fractional weight. Each paper
citation contributes total weight 1, distributed across the Cartesian product
of source and target authors as 1 / (source_author_count * target_author_count).
Self-loops are skipped by default.
"""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build an author citation graph for all authors in a validation JSONL, "
            "using only DBLP citation relationships before a cutoff year."
        )
    )
    parser.add_argument(
        "--dblp-json",
        default="data/dblp/dblp.v12.json",
        help="Path to dblp.v12.json.",
    )
    parser.add_argument(
        "--validation-jsonl",
        default="outputs/temporal_task_splits_full/validation_2018.jsonl",
        help="Validation task JSONL whose authors define graph nodes.",
    )
    parser.add_argument(
        "--cutoff-year",
        type=int,
        default=2018,
        help="Use only papers with year < cutoff year for citation edges.",
    )
    parser.add_argument(
        "--out-dir",
        default="outputs/expert_citation_graph_valid2018_pre2018",
        help="Output directory for graph files.",
    )
    parser.add_argument(
        "--include-self-loops",
        action="store_true",
        help="Keep A -> A citation edges. By default they are skipped.",
    )
    parser.add_argument(
        "--flush-edge-threshold",
        type=int,
        default=1000000,
        help="Flush buffered unique directed edges to SQLite after this many keys.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=500000,
        help="Print progress every N parsed DBLP records. Use 0 to disable.",
    )
    return parser.parse_args()


def safe_int(value: object) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def iter_json_objects(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            text = raw.strip()
            if not text:
                continue
            if text[0] == ",":
                text = text[1:]
            if not text.startswith("{"):
                continue
            try:
                yield json.loads(text)
            except json.JSONDecodeError:
                continue


def iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def ordered_author_ids(obj: dict) -> List[str]:
    authors = obj.get("authors") or []
    if not isinstance(authors, list):
        return []
    seen = set()
    result: List[str] = []
    for author in authors:
        if not isinstance(author, dict):
            continue
        author_id = str(author.get("id", "")).strip()
        if not author_id or author_id in seen:
            continue
        seen.add(author_id)
        result.append(author_id)
    return result


def load_validation_authors(path: Path) -> Tuple[List[str], Dict[str, str], Dict[str, int], int]:
    validation_papers_by_author: Counter[str] = Counter()
    names_by_author: Dict[str, Counter[str]] = {}
    validation_papers = 0

    for obj in iter_jsonl(path):
        validation_papers += 1
        authors = obj.get("authors") or []
        if not isinstance(authors, list):
            continue
        seen_in_paper = set()
        for author in authors:
            if not isinstance(author, dict):
                continue
            author_id = str(author.get("id", "")).strip()
            if not author_id or author_id in seen_in_paper:
                continue
            seen_in_paper.add(author_id)
            validation_papers_by_author[author_id] += 1
            name = str(author.get("name", "")).strip()
            if name:
                names_by_author.setdefault(author_id, Counter())[name] += 1

    author_ids = sorted(
        validation_papers_by_author,
        key=lambda value: (0, int(value)) if value.isdigit() else (1, value),
    )
    display_names = {
        author_id: names_by_author.get(author_id, Counter()).most_common(1)[0][0]
        if names_by_author.get(author_id)
        else ""
        for author_id in author_ids
    }
    return author_ids, display_names, dict(validation_papers_by_author), validation_papers


def setup_sqlite(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA temp_store=FILE")
    conn.execute("DROP TABLE IF EXISTS directed_edges")
    conn.execute(
        "CREATE TABLE directed_edges ("
        "src INTEGER NOT NULL, "
        "dst INTEGER NOT NULL, "
        "weight REAL NOT NULL, "
        "PRIMARY KEY (src, dst)"
        ") WITHOUT ROWID"
    )
    conn.commit()
    return conn


def flush_edges(conn: sqlite3.Connection, edge_buffer: Dict[Tuple[int, int], float]) -> None:
    if not edge_buffer:
        return
    conn.executemany(
        "INSERT INTO directed_edges(src, dst, weight) VALUES (?, ?, ?) "
        "ON CONFLICT(src, dst) DO UPDATE SET weight = weight + excluded.weight",
        ((src, dst, weight) for (src, dst), weight in edge_buffer.items()),
    )
    conn.commit()
    edge_buffer.clear()


def export_directed_edges(
    conn: sqlite3.Connection,
    path: Path,
    author_ids: Sequence[str],
) -> Tuple[int, float]:
    row_count = 0
    weight_sum = 0.0
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["source_author_id", "target_author_id", "weight"])
        for src, dst, weight in conn.execute(
            "SELECT src, dst, weight FROM directed_edges ORDER BY src, dst"
        ):
            writer.writerow([author_ids[src], author_ids[dst], weight])
            row_count += 1
            weight_sum += float(weight)
    return row_count, weight_sum


def export_undirected_edges(
    conn: sqlite3.Connection,
    path: Path,
    author_ids: Sequence[str],
) -> Tuple[int, float]:
    row_count = 0
    weight_sum = 0.0
    query = (
        "SELECT "
        "CASE WHEN src < dst THEN src ELSE dst END AS a, "
        "CASE WHEN src < dst THEN dst ELSE src END AS b, "
        "SUM(weight) AS weight "
        "FROM directed_edges "
        "GROUP BY a, b "
        "ORDER BY a, b"
    )
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["author_id_a", "author_id_b", "weight"])
        for a, b, weight in conn.execute(query):
            writer.writerow([author_ids[a], author_ids[b], weight])
            row_count += 1
            weight_sum += float(weight)
    return row_count, weight_sum


def load_author_weights(conn: sqlite3.Connection, author_count: int) -> Tuple[List[float], List[float]]:
    out_weights = [0.0] * author_count
    in_weights = [0.0] * author_count
    for src, weight in conn.execute("SELECT src, SUM(weight) FROM directed_edges GROUP BY src"):
        out_weights[src] = float(weight)
    for dst, weight in conn.execute("SELECT dst, SUM(weight) FROM directed_edges GROUP BY dst"):
        in_weights[dst] = float(weight)
    return out_weights, in_weights


def write_nodes(
    path: Path,
    author_ids: Sequence[str],
    display_names: Dict[str, str],
    validation_papers_by_author: Dict[str, int],
    historical_papers_by_author: Sequence[int],
    out_weights: Sequence[float],
    in_weights: Sequence[float],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(
            [
                "author_idx",
                "author_id",
                "display_name",
                "validation_papers",
                "historical_papers_pre_cutoff",
                "out_citation_weight",
                "in_citation_weight",
            ]
        )
        for idx, author_id in enumerate(author_ids):
            writer.writerow(
                [
                    idx,
                    author_id,
                    display_names.get(author_id, ""),
                    validation_papers_by_author.get(author_id, 0),
                    historical_papers_by_author[idx],
                    f"{out_weights[idx]:.12g}",
                    f"{in_weights[idx]:.12g}",
                ]
            )


def main() -> None:
    args = parse_args()
    dblp_path = Path(args.dblp_json)
    validation_path = Path(args.validation_jsonl)
    out_dir = Path(args.out_dir)
    if not dblp_path.exists():
        raise FileNotFoundError(f"DBLP JSON not found: {dblp_path}")
    if not validation_path.exists():
        raise FileNotFoundError(f"validation JSONL not found: {validation_path}")

    out_dir.mkdir(parents=True, exist_ok=True)
    sqlite_path = out_dir / "graph.sqlite"
    nodes_path = out_dir / "nodes.tsv"
    directed_edges_path = out_dir / "edges_directed.tsv"
    undirected_edges_path = out_dir / "edges_undirected.tsv"
    summary_path = out_dir / "_summary.tsv"

    author_ids, display_names, validation_papers_by_author, validation_papers = (
        load_validation_authors(validation_path)
    )
    author_to_idx = {author_id: idx for idx, author_id in enumerate(author_ids)}
    historical_papers_by_author = [0] * len(author_ids)

    print(
        f"validation_papers={validation_papers:,} validation_authors={len(author_ids):,}",
        flush=True,
    )

    paper_authors: Dict[int, Tuple[int, ...]] = {}
    parsed = 0
    pre_cutoff_records = 0
    indexed_papers = 0
    for obj in iter_json_objects(dblp_path):
        parsed += 1
        if args.progress_every > 0 and parsed % args.progress_every == 0:
            print(
                "index_progress "
                f"parsed={parsed:,} pre_cutoff={pre_cutoff_records:,} "
                f"indexed_papers={indexed_papers:,}",
                flush=True,
            )

        year = safe_int(obj.get("year"))
        if year is None or year >= args.cutoff_year:
            continue
        pre_cutoff_records += 1
        paper_id = safe_int(obj.get("id"))
        if paper_id is None:
            continue
        idxs = tuple(
            author_to_idx[author_id]
            for author_id in ordered_author_ids(obj)
            if author_id in author_to_idx
        )
        if not idxs:
            continue
        paper_authors[paper_id] = idxs
        indexed_papers += 1
        for idx in idxs:
            historical_papers_by_author[idx] += 1

    print(
        f"paper_author_index={len(paper_authors):,} pre_cutoff_records={pre_cutoff_records:,}",
        flush=True,
    )

    conn = setup_sqlite(sqlite_path)
    edge_buffer: Dict[Tuple[int, int], float] = {}
    parsed = 0
    citing_papers_with_refs = 0
    resolved_paper_citations = 0
    expanded_author_citation_events = 0
    expanded_author_citation_weight = 0.0
    for obj in iter_json_objects(dblp_path):
        parsed += 1
        if args.progress_every > 0 and parsed % args.progress_every == 0:
            print(
                "edge_progress "
                f"parsed={parsed:,} citing_papers={citing_papers_with_refs:,} "
                f"resolved_paper_citations={resolved_paper_citations:,} "
                f"buffer={len(edge_buffer):,}",
                flush=True,
            )

        year = safe_int(obj.get("year"))
        if year is None or year >= args.cutoff_year:
            continue
        paper_id = safe_int(obj.get("id"))
        if paper_id is None:
            continue
        src_idxs = paper_authors.get(paper_id)
        if not src_idxs:
            continue
        references = obj.get("references") or []
        if not isinstance(references, list) or not references:
            continue
        citing_papers_with_refs += 1

        for ref in references:
            ref_id = safe_int(ref)
            if ref_id is None:
                continue
            dst_idxs = paper_authors.get(ref_id)
            if not dst_idxs:
                continue
            resolved_paper_citations += 1
            edge_increment = 1.0 / (len(src_idxs) * len(dst_idxs))
            for src in src_idxs:
                for dst in dst_idxs:
                    if src == dst and not args.include_self_loops:
                        continue
                    edge_buffer[(src, dst)] = edge_buffer.get((src, dst), 0.0) + edge_increment
                    expanded_author_citation_events += 1
                    expanded_author_citation_weight += edge_increment

        if len(edge_buffer) >= args.flush_edge_threshold:
            flush_edges(conn, edge_buffer)

    flush_edges(conn, edge_buffer)

    directed_edge_count, directed_weight_sum = export_directed_edges(
        conn, directed_edges_path, author_ids
    )
    undirected_edge_count, undirected_weight_sum = export_undirected_edges(
        conn, undirected_edges_path, author_ids
    )
    out_weights, in_weights = load_author_weights(conn, len(author_ids))
    write_nodes(
        nodes_path,
        author_ids,
        display_names,
        validation_papers_by_author,
        historical_papers_by_author,
        out_weights,
        in_weights,
    )

    with summary_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["metric", "value"])
        writer.writerow(["validation_jsonl", str(validation_path)])
        writer.writerow(["dblp_json", str(dblp_path)])
        writer.writerow(["cutoff_year", args.cutoff_year])
        writer.writerow(["include_self_loops", int(args.include_self_loops)])
        writer.writerow(["edge_weighting", "cartesian_normalized"])
        writer.writerow(["validation_papers", validation_papers])
        writer.writerow(["validation_authors", len(author_ids)])
        writer.writerow(["pre_cutoff_records", pre_cutoff_records])
        writer.writerow(["indexed_pre_cutoff_papers_with_validation_authors", indexed_papers])
        writer.writerow(["citing_papers_with_references", citing_papers_with_refs])
        writer.writerow(["resolved_pre_cutoff_paper_citations", resolved_paper_citations])
        writer.writerow(["expanded_author_citation_events", expanded_author_citation_events])
        writer.writerow(
            ["expanded_author_citation_weight", f"{expanded_author_citation_weight:.12g}"]
        )
        writer.writerow(["directed_edges", directed_edge_count])
        writer.writerow(["directed_weight_sum", f"{directed_weight_sum:.12g}"])
        writer.writerow(["undirected_edges", undirected_edge_count])
        writer.writerow(["undirected_weight_sum", f"{undirected_weight_sum:.12g}"])
        writer.writerow(["sqlite_graph", str(sqlite_path)])
        writer.writerow(["nodes_tsv", str(nodes_path)])
        writer.writerow(["directed_edges_tsv", str(directed_edges_path)])
        writer.writerow(["undirected_edges_tsv", str(undirected_edges_path)])

    conn.close()

    print(f"nodes={len(author_ids):,}")
    print(f"directed_edges={directed_edge_count:,} directed_weight={directed_weight_sum:,.6f}")
    print(f"undirected_edges={undirected_edge_count:,} undirected_weight={undirected_weight_sum:,.6f}")
    print(f"out_dir={out_dir}")
    print(f"summary={summary_path}")


if __name__ == "__main__":
    main()
