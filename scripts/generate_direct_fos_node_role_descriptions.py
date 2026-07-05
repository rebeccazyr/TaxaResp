#!/usr/bin/env python3
"""Generate per-paper role-description prompts for direct FoS task nodes.

This script uses pre-completion/direct paper FoS nodes, not ancestor-completed
task taxonomy nodes. It groups all direct positive-weight FoS nodes for one
paper into a single prompt so the LLM can make node descriptions distinct.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


DEFAULT_FILTERED_IDS = (
    "output/cross_domain_filter_direct_fos_by_level/"
    "level1_direct_fos_cross_domain_paper_ids.txt"
)
DEFAULT_DIRECT_NODES = (
    "output/cross_domain_filter_direct_fos_by_level/direct_fos_task_nodes.tsv"
)
DEFAULT_TASK_NODES = "output/hierec_embedding_server_inputs/task_nodes.jsonl"
DEFAULT_OUT_DIR = "output/direct_fos_node_role_descriptions_level1_cross_domain"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--paper-ids", default=DEFAULT_FILTERED_IDS)
    p.add_argument("--direct-fos-nodes", default=DEFAULT_DIRECT_NODES)
    p.add_argument("--task-nodes-jsonl", default=DEFAULT_TASK_NODES)
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    p.add_argument("--limit", type=int, default=3, help="0 means all selected papers")
    p.add_argument("--generate", action="store_true", help="Call the LLM; default only writes prompts")
    p.add_argument("--backend", choices=("together", "openai"), default="together")
    p.add_argument("--model", default="openai/gpt-oss-120b")
    p.add_argument("--api-key", default="")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--retry", type=int, default=3)
    p.add_argument("--retry-sleep", type=float, default=2.0)
    p.add_argument("--sleep-seconds", type=float, default=0.0)
    p.add_argument("--workers", type=int, default=1, help="Concurrent LLM requests")
    return p.parse_args()


def read_selected_paper_ids(path: Path, limit: int) -> List[str]:
    ids = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return ids[:limit] if limit > 0 else ids


def read_paper_texts_from_task_nodes(path: Path, paper_ids: set[str]) -> Dict[str, str]:
    texts: Dict[str, str] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            paper_id = str(row.get("paper_id", ""))
            if paper_id in paper_ids and paper_id not in texts:
                texts[paper_id] = str(row.get("task_paper_text") or "")
                if len(texts) == len(paper_ids):
                    break
    return texts


def read_direct_nodes(path: Path, paper_ids: set[str]) -> Dict[str, List[dict]]:
    by_paper: Dict[str, List[dict]] = {paper_id: [] for paper_id in paper_ids}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            paper_id = str(row.get("paper_id", ""))
            if paper_id not in by_paper:
                continue
            try:
                weight = float(row.get("weight") or 0.0)
            except ValueError:
                weight = 0.0
            if weight <= 0:
                continue
            by_paper[paper_id].append(
                {
                    "node_id": str(row.get("fos_id", "")),
                    "node_name": str(row.get("fos_name", "")),
                    "node_level": str(row.get("node_level", "")),
                    "weight": weight,
                    "raw_skill": str(row.get("raw_skill", "")),
                }
            )
    for rows in by_paper.values():
        rows.sort(key=lambda r: (int(r["node_level"]) if r["node_level"].isdigit() else 99, r["node_name"]))
    return by_paper


def system_prompt() -> str:
    return (
        "You generate JSON node-specific research expertise profiles for "
        "research team formation. Return only valid JSON."
    )


def build_prompt(paper_id: str, title_abstract: str, nodes: Sequence[dict]) -> str:
    node_lines = "\n".join(
        f"{node['node_id']} | {node['node_name']} | {node['node_level']}"
        for node in nodes
    )
    return f"""You are analyzing a research paper as a team-formation task.

You are given a paper (title + abstract) and a LIST of taxonomy nodes
for this paper. For EACH node, write a short node-specific research-
expertise paragraph describing the SINGLE technical facet of this paper
that this node is responsible for. The paragraphs will be embedded and
compared against experts' historical paper embeddings -- write in a
paper-like research-expertise style, and make the paragraphs
DISCRIMINATE between nodes.

Paper id: {paper_id}
Title/abstract:
{title_abstract}

Taxonomy nodes (id | name | level):
{node_lines}

REQUIREMENTS:
1. ONE FACET PER NODE. Each paragraph covers only the part of the paper
   tied to this node. Never restate the whole paper in every node.
2. NODES MUST DIFFER. No two paragraphs should be near-identical. Use the
   fact that you see all nodes at once to push each toward a distinct facet.
3. A broad/high-level node must NOT absorb the content of its more specific
   siblings. If a node is broad, describe the paper's facet at the level of
   generality that node actually owns -- do not pull in sibling specifics.
4. evidence_from_abstract: 1-3 short spans. Prefer non-overlapping sentences
   or clauses across nodes. If a sentence genuinely contains two concepts,
   two nodes may share it but must highlight different clauses/aspects.
5. Avoid phrases like "This task requires an expert who can".

OUTPUT -- JSON array, one object per node:
{{ paper_id, node_id, role_description, key_capabilities, evidence_from_abstract }}

Return the JSON array in exactly the same node order as the input list."""


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_client(args: argparse.Namespace):
    if args.backend == "together":
        from together import Together

        api_key = args.api_key or os.environ.get("TOGETHER_API_KEY", "")
        return Together(api_key=api_key) if api_key else Together()

    from openai import OpenAI

    api_key = args.api_key or os.environ.get("OPENAI_API_KEY", "")
    return OpenAI(api_key=api_key) if api_key else OpenAI()


def extract_json_array(text: str) -> list:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned).strip()
    try:
        obj = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\[.*\]", cleaned, flags=re.DOTALL)
        if not match:
            raise
        obj = json.loads(match.group(0))
    if not isinstance(obj, list):
        raise ValueError("LLM response is not a JSON array")
    return obj


def call_llm(client, args: argparse.Namespace, prompt: str) -> list:
    if args.backend == "together":
        response = client.chat.completions.create(
            model=args.model,
            messages=[
                {"role": "system", "content": system_prompt()},
                {"role": "user", "content": prompt},
            ],
            temperature=args.temperature,
        )
        return extract_json_array(response.choices[0].message.content)

    response = client.chat.completions.create(
        model=args.model,
        messages=[
            {"role": "system", "content": system_prompt()},
            {"role": "user", "content": prompt},
        ],
        temperature=args.temperature,
    )
    content = response.choices[0].message.content
    return extract_json_array(content)


def call_with_retry(client, args: argparse.Namespace, prompt: str) -> list:
    last_error = None
    for attempt in range(1, args.retry + 1):
        try:
            return call_llm(client, args, prompt)
        except Exception as exc:  # noqa: BLE001 - preserve retry diagnostics
            last_error = exc
            if attempt >= args.retry:
                raise
            time.sleep(args.retry_sleep)
    raise RuntimeError(f"LLM generation failed: {last_error}")


def normalize_result(obj: dict, paper_id: str, input_node_ids: set[str]) -> dict:
    node_id = str(obj.get("node_id", ""))
    if node_id not in input_node_ids:
        raise ValueError(f"Unexpected node_id in LLM response for paper {paper_id}: {node_id}")
    return {
        "id": f"{paper_id}::{node_id}",
        "paper_id": paper_id,
        "node_id": node_id,
        "role_description": str(obj.get("role_description") or "").strip(),
        "key_capabilities": obj.get("key_capabilities") if isinstance(obj.get("key_capabilities"), list) else [],
        "evidence_from_abstract": (
            obj.get("evidence_from_abstract")
            if isinstance(obj.get("evidence_from_abstract"), list)
            else []
        ),
    }


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    paper_ids = read_selected_paper_ids(Path(args.paper_ids), args.limit)
    paper_id_set = set(paper_ids)
    paper_texts = read_paper_texts_from_task_nodes(Path(args.task_nodes_jsonl), paper_id_set)
    nodes_by_paper = read_direct_nodes(Path(args.direct_fos_nodes), paper_id_set)

    prompt_rows = []
    for paper_id in paper_ids:
        nodes = nodes_by_paper.get(paper_id, [])
        if not nodes:
            raise SystemExit(f"No direct FoS nodes found for paper_id={paper_id}")
        title_abstract = paper_texts.get(paper_id, "")
        if not title_abstract:
            raise SystemExit(f"No title/abstract text found for paper_id={paper_id}")
        prompt_rows.append(
            {
                "paper_id": paper_id,
                "title_abstract": title_abstract,
                "nodes": nodes,
                "prompt": build_prompt(paper_id, title_abstract, nodes),
            }
        )

    prompts_path = out_dir / "direct_fos_node_role_prompts.jsonl"
    write_jsonl(prompts_path, prompt_rows)
    print(f"prompts={prompts_path}")
    print(f"papers={len(prompt_rows)} direct_nodes={sum(len(row['nodes']) for row in prompt_rows)}")

    if not args.generate:
        print("dry_run=1")
        return

    desc_path = out_dir / "direct_fos_node_role_descriptions.jsonl"
    raw_path = out_dir / "direct_fos_node_role_descriptions_raw_by_paper.jsonl"

    # Resume: skip papers already written to the output (kill-safe).
    done_papers: set[str] = set()
    if desc_path.exists():
        with desc_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    done_papers.add(str(json.loads(line).get("paper_id", "")))
                except json.JSONDecodeError:
                    continue
    if done_papers:
        print(f"resume=1 already_done_papers={len(done_papers)}")

    client = build_client(args)
    pending = [row for row in prompt_rows if row["paper_id"] not in done_papers]
    print(f"to_generate={len(pending)} workers={args.workers}", flush=True)

    # Append each paper as soon as it is generated, flushing so progress
    # survives interruption and is visible via `wc -l` on the output file.
    # A lock serializes writes so concurrent workers never interleave lines.
    import threading
    from concurrent.futures import ThreadPoolExecutor

    write_lock = threading.Lock()
    counter = {"done": 0}

    def process(row: dict) -> None:
        results = call_with_retry(client, args, row["prompt"])
        input_node_ids = {node["node_id"] for node in row["nodes"]}
        normalized = [normalize_result(obj, row["paper_id"], input_node_ids) for obj in results]
        with write_lock:
            raw_f.write(json.dumps({"paper_id": row["paper_id"], "results": results}, ensure_ascii=False) + "\n")
            raw_f.flush()
            for record in normalized:
                desc_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            desc_f.flush()
            counter["done"] += 1
            print(f"progress={counter['done']}/{len(pending)} paper={row['paper_id']} nodes={len(normalized)}", flush=True)

    with desc_path.open("a", encoding="utf-8") as desc_f, raw_path.open("a", encoding="utf-8") as raw_f:
        if args.workers <= 1:
            for row in pending:
                process(row)
                if args.sleep_seconds > 0:
                    time.sleep(args.sleep_seconds)
        else:
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                for _ in pool.map(process, pending):
                    pass

    print(f"raw={raw_path}")
    print(f"descriptions={desc_path}")


if __name__ == "__main__":
    main()
