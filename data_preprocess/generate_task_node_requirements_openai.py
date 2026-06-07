#!/usr/bin/env python3
"""Generate task-node requirements from task_node_prompts.jsonl with OpenAI.

This script is optional. You can replace it with any local/server LLM as long as
the output JSONL has: paper_id, node_id, requirement.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from embedding_pipeline_utils import read_jsonl


SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "paper_id": {"type": "string"},
        "node_id": {"type": "string"},
        "requirement": {"type": "string"},
        "key_capabilities": {"type": "array", "items": {"type": "string"}},
        "evidence_from_abstract": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "paper_id",
        "node_id",
        "requirement",
        "key_capabilities",
        "evidence_from_abstract",
    ],
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate task-node requirements with OpenAI")
    p.add_argument("--prompts-jsonl", required=True)
    p.add_argument("--out-jsonl", required=True)
    p.add_argument("--model", default="gpt-4.1-mini")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-rows", type=int, default=0, help="0 means all")
    p.add_argument("--sleep-seconds", type=float, default=0.0)
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def load_done(path: Path) -> set:
    if not path.exists():
        return set()
    done = set()
    for obj in read_jsonl(path):
        if obj.get("paper_id") and obj.get("node_id"):
            done.add((str(obj["paper_id"]), str(obj["node_id"])))
    return done


def main() -> None:
    args = parse_args()
    from openai import OpenAI

    client = OpenAI()
    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = load_done(out_path) if args.resume else set()
    mode = "a" if args.resume else "w"

    rows = list(read_jsonl(Path(args.prompts_jsonl)))
    if args.max_rows > 0:
        rows = rows[: args.max_rows]

    written = 0
    with out_path.open(mode, encoding="utf-8") as out:
        for idx, row in enumerate(rows, start=1):
            key = (str(row["paper_id"]), str(row["node_id"]))
            if key in done:
                continue
            response = client.responses.create(
                model=args.model,
                input=[
                    {
                        "role": "system",
                        "content": (
                            "You generate JSON task-node requirements for "
                            "research team formation. Return only valid JSON."
                        ),
                    },
                    {"role": "user", "content": row["prompt"]},
                ],
                temperature=args.temperature,
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "task_node_requirement",
                        "schema": SCHEMA,
                        "strict": True,
                    }
                },
            )
            obj = json.loads(response.output_text)
            obj["paper_id"] = str(row["paper_id"])
            obj["node_id"] = str(row["node_id"])
            out.write(json.dumps(obj, ensure_ascii=False) + "\n")
            out.flush()
            written += 1
            if written % 50 == 0:
                print(f"progress written={written} idx={idx}/{len(rows)}")
            if args.sleep_seconds > 0:
                time.sleep(args.sleep_seconds)

    print(f"written={written}")
    print(f"out_jsonl={out_path}")


if __name__ == "__main__":
    main()
