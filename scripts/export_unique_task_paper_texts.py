#!/usr/bin/env python3
"""Export one title/abstract text record per task paper from task_nodes.jsonl."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task-nodes-jsonl", default="output/hierec_embedding_server_inputs/task_nodes.jsonl")
    p.add_argument(
        "--out-jsonl",
        default="output/virtual_root_role_descriptions/task_paper_texts.jsonl",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    seen = set()
    rows = 0
    with Path(args.task_nodes_jsonl).open("r", encoding="utf-8") as f, out_path.open(
        "w", encoding="utf-8"
    ) as out:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            paper_id = str(obj["paper_id"])
            if paper_id in seen:
                continue
            seen.add(paper_id)
            text = " ".join(str(obj.get("task_paper_text") or "").split())
            if not text:
                continue
            out.write(
                json.dumps(
                    {
                        "id": paper_id,
                        "paper_id": paper_id,
                        "text": text,
                        "members": obj.get("members") or [],
                        "team_size": obj.get("team_size"),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            rows += 1
    print(f"wrote={out_path} rows={rows}")


if __name__ == "__main__":
    main()
