#!/usr/bin/env python3
"""Export target paper title/abstract texts for Stage-1 splits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sample-jsonl", required=True)
    p.add_argument("--out-jsonl", required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    with Path(args.sample_jsonl).open("r", encoding="utf-8") as src, out_path.open("w", encoding="utf-8") as out:
        for line in src:
            if not line.strip():
                continue
            row = json.loads(line)
            paper_id = str(row.get("paper_id") or row.get("id") or "").strip()
            title = " ".join(str(row.get("title") or "").split())
            abstract = " ".join(str(row.get("abstract") or "").split())
            text = f"{title}. {abstract}".strip()
            if text.endswith(". ."):
                text = text[:-2]
            if not paper_id or not text:
                continue
            out.write(json.dumps({"id": paper_id, "paper_id": paper_id, "text": text}, ensure_ascii=False))
            out.write("\n")
            rows += 1
    print(f"records={rows}")
    print(f"out_jsonl={out_path}")


if __name__ == "__main__":
    main()
