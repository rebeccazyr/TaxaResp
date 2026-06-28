#!/usr/bin/env python3
"""Generate task-node research expertise text from task_node_prompts.jsonl.

This script is optional. You can replace it with any local/server LLM as long as
the output JSONL has: paper_id, node_id, requirement.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

from embedding_pipeline_utils import read_jsonl


SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "paper_id": {"type": "string"},
        "node_id": {"type": "string"},
        "requirement": {"type": "string"},
        "role_description": {"type": "string"},
        "key_capabilities": {"type": "array", "items": {"type": "string"}},
        "evidence_from_abstract": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "paper_id",
        "node_id",
        "requirement",
        "role_description",
        "key_capabilities",
        "evidence_from_abstract",
    ],
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate task-node research expertise text with an LLM")
    p.add_argument("--prompts-jsonl", required=True)
    p.add_argument("--out-jsonl", required=True)
    p.add_argument(
        "--backend",
        choices=("together", "openai"),
        default="together",
        help="LLM provider. Together uses chat.completions; OpenAI uses responses.",
    )
    p.add_argument("--model", default="openai/gpt-oss-120b")
    p.add_argument(
        "--api-key",
        default="",
        help="Provider API key. Defaults to TOGETHER_API_KEY or OPENAI_API_KEY.",
    )
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-rows", type=int, default=0, help="0 means all")
    p.add_argument("--sleep-seconds", type=float, default=0.0)
    p.add_argument("--retry", type=int, default=3)
    p.add_argument("--retry-sleep", type=float, default=2.0)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--max-in-flight", type=int, default=64)
    return p.parse_args()


def load_done(path: Path) -> set:
    if not path.exists():
        return set()
    done = set()
    for obj in read_jsonl(path):
        if obj.get("paper_id") and obj.get("node_id"):
            done.add((str(obj["paper_id"]), str(obj["node_id"])))
    return done


def system_prompt() -> str:
    return (
        "You generate JSON task-node research expertise profiles for research "
        "team formation. For the given paper and taxonomy node, write the "
        "task-specific research expertise that should align with experts' "
        "historical paper title/abstract embeddings. Use paper-like technical "
        "language, not job-ad or staffing language. Return only valid JSON."
    )


def user_prompt(prompt: str) -> str:
    return (
        f"{prompt}\n\n"
        "Write the `requirement` and `role_description` fields as the same "
        "concise research expertise paragraph. Do not start with phrases like "
        "'This task requires an expert who can'. Prefer noun-phrase and "
        "paper-abstract style language such as methods, systems, datasets, "
        "problem settings, evaluation criteria, and domain concepts. The text "
        "must be specific to the task paper and taxonomy node. You must use "
        "abstract details related to the current subtree skills, but do not "
        "repeat the full paper contribution for every node. Keep only the "
        "task-specific capability that this node is responsible for. Do not "
        "turn sibling methods, sibling systems, or broad paper context into this "
        "node's required capability. For a leaf or small subtree, the generated "
        "paragraph should be narrow: describe the current subtree's direct role "
        "in the paper, with other abstract details used only as minimal context "
        "when necessary. Do not simply restate the taxonomy node name or subtree "
        "skills. Fill "
        "`evidence_from_abstract` with 2-4 short phrases from the abstract that "
        "justify the generated expertise text."
    )


def extract_json_object(text: str) -> dict:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def normalize_obj(obj: dict, row: dict) -> dict:
    role_description = str(
        obj.get("role_description") or obj.get("requirement") or ""
    ).strip()
    if not role_description:
        raise ValueError("LLM response is missing requirement/role_description text")

    key_capabilities = obj.get("key_capabilities")
    if not isinstance(key_capabilities, list):
        key_capabilities = []

    evidence = obj.get("evidence_from_abstract")
    if not isinstance(evidence, list):
        evidence = []

    return {
        "paper_id": str(row["paper_id"]),
        "node_id": str(row["node_id"]),
        "requirement": role_description,
        "role_description": role_description,
        "key_capabilities": [str(x) for x in key_capabilities],
        "evidence_from_abstract": [str(x) for x in evidence],
    }


def generate_with_together(client, args: argparse.Namespace, row: dict) -> dict:
    response = client.chat.completions.create(
        model=args.model,
        messages=[
            {"role": "system", "content": system_prompt()},
            {"role": "user", "content": user_prompt(row["prompt"])},
        ],
        temperature=args.temperature,
    )
    content = response.choices[0].message.content
    return normalize_obj(extract_json_object(content), row)


def generate_with_openai(client, args: argparse.Namespace, row: dict) -> dict:
    if not hasattr(client, "responses"):
        response = client.chat.completions.create(
            model=args.model,
            messages=[
                {"role": "system", "content": system_prompt()},
                {"role": "user", "content": user_prompt(row["prompt"])},
            ],
            temperature=args.temperature,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content
        return normalize_obj(extract_json_object(content), row)

    response = client.responses.create(
        model=args.model,
        input=[
            {"role": "system", "content": system_prompt()},
            {"role": "user", "content": user_prompt(row["prompt"])},
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
    return normalize_obj(json.loads(response.output_text), row)


def generate_with_retry(client, args: argparse.Namespace, row: dict) -> dict:
    last_error = None
    for attempt in range(1, args.retry + 1):
        try:
            if args.backend == "together":
                return generate_with_together(client, args, row)
            return generate_with_openai(client, args, row)
        except Exception as exc:
            last_error = exc
            if attempt >= args.retry:
                raise
            time.sleep(args.retry_sleep)
    raise RuntimeError(f"LLM generation failed: {last_error}")


def build_client(args: argparse.Namespace):
    if args.backend == "together":
        from together import Together

        api_key = args.api_key or os.environ.get("TOGETHER_API_KEY", "")
        return Together(api_key=api_key) if api_key else Together()

    from openai import OpenAI

    api_key = args.api_key or os.environ.get("OPENAI_API_KEY", "")
    return OpenAI(api_key=api_key) if api_key else OpenAI()


def main() -> None:
    args = parse_args()

    client = build_client(args)
    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = load_done(out_path) if args.resume else set()
    mode = "a" if args.resume else "w"

    rows = list(read_jsonl(Path(args.prompts_jsonl)))
    if args.max_rows > 0:
        rows = rows[: args.max_rows]

    written = 0
    with out_path.open(mode, encoding="utf-8") as out:
        if args.workers <= 1:
            for idx, row in enumerate(rows, start=1):
                key = (str(row["paper_id"]), str(row["node_id"]))
                if key in done:
                    continue
                obj = generate_with_retry(client, args, row)
                out.write(json.dumps(obj, ensure_ascii=False) + "\n")
                out.flush()
                written += 1
                if written % 50 == 0:
                    print(f"progress written={written} idx={idx}/{len(rows)}", flush=True)
                if args.sleep_seconds > 0:
                    time.sleep(args.sleep_seconds)
        else:
            futures = {}
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                for idx, row in enumerate(rows, start=1):
                    key = (str(row["paper_id"]), str(row["node_id"]))
                    if key in done:
                        continue
                    future = executor.submit(generate_with_retry, client, args, row)
                    futures[future] = idx
                    while len(futures) >= args.max_in_flight:
                        completed, _ = wait(futures, return_when=FIRST_COMPLETED)
                        for completed_future in completed:
                            futures.pop(completed_future)
                            obj = completed_future.result()
                            out.write(json.dumps(obj, ensure_ascii=False) + "\n")
                            written += 1
                            if written % 50 == 0:
                                out.flush()
                                print(
                                    f"progress written={written} submitted_idx={idx}/{len(rows)}",
                                    flush=True,
                                )
                        if args.sleep_seconds > 0:
                            time.sleep(args.sleep_seconds)
                while futures:
                    completed, _ = wait(futures, return_when=FIRST_COMPLETED)
                    for completed_future in completed:
                        futures.pop(completed_future)
                        obj = completed_future.result()
                        out.write(json.dumps(obj, ensure_ascii=False) + "\n")
                        written += 1
                        if written % 50 == 0:
                            out.flush()
                            print(f"progress written={written}", flush=True)
                out.flush()

    print(f"written={written}")
    print(f"out_jsonl={out_path}")


if __name__ == "__main__":
    main()
