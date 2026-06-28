#!/usr/bin/env python3
"""Generate one whole-task virtual-root role description per test paper."""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--task-nodes-jsonl",
        default="output/hierec_embedding_server_inputs/task_nodes.jsonl",
    )
    p.add_argument(
        "--out-jsonl",
        default="output/virtual_root_role_descriptions/root_role_texts.jsonl",
    )
    p.add_argument(
        "--backend",
        choices=("template", "together", "openai-compatible-chat"),
        default="template",
        help=(
            "template is deterministic and offline; together matches the "
            "task-node LLM backend; openai-compatible-chat uses an "
            "OpenAI-compatible chat endpoint."
        ),
    )
    p.add_argument("--base-url", default="")
    p.add_argument(
        "--api-key",
        default=os.environ.get("LLM_API_KEY") or os.environ.get("TOGETHER_API_KEY") or os.environ.get("OPENAI_API_KEY", ""),
    )
    p.add_argument("--model", default="openai/gpt-oss-120b")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--retry", type=int, default=3)
    p.add_argument("--retry-sleep", type=float, default=2.0)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--max-in-flight", type=int, default=16)
    return p.parse_args()


def read_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def first_sentence(text: str) -> str:
    text = " ".join(str(text or "").split())
    if not text:
        return ""
    # task_paper_text is written as "Title. Abstract"; this keeps the title.
    match = re.match(r"(.{20,240}?[.!?])\s+[A-Z0-9]", text)
    if match:
        return match.group(1).strip()
    return text[:220].strip()


def parse_weighted_skills(value: str, top_n: int = 5) -> List[str]:
    skills: List[Tuple[str, float]] = []
    for item in str(value or "").split(";"):
        item = item.strip()
        if not item:
            continue
        if ":" in item:
            name, raw_weight = item.rsplit(":", 1)
            try:
                weight = float(raw_weight)
            except ValueError:
                weight = 0.0
        else:
            name, weight = item, 0.0
        name = " ".join(name.split())
        if name:
            skills.append((name, weight))
    skills.sort(key=lambda x: (-x[1], x[0].lower()))
    return [name for name, _ in skills[:top_n]]


def collect_tasks(path: Path, limit: int = 0) -> List[dict]:
    by_paper: Dict[str, dict] = {}
    order: List[str] = []
    for row in read_jsonl(path):
        paper_id = str(row["paper_id"])
        if paper_id not in by_paper:
            by_paper[paper_id] = {
                "paper_id": paper_id,
                "team_size": row.get("team_size"),
                "members": row.get("members") or [],
                "task_paper_text": row.get("task_paper_text") or "",
                "all_task_skills": row.get("all_task_skills") or "",
                "node_count": 0,
            }
            order.append(paper_id)
            if limit and len(order) >= limit:
                # Still allow the current paper to collect all following rows
                # only when they are adjacent; task_nodes.jsonl is grouped.
                pass
        if paper_id in by_paper:
            by_paper[paper_id]["node_count"] += 1
        if limit and len(order) >= limit and paper_id != order[-1]:
            break
    return [by_paper[paper_id] for paper_id in order[: limit or None]]


def load_done(path: Path) -> Set[str]:
    if not path.exists():
        return set()
    done = set()
    for obj in read_jsonl(path):
        paper_id = obj.get("paper_id")
        if paper_id:
            done.add(str(paper_id))
    return done


def template_role(task: dict) -> str:
    title = first_sentence(task.get("task_paper_text") or "")
    skills = parse_weighted_skills(task.get("all_task_skills") or "", top_n=5)
    if skills:
        skill_phrase = ", ".join(skills[:-1]) + f", and {skills[-1]}" if len(skills) > 1 else skills[0]
        return (
            "This task requires an expert who can lead and integrate research on "
            f"{title.rstrip('.!?')}, coordinating the main technical directions around "
            f"{skill_phrase} into a coherent study."
        )
    return (
        "This task requires an expert who can lead and integrate the full research "
        f"agenda of {title.rstrip('.!?')} into a coherent study."
    )


def build_prompt(task: dict) -> str:
    return (
        "Generate a JSON virtual-root research expertise profile for research "
        "team formation. The virtual root represents the whole task, not any "
        "single taxonomy node.\n\n"
        "Write `requirement` and `role_description` as the same single sentence. "
        "The sentence must start with "
        '"This task requires an expert who can lead and integrate". '
        "Use the paper title/abstract and weighted task skills to describe the "
        "overall research responsibility. Do not enumerate all taxonomy/FoS "
        "nodes, and do not describe only one sub-skill. Keep it paper-specific "
        "and technical, but make it broader than a child task-node role.\n\n"
        "Return only valid JSON with fields: paper_id, node_id, requirement, "
        "role_description, key_capabilities, evidence_from_abstract. Use "
        '"virtual_root" as node_id. Fill evidence_from_abstract with 2-4 short '
        "phrases from the title/abstract.\n\n"
        f"Paper id: {task['paper_id']}\n"
        f"Title/abstract:\n{task.get('task_paper_text') or ''}\n\n"
        f"All weighted task skills:\n{task.get('all_task_skills') or ''}"
    )


def extract_json_object(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def normalize_llm_obj(obj: dict, task: dict) -> dict:
    role = str(obj.get("role_description") or obj.get("requirement") or "").strip()
    if not role:
        raise ValueError("LLM response is missing requirement/role_description")
    if not role.startswith("This task requires an expert who can lead and integrate"):
        role = "This task requires an expert who can lead and integrate " + role

    key_capabilities = obj.get("key_capabilities")
    if not isinstance(key_capabilities, list):
        key_capabilities = []
    evidence = obj.get("evidence_from_abstract")
    if not isinstance(evidence, list):
        evidence = []

    return {
        "paper_id": str(task["paper_id"]),
        "node_id": "virtual_root",
        "requirement": role,
        "role_description": role,
        "key_capabilities": [str(x) for x in key_capabilities],
        "evidence_from_abstract": [str(x) for x in evidence],
    }


def chat_role(client, args: argparse.Namespace, task: dict) -> str:
    prompt = build_prompt(task)
    last_error = None
    for attempt in range(1, args.retry + 1):
        try:
            kwargs = {
                "model": args.model,
                "temperature": args.temperature,
                "max_tokens": args.max_tokens,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You generate JSON virtual-root research expertise "
                            "profiles for research team formation. Use concise, "
                            "paper-specific technical language. Return only "
                            "valid JSON."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
            }
            if args.backend == "openai-compatible-chat":
                kwargs["response_format"] = {"type": "json_object"}
            response = client.chat.completions.create(**kwargs)
            text = response.choices[0].message.content or ""
            return normalize_llm_obj(extract_json_object(text), task)["role_description"]
        except Exception as exc:  # OpenAI-compatible servers vary.
            last_error = exc
            if attempt >= args.retry:
                raise
            time.sleep(args.retry_sleep)
    raise RuntimeError(last_error)


def main() -> None:
    args = parse_args()
    tasks = collect_tasks(Path(args.task_nodes_jsonl), args.limit)
    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    client = None
    if args.backend == "together":
        if not args.api_key:
            raise SystemExit("Missing Together API key. Set TOGETHER_API_KEY or pass --api-key.")
        from together import Together

        client = Together(api_key=args.api_key)
    elif args.backend == "openai-compatible-chat":
        if not (args.api_key and args.model):
            raise SystemExit("--api-key and --model are required for chat backend")
        from openai import OpenAI

        kwargs = {"api_key": args.api_key}
        if args.base_url:
            kwargs["base_url"] = args.base_url
        client = OpenAI(**kwargs)

    done = load_done(out_path) if args.resume else set()
    mode = "a" if args.resume else "w"

    def build_obj(task: dict) -> dict:
        role = chat_role(client, args, task) if client else template_role(task)
        return {
            "paper_id": task["paper_id"],
            "id": task["paper_id"],
            "node_id": "virtual_root",
            "root_role": role,
            "text": role,
            "generation_method": args.backend,
            "team_size": task.get("team_size"),
            "members": task.get("members") or [],
            "node_count": task.get("node_count"),
            "all_task_skills": task.get("all_task_skills") or "",
            "task_paper_text": task.get("task_paper_text") or "",
        }

    with out_path.open(mode, encoding="utf-8") as f:
        written = 0
        skipped = 0
        if args.workers <= 1 or not client:
            for idx, task in enumerate(tasks, start=1):
                if str(task["paper_id"]) in done:
                    skipped += 1
                    continue
                obj = build_obj(task)
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")
                f.flush()
                written += 1
                if written % 25 == 0:
                    print(f"generated_root_roles={written} skipped={skipped} idx={idx}/{len(tasks)}", flush=True)
        else:
            futures = {}
            submitted = 0
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                for idx, task in enumerate(tasks, start=1):
                    if str(task["paper_id"]) in done:
                        skipped += 1
                        continue
                    futures[executor.submit(build_obj, task)] = idx
                    submitted += 1
                    while len(futures) >= args.max_in_flight:
                        completed, _ = wait(futures, return_when=FIRST_COMPLETED)
                        for future in completed:
                            idx_done = futures.pop(future)
                            obj = future.result()
                            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
                            f.flush()
                            written += 1
                            if written % 25 == 0:
                                print(
                                    f"generated_root_roles={written} skipped={skipped} "
                                    f"idx={idx_done}/{len(tasks)} submitted={submitted}",
                                    flush=True,
                                )
                while futures:
                    completed, _ = wait(futures, return_when=FIRST_COMPLETED)
                    for future in completed:
                        idx_done = futures.pop(future)
                        obj = future.result()
                        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
                        f.flush()
                        written += 1
                        if written % 25 == 0:
                            print(
                                f"generated_root_roles={written} skipped={skipped} "
                                f"idx={idx_done}/{len(tasks)} submitted={submitted}",
                                flush=True,
                            )
    print(f"wrote={out_path} rows={len(tasks)}", flush=True)


if __name__ == "__main__":
    main()
