#!/usr/bin/env python3
"""Extract academic keywords from task descriptions via GPT-4o mini.

The script reads task descriptions (paper abstracts) from the DBLP-derived CSV
and asks an OpenAI GPT-4o mini model to identify domain-specific academic
terms grouped into semantic categories (data, task, method, application).
The resulting annotations are written to a CSV so they can be reused without
repeatedly calling the API.

Usage (after installing the ``openai`` package and exporting ``OPENAI_API_KEY``)
----------------------------------------------------------------------
$ python data_preprocess/extract_task_description_terms.py \
    --input data/dblp_new/cs_paper_with_author_published_2plus.csv \
    --output data_preprocess/task_description_terms.csv

Use ``--limit`` to process a subset while testing:
$ python data_preprocess/extract_task_description_terms.py --limit 5
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
import sys
import threading
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple

try:
    from openai import OpenAI  # type: ignore
except ImportError:  # pragma: no cover - dependency check
    OpenAI = None  # type: ignore

_THREAD_LOCAL = threading.local()

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPO_ROOT / "data" / "dblp_new" / "cs_paper_with_author_published_2plus.csv"
DEFAULT_OUTPUT = Path(__file__).resolve().with_name("task_description_terms.csv")

CATEGORY_DESCRIPTIONS = {
    "data": "datasets, input data types, or data sources mentioned in the abstract",
    "task": "computational problems or machine learning tasks addressed by the paper",
    "method": "algorithms, models, techniques, or theoretical frameworks used or proposed",
    "domain": "application domains or fields (e.g., healthcare, NLP, telecommunications)",
}

CATEGORY_KEYS: Tuple[str, ...] = tuple(CATEGORY_DESCRIPTIONS.keys())
CATEGORY_LIMITS = {key: 3 for key in CATEGORY_KEYS}

SYSTEM_PROMPT = (
    "You are an expert research assistant for computer science literature analysis.\n\n"
    "Your task is to extract precise academic terms or short noun phrases from a paper abstract "
    "and categorize them into four semantic categories.\n\n"
    "Extraction rules:\n"
    "- Each phrase must be a contiguous span copied exactly from the abstract text.\n"
    "- Do NOT paraphrase or generate new phrases.\n"
    "- Extract concise academic terms (1–6 words).\n"
    "- Prefer canonical research terminology commonly used in computer science literature.\n"
    "- Avoid vague phrases such as \"technical challenges\", \"approaches\", \"future work\".\n"
    "- Avoid generic method words such as \"algorithm\", \"method\", \"approach\", or \"framework\" "
    "unless they are part of a specific academic term.\n"
    "- If a category is not present in the abstract, return [].\n\n"
    "Categories:\n"
    "data: datasets, input data types, or data sources mentioned in the abstract\n"
    "task: computational problems or machine learning tasks addressed by the paper\n"
    "method: specific named algorithms, models, or techniques used or proposed\n"
    "domain: established research fields or application domains\n\n"
    "Return ONLY valid JSON with exactly these keys:\n"
    "[\"data\", \"task\", \"method\", \"domain\"]\n\n"
    "Each value must contain at most 3 phrases.\n"
    "If more than 3 candidates exist, select the three most central and specific terms."
)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract categorized academic terms from task descriptions using GPT-4o mini.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="CSV file with a task_description column.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Destination CSV containing the extracted terms.",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="OpenAI model identifier to query.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N records (useful for smoke tests).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature for the chat completion call.",
    )
    parser.add_argument(
        "--retry",
        type=int,
        default=3,
        help="Maximum number of retries per request when the API fails.",
    )
    parser.add_argument(
        "--retry-wait",
        type=float,
        default=2.0,
        help="Base waiting time (in seconds) between retries; scaled exponentially.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug-level logging output.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=10,
        help="Number of concurrent OpenAI API requests to issue.",
    )
    return parser.parse_args()


def normalize_description(text: str) -> str:
    """Collapse whitespace to stabilize prompts."""

    cleaned = " ".join(text.split())
    return cleaned.strip()


def iter_rows(path: Path, limit: int | None) -> Iterable[Tuple[str, str]]:
    """Yield (id, task_description) pairs from the CSV."""

    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if "task_description" not in reader.fieldnames:
            raise ValueError("Input CSV must contain a 'task_description' column.")

        for index, row in enumerate(reader):
            if limit is not None and index >= limit:
                break
            task_id = (row.get("id") or row.get("paper_id") or f"row_{index}").strip()
            description = normalize_description(row.get("task_description", ""))
            if not description:
                logging.debug("Skipping %s due to empty task description.", task_id)
                continue
            yield task_id, description


def build_prompt(description: str) -> str:
    category_text = "\n".join(
        f"- {name}: {details}" for name, details in CATEGORY_DESCRIPTIONS.items()
    )
    categories = ", ".join(f'"{key}"' for key in CATEGORY_KEYS)
    return (
        "You are given the abstract of a computer science paper. Identify key "
        "technical concepts for the following categories:\n"
        f"{category_text}\n"
        "Return valid JSON with keys in this exact order: "
        f"[{categories}]. Each value must be an array.\n"
        f"Abstract: {description}"
    )


def call_llm(client: OpenAI, *, model: str, prompt: str, temperature: float,
             retry: int, retry_wait: float) -> str:
    """Invoke the chat completion API with retries."""

    last_error: Exception | None = None
    for attempt in range(1, retry + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
            )
            content = response.choices[0].message.content or ""
            return content.strip()
        except Exception as exc:  # pragma: no cover - network errors
            last_error = exc
            sleep_for = retry_wait * (2 ** (attempt - 1))
            logging.warning(
                "LLM call failed on attempt %s/%s: %s. Retrying in %.1fs",
                attempt,
                retry,
                exc,
                sleep_for,
            )
            time.sleep(sleep_for)
    if last_error:
        raise last_error
    raise RuntimeError("LLM call failed without raising an exception.")


def extract_terms_from_response(raw_content: str) -> Tuple[Dict[str, List[str]], str]:
    """Parse the JSON output; fall back to heuristic extraction when needed."""

    content = raw_content.strip()
    payload_text = content
    json_block = _find_json_block(content)
    if json_block is not None:
        payload_text = json_block

    normalized_payload = _empty_category_payload()
    try:
        payload = json.loads(payload_text)
    except json.JSONDecodeError:
        fallback = _fallback_category_payload(content)
        return fallback, json.dumps(fallback)

    if isinstance(payload, dict):
        for key in CATEGORY_KEYS:
            normalized_payload[key] = _clean_terms(key, payload.get(key, []))
    else:
        fallback = _fallback_category_payload(content)
        return fallback, json.dumps(fallback)

    extra_fields = {
        key: value for key, value in payload.items() if key not in CATEGORY_KEYS
    }
    serialized = json.dumps({**normalized_payload, **extra_fields})
    return normalized_payload, serialized


def _find_json_block(text: str) -> str | None:
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        return match.group(0)
    return None


def _fallback_terms_from_text(text: str) -> List[str]:
    # Extract candidate tokens from bullet lists or comma-separated output.
    candidates = re.split(r"[\n,;\-]+", text)
    cleaned = []
    for candidate in candidates:
        term = candidate.strip().strip("-•* ")
        if term and len(term.split()) <= 6:
            cleaned.append(term)
    return cleaned[:12]


def _empty_category_payload() -> Dict[str, List[str]]:
    return {key: [] for key in CATEGORY_KEYS}


def _fallback_category_payload(text: str) -> Dict[str, List[str]]:
    payload = _empty_category_payload()
    fallback_terms = _fallback_terms_from_text(text)
    for category in CATEGORY_KEYS:
        payload[category] = fallback_terms[: CATEGORY_LIMITS[category]]
    return payload


def _clean_terms(category: str, value: object) -> List[str]:
    limit = CATEGORY_LIMITS.get(category, 4)
    if isinstance(value, str):
        candidates = _fallback_terms_from_text(value)
    elif isinstance(value, Sequence):
        candidates = [str(term) for term in value]
    else:
        return []
    return _normalize_terms_from_list(candidates, limit)


def _normalize_terms_from_list(terms: Sequence[str], limit: int) -> List[str]:
    preferred: List[str] = []
    fallback: List[str] = []
    for term in terms:
        normalized = term.strip()
        if not normalized:
            continue
        words = normalized.split()
        if len(words) > 6:
            normalized = " ".join(words[:6])
            words = normalized.split()
        target_list = preferred if len(words) >= 2 else fallback
        if normalized not in preferred and normalized not in fallback:
            target_list.append(normalized)
        if len(preferred) >= limit:
            break
    combined = preferred + fallback
    return combined[:limit]


def ensure_openai_client() -> OpenAI:
    if OpenAI is None:  # pragma: no cover - dependency check
        raise SystemExit(
            "The 'openai' package is required. Install it via 'pip install openai'."
        )
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("Environment variable OPENAI_API_KEY is not set.")
    return OpenAI()


def _get_thread_client() -> OpenAI:
    client = getattr(_THREAD_LOCAL, "client", None)
    if client is None:
        client = ensure_openai_client()
        _THREAD_LOCAL.client = client
    return client


def load_existing_ids(path: Path) -> Set[str]:
    if not path.exists():
        return set()

    processed: Set[str] = set()
    try:
        with path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames or "id" not in reader.fieldnames:
                return set()
            for row in reader:
                task_id = (row.get("id") or "").strip()
                if task_id:
                    processed.add(task_id)
    except Exception as exc:  # pragma: no cover - file errors
        logging.warning("Could not read existing output for resume: %s", exc)
        return set()
    return processed


def _output_schema_matches(path: Path, desired_fields: Sequence[str]) -> bool:
    if not path.exists():
        return False
    try:
        with path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            header = next(reader, None)
    except Exception:  # pragma: no cover - file errors
        return False
    return header == list(desired_fields)


def _rewrite_existing_output(path: Path, desired_fields: Sequence[str]) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    try:
        with path.open("r", newline="", encoding="utf-8") as src, tmp_path.open(
            "w", newline="", encoding="utf-8"
        ) as dst:
            reader = csv.DictReader(src)
            writer = csv.DictWriter(dst, fieldnames=list(desired_fields))
            writer.writeheader()
            for row in reader:
                writer.writerow({field: row.get(field, "") for field in desired_fields})
    except Exception as exc:  # pragma: no cover - file errors
        try:
            tmp_path.unlink()
        except FileNotFoundError:
            pass
        raise
    else:
        tmp_path.replace(path)


def _process_single_task(
    task_id: str,
    description: str,
    model: str,
    temperature: float,
    retry: int,
    retry_wait: float,
) -> Tuple[str, Dict[str, List[str]], str, str]:
    client = _get_thread_client()
    logging.info("Processing %s", task_id)
    prompt = build_prompt(description)
    try:
        raw_response = call_llm(
            client,
            model=model,
            prompt=prompt,
            temperature=temperature,
            retry=retry,
            retry_wait=retry_wait,
        )
    except Exception as exc:  # pragma: no cover - network errors
        raise RuntimeError(f"Failed to process {task_id}") from exc
    terms_by_category, parsed_payload = extract_terms_from_response(raw_response)
    return task_id, terms_by_category, parsed_payload, description


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="[%(levelname)s] %(message)s",
    )
    ensure_openai_client()  # fail fast if dependencies/API key missing

    concurrency = max(1, args.concurrency)
    max_pending = max(concurrency * 4, concurrency)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["id", "terms", "task_description"]
    rows_written = 0
    processed_ids = load_existing_ids(args.output)
    schema_matches = _output_schema_matches(args.output, fieldnames)
    if args.output.exists() and processed_ids and not schema_matches:
        logging.info(
            "Updating existing output schema to match new format: %s",
            args.output,
        )
        _rewrite_existing_output(args.output, fieldnames)
        processed_ids = load_existing_ids(args.output)
        schema_matches = True
    append_mode = args.output.exists() and bool(processed_ids) and schema_matches
    existing_rows = len(processed_ids)
    if append_mode:
        logging.info(
            "Resuming run: %s rows already present in %s",
            existing_rows,
            args.output,
        )

    file_mode = "a" if append_mode else "w"

    with args.output.open(file_mode, newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not append_mode:
            writer.writeheader()

        def drain_completed(pending_futures: Set[Future]) -> None:
            nonlocal rows_written
            if not pending_futures:
                return
            done, remaining = wait(pending_futures, return_when=FIRST_COMPLETED)
            pending_futures.clear()
            pending_futures.update(remaining)
            for future in done:
                try:
                    task_id, terms_by_category, _, description = future.result()
                except Exception as exc:
                    logging.error("A task failed: %s", exc)
                    continue
                processed_ids.add(task_id)
                writer.writerow(
                    {
                        "id": task_id,
                        "terms": json.dumps(terms_by_category),
                        "task_description": description,
                    }
                )
                handle.flush()
                rows_written += 1

        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            pending: Set[Future] = set()
            for task_id, description in iter_rows(args.input, args.limit):
                if task_id in processed_ids:
                    logging.debug("Skipping %s (already processed).", task_id)
                    continue
                future = executor.submit(
                    _process_single_task,
                    task_id,
                    description,
                    args.model,
                    args.temperature,
                    args.retry,
                    args.retry_wait,
                )
                pending.add(future)
                if len(pending) >= max_pending:
                    drain_completed(pending)

            while pending:
                drain_completed(pending)

    if rows_written == 0:
        if existing_rows:
            logging.info(
                "No new tasks were processed. Output remains at %s rows in %s",
                existing_rows,
                args.output,
            )
        else:
            logging.warning(
                "No rows were processed; output file only contains the header: %s",
                args.output,
            )
    else:
        total_rows = rows_written + existing_rows
        logging.info(
            "Finished processing %s new tasks (total %s). Results saved to %s",
            rows_written,
            total_rows,
            args.output,
        )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:  # pragma: no cover - user cancellation
        sys.exit("Interrupted by user.")
