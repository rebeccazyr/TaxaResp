#!/usr/bin/env python3
"""Generate definition-like summaries for extracted academic terms.

The script looks at the GPT-tagged ``task_description_terms.csv`` output and,
for each semantic category (data, task, method, domain), produces a CSV file
containing:
    - the term as it appears in the ``*_terms.txt`` exports,
    - the inferred term kind (reusable academic concept or paper-specific label),
    - an optional description (paper-specific entries reuse their abstract
      snippets), and
    - the IDs of the abstracts that reference the term.

This makes it easier to review noisy terms and keep only those that are clearly
usable as academic phrases with clear definitions inferred from context.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
import time
from collections import defaultdict
from pathlib import Path
from enum import Enum
from typing import DefaultDict, Dict, Iterable, List, NamedTuple, Tuple

import threading

try:
    from openai import OpenAI  # type: ignore
except ImportError:  # pragma: no cover - dependency check
    OpenAI = None  # type: ignore

CATEGORIES: Tuple[str, ...] = ("data", "task", "method", "domain")
_THREAD_LOCAL = threading.local()
DEFAULT_TERMS_DIR = Path(__file__).parent
DEFAULT_OUTPUT_DIR = DEFAULT_TERMS_DIR / "term_descriptions"
DEFAULT_INPUT = DEFAULT_TERMS_DIR / "task_description_terms.csv"

DEFAULT_MODEL = "gpt-4o-mini"
SYSTEM_PROMPT = (
    "You are an expert editor creating concise dictionary-style definitions for "
    "computer science research terms. Use the provided abstract snippets to infer "
    "what the term refers to. Respond with one clear definition (1-2 sentences) "
    "that starts with the term itself (e.g., 'Graph neural networks are ...'). "
    "Avoid mentioning the paper or authors; focus on defining the term.")
CLASSIFIER_SYSTEM_PROMPT = (
    "You are an expert terminology curator deciding whether terms mentioned in "
    "computer science research abstracts refer to reusable academic concepts or "
    "paper-specific names/noise. Judge carefully using only the provided contexts."
)


class TermKind(str, Enum):
    REUSABLE = "reusable"
    PAPER_SPECIFIC = "paper_specific"
    NOISE = "noise"


class TermClassification(NamedTuple):
    kind: TermKind
    reason: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate definition-style summaries for extracted terms.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="CSV file created by extract_task_description_terms.py",
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
        help="Directory where per-category description CSVs will be written.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="OpenAI chat completion model to use for definition generation.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.3,
        help="Sampling temperature for the chat completion call.",
    )
    parser.add_argument(
        "--max-contexts",
        type=int,
        default=0,
        help="Maximum number of abstracts to include per term when prompting the LLM (0 = no limit).",
    )
    parser.add_argument(
        "--max-abstract-chars",
        type=int,
        default=0,
        help="Truncate each abstract to this many characters before sending to the LLM (0 = no limit).",
    )
    parser.add_argument(
        "--retry",
        type=int,
        default=3,
        help="Maximum number of retries for each OpenAI request.",
    )
    parser.add_argument(
        "--retry-wait",
        type=float,
        default=3.0,
        help="Base waiting time (seconds) between retries; doubles after each failure.",
    )
    return parser.parse_args()


def clean_term(raw: str) -> str:
    cleaned = raw.strip().strip("'\"`").strip()
    return re.sub(r"\s+", " ", cleaned)


def read_terms(terms_dir: Path) -> Dict[str, List[str]]:
    term_lists: Dict[str, List[str]] = {}
    for category in CATEGORIES:
        path = terms_dir / f"{category}_terms.txt"
        if not path.exists():
            raise FileNotFoundError(f"Missing term list: {path}")
        with path.open(encoding="utf-8") as handle:
            term_lists[category] = [clean_term(line) for line in handle if line.strip()]
    return term_lists


def load_contexts(csv_path: Path) -> Dict[str, DefaultDict[str, List[Tuple[str, str]]]]:
    contexts: Dict[str, DefaultDict[str, List[Tuple[str, str]]]] = {
        category: defaultdict(list) for category in CATEGORIES
    }
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            task_id = (row.get("id") or row.get("paper_id") or "").strip()
            description = (row.get("task_description") or "").strip()
            if not task_id or not description:
                continue
            blob = row.get("terms") or ""
            try:
                payload = json.loads(blob)
            except json.JSONDecodeError:
                continue
            if not isinstance(payload, dict):
                continue
            for category in CATEGORIES:
                for raw_term in payload.get(category, []) or []:
                    term = clean_term(str(raw_term))
                    if term:
                        contexts[category][term].append((task_id, description))
    return contexts
def ensure_openai_client() -> OpenAI:
    if OpenAI is None:  # pragma: no cover - dependency check
        raise SystemExit("The 'openai' package is required. Install it via 'pip install openai'.")
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("Environment variable OPENAI_API_KEY is not set.")
    return OpenAI()


def _get_client() -> OpenAI:
    client = getattr(_THREAD_LOCAL, "client", None)
    if client is None:
        client = ensure_openai_client()
        _THREAD_LOCAL.client = client
    return client


def _select_contexts(
    contexts: List[Tuple[str, str]],
    max_contexts: int,
    max_chars: int,
) -> List[Tuple[str, str]]:
    selected: List[Tuple[str, str]] = []
    for task_id, abstract in contexts:
        if max_contexts > 0 and len(selected) >= max_contexts:
            break
        snippet = re.sub(r"\s+", " ", abstract).strip()
        if max_chars > 0:
            snippet = snippet[:max_chars]
        selected.append((task_id, snippet))
    return selected


def build_prompt(
    term: str,
    category: str,
    contexts: List[Tuple[str, str]],
) -> str:
    context_text = "\n\n".join(
        f"Context {index + 1} (ID={task_id}):\n{abstract}"
        for index, (task_id, abstract) in enumerate(contexts)
    )
    return (
        f"Term: {term}\n"
        f"Category: {category}\n\n"
        "Use the following abstract excerpts to write a concise definition of the term.\n"
        "Context snippets:\n"
        f"{context_text}\n\n"
        "Definition:"
    )


def build_classification_prompt(
    term: str,
    category: str,
    contexts: List[Tuple[str, str]],
) -> str:
    context_text = "\n\n".join(
        f"Context {index + 1} (ID={task_id}):\n{abstract}"
        for index, (task_id, abstract) in enumerate(contexts)
    )
    return (
        f"Term: {term}\n"
        f"Category: {category}\n\n"
        "Use the following abstract excerpts to decide whether the term is a"
        " reusable academic concept rather than a paper-specific label or noisy"
        " fragment.\n"
        "Context snippets:\n"
        f"{context_text}\n\n"
        "Respond using exactly one of the following formats:\n"
        "REUSABLE - <short reason>\n"
        "PAPER_SPECIFIC - <short reason>\n"
        "NOISE - <short reason>"
    )


def call_llm(
    client: OpenAI,
    *,
    model: str,
    prompt: str,
    temperature: float,
    retry: int,
    retry_wait: float,
    system_prompt: str = SYSTEM_PROMPT,
) -> str:
    last_error: Exception | None = None
    for attempt in range(1, retry + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": system_prompt},
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


def build_definition(
    term: str,
    category: str,
    contexts: List[Tuple[str, str]],
    *,
    model: str,
    temperature: float,
    retry: int,
    retry_wait: float,
    max_contexts: int,
    max_abstract_chars: int,
) -> str:
    if not contexts:
        return ""
    client = _get_client()
    selected_contexts = _select_contexts(contexts, max_contexts, max_abstract_chars)
    prompt = build_prompt(term, category, selected_contexts)
    try:
        definition = call_llm(
            client,
            model=model,
            prompt=prompt,
            temperature=temperature,
            retry=retry,
            retry_wait=retry_wait,
        )
    except Exception as exc:
        logging.error("Failed to build definition for %s (%s): %s", term, category, exc)
        return ""
    definition = re.sub(r"\s+", " ", definition).strip()
    return definition


def build_paper_specific_description(
    term: str,
    contexts: List[Tuple[str, str]],
    *,
    max_abstract_chars: int,
) -> str:
    if not contexts:
        return ""
    task_id, abstract = contexts[0]
    snippet = re.sub(r"\s+", " ", abstract).strip()
    if max_abstract_chars > 0:
        snippet = snippet[:max_abstract_chars]
    prefix = f"{term} (source {task_id})" if task_id else term
    return f"{prefix} is described in the cited abstract as: {snippet}"


def _parse_classifier_response(response: str) -> TermClassification | None:
    normalized = response.strip()
    lowered = normalized.lower()
    options = (
        ("reusable", TermKind.REUSABLE),
        ("paper_specific", TermKind.PAPER_SPECIFIC),
        ("paper-specific", TermKind.PAPER_SPECIFIC),
        ("paper specific", TermKind.PAPER_SPECIFIC),
        ("noise", TermKind.NOISE),
    )
    for prefix, kind in options:
        if lowered.startswith(prefix):
            reason = ""
            if "-" in normalized:
                reason = normalized.split("-", 1)[1].strip()
            return TermClassification(kind, reason)
    return None


def classify_term(
    term: str,
    category: str,
    contexts: List[Tuple[str, str]],
    *,
    model: str,
    retry: int,
    retry_wait: float,
    max_contexts: int,
    max_abstract_chars: int,
) -> TermClassification:
    if not contexts:
        return TermClassification(TermKind.NOISE, "missing supporting contexts")
    client = _get_client()
    selected_contexts = _select_contexts(contexts, max_contexts, max_abstract_chars)
    prompt = build_classification_prompt(term, category, selected_contexts)
    try:
        verdict = call_llm(
            client,
            model=model,
            prompt=prompt,
            temperature=0.0,
            retry=retry,
            retry_wait=retry_wait,
            system_prompt=CLASSIFIER_SYSTEM_PROMPT,
        )
    except Exception as exc:
        logging.error(
            "Failed to classify term %s (%s): %s",
            term,
            category,
            exc,
        )
        return TermClassification(TermKind.NOISE, f"classifier error: {exc}")
    parsed = _parse_classifier_response(verdict)
    if parsed is None:
        logging.warning(
            "Ambiguous classifier response for term %s (%s): %s",
            term,
            category,
            verdict,
        )
        return TermClassification(TermKind.NOISE, "ambiguous classifier response")
    return parsed


def write_output(
    category: str,
    terms: Iterable[str],
    contexts: Dict[str, DefaultDict[str, List[Tuple[str, str]]]],
    output_dir: Path,
    *,
    model: str,
    temperature: float,
    retry: int,
    retry_wait: float,
    max_contexts: int,
    max_abstract_chars: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{category}_term_descriptions.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["term", "term_kind", "description", "source_ids"])
        for term in terms:
            if not term:
                continue
            context_list = contexts.get(category, {}).get(term, [])
            classification = classify_term(
                term,
                category,
                context_list,
                model=model,
                retry=retry,
                retry_wait=retry_wait,
                max_contexts=max_contexts,
                max_abstract_chars=max_abstract_chars,
            )
            if classification.kind is TermKind.NOISE:
                continue
            if classification.kind is TermKind.REUSABLE:
                description = build_definition(
                    term,
                    category,
                    context_list,
                    model=model,
                    temperature=temperature,
                    retry=retry,
                    retry_wait=retry_wait,
                    max_contexts=max_contexts,
                    max_abstract_chars=max_abstract_chars,
                )
            else:
                description = build_paper_specific_description(
                    term,
                    context_list,
                    max_abstract_chars=max_abstract_chars,
                )
            ids = sorted({task_id for task_id, _ in context_list if task_id})
            writer.writerow(
                [
                    term,
                    classification.kind.value,
                    description,
                    ";".join(ids),
                ]
            )


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    ensure_openai_client()
    term_lists = read_terms(args.terms_dir)
    contexts = load_contexts(args.input)
    for category, terms in term_lists.items():
        write_output(
            category,
            terms,
            contexts,
            args.output_dir,
            model=args.model,
            temperature=args.temperature,
            retry=args.retry,
            retry_wait=args.retry_wait,
            max_contexts=args.max_contexts,
            max_abstract_chars=args.max_abstract_chars,
        )


if __name__ == "__main__":
    main()
