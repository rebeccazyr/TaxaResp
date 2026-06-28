"""Generate unique term lists for each semantic category.

The script reads the GPT-tagged task description CSV and writes four CSV files
(one per category: data, task, method, domain). Each output file contains the
unique terms for that category plus the IDs of every abstract that mentions the
term.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Set

CATEGORIES = ("data", "task", "method", "domain")
DEFAULT_INPUT = Path(__file__).with_name("task_description_terms.csv")
DEFAULT_OUTPUT_DIR = Path(__file__).parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create per-category CSV files listing unique terms and their IDs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="CSV file created by extract_task_description_terms.py",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where the per-category CSV files will be written",
    )
    return parser.parse_args()


def load_rows(path: Path) -> Iterable[Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Input CSV not found: {path}")

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if "terms" not in reader.fieldnames:
            raise ValueError("Input CSV must contain a 'terms' column.")
        for row in reader:
            yield row


def collect_terms(rows: Iterable[Dict[str, str]]) -> Dict[str, Dict[str, Set[str]]]:
    collected: Dict[str, Dict[str, Set[str]]] = {
        category: defaultdict(set) for category in CATEGORIES
    }

    for row in rows:
        task_id = (row.get("id") or row.get("paper_id") or "").strip()
        if not task_id:
            continue
        blob = row.get("terms") or ""
        if not blob:
            continue
        try:
            parsed = json.loads(blob)
        except json.JSONDecodeError:
            continue

        for category in CATEGORIES:
            terms: List[str] = parsed.get(category) or []
            for term in terms:
                cleaned = term.strip()
                if cleaned:
                    collected[category][cleaned].add(task_id)

    return collected


def write_outputs(collected: Dict[str, Dict[str, Set[str]]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for category, term_map in collected.items():
        terms = sorted(term_map)

        csv_path = output_dir / f"{category}_terms.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["term", "ids"])
            for term in terms:
                ids = sorted(term_map[term])
                writer.writerow([term, ";".join(ids)])

        txt_path = output_dir / f"{category}_terms.txt"
        with txt_path.open("w", encoding="utf-8") as handle:
            for term in terms:
                handle.write(f"{term}\n")


def main() -> None:
    args = parse_args()
    rows = load_rows(args.input)
    collected = collect_terms(rows)
    write_outputs(collected, args.output_dir)


if __name__ == "__main__":
    main()
