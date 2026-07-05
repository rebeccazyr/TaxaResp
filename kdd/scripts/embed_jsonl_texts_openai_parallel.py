#!/usr/bin/env python3
"""Embed JSONL text records with OpenAI API using bounded concurrency.

Outputs are compatible with embedding_pipeline_utils.load_embedding_table:
- ids TSV has an `id` column in the same order as the .npy rows.
- embeddings .npy is a float32 matrix.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
from tqdm import tqdm


def load_dotenv() -> None:
    for directory in [Path.cwd(), *Path.cwd().parents]:
        paths = [directory / name for name in (".env", ".env.openai", ".env.together")]
        existing = [path for path in paths if path.exists()]
        if not existing:
            continue
        for path in existing:
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, value = line.split("=", 1)
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")
                    if key and key not in os.environ:
                        os.environ[key] = value
        return


def parse_args() -> argparse.Namespace:
    load_dotenv()
    p = argparse.ArgumentParser(description="Parallel OpenAI JSONL embedding")
    p.add_argument("--input-jsonl", required=True)
    p.add_argument("--ids-out", required=True)
    p.add_argument("--embeddings-out", required=True)
    p.add_argument(
        "--metrics-out",
        default="",
        help="Optional JSON file with throughput and concurrency metrics.",
    )
    p.add_argument("--model", default="text-embedding-3-small")
    p.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", ""))
    p.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", ""))
    p.add_argument("--id-field", default="id")
    p.add_argument("--text-field", default="text")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--max-in-flight", type=int, default=48)
    p.add_argument("--target-tpm", type=int, default=4_500_000)
    p.add_argument("--target-rpm", type=int, default=9_000)
    p.add_argument("--retry", type=int, default=8)
    p.add_argument("--retry-sleep", type=float, default=2.0)
    p.add_argument("--normalize", action="store_true")
    p.add_argument("--dimensions", type=int, default=0)
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def iter_records(path: Path, id_field: str, text_field: str) -> Iterable[Tuple[str, str, int]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            rec_id = str(obj.get(id_field) or obj.get("paper_id") or "")
            text = str(obj.get(text_field) or "")
            text = " ".join(text.split())
            if rec_id and text:
                yield rec_id, text, len(text)


def count_records(path: Path, id_field: str, text_field: str) -> int:
    return sum(1 for _ in iter_records(path, id_field, text_field))


def existing_embedding_mask(arr: np.ndarray, chunk_size: int = 4096) -> np.ndarray:
    mask = np.zeros(arr.shape[0], dtype=bool)
    for start in range(0, arr.shape[0], chunk_size):
        end = min(start + chunk_size, arr.shape[0])
        mask[start:end] = np.any(arr[start:end] != 0.0, axis=1)
    return mask


def estimate_tokens(texts: List[str]) -> int:
    # Conservative approximation for English title/abstract text.
    return max(1, math.ceil(sum(len(t) for t in texts) / 4))


class RateLimiter:
    def __init__(self, target_tpm: int, target_rpm: int) -> None:
        self.target_tpm = target_tpm
        self.target_rpm = target_rpm
        self.lock = threading.Lock()
        self.window_start = time.monotonic()
        self.tokens = 0
        self.requests = 0

    def acquire(self, tokens: int) -> None:
        if self.target_tpm <= 0 and self.target_rpm <= 0:
            return
        while True:
            with self.lock:
                now = time.monotonic()
                elapsed = now - self.window_start
                if elapsed >= 60.0:
                    self.window_start = now
                    self.tokens = 0
                    self.requests = 0
                    elapsed = 0.0
                token_ok = self.target_tpm <= 0 or self.tokens + tokens <= self.target_tpm
                request_ok = self.target_rpm <= 0 or self.requests + 1 <= self.target_rpm
                if token_ok and request_ok:
                    self.tokens += tokens
                    self.requests += 1
                    return
                sleep_for = max(0.2, 60.0 - elapsed)
            time.sleep(sleep_for)


class ConcurrencyMetrics:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.active_requests = 0
        self.max_active_requests = 0
        self.max_pending_futures = 0
        self.submitted_batches = 0
        self.completed_batches = 0
        self.submitted_records = 0
        self.completed_records = 0

    def on_submit(self, pending_futures: int, batch_records: int) -> None:
        with self.lock:
            self.submitted_batches += 1
            self.submitted_records += batch_records
            self.max_pending_futures = max(self.max_pending_futures, pending_futures)

    def on_request_start(self) -> None:
        with self.lock:
            self.active_requests += 1
            self.max_active_requests = max(self.max_active_requests, self.active_requests)

    def on_request_end(self) -> None:
        with self.lock:
            self.active_requests -= 1

    def on_complete(self, batch_records: int) -> None:
        with self.lock:
            self.completed_batches += 1
            self.completed_records += batch_records

    def as_dict(self) -> dict:
        with self.lock:
            return {
                "max_active_requests_observed": self.max_active_requests,
                "max_pending_futures_observed": self.max_pending_futures,
                "submitted_batches": self.submitted_batches,
                "completed_batches": self.completed_batches,
                "submitted_records": self.submitted_records,
                "completed_records": self.completed_records,
            }


def make_client(args: argparse.Namespace):
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise SystemExit("Missing package `openai`. Run: pip install openai") from exc
    if not args.api_key:
        raise SystemExit("Missing OPENAI_API_KEY or --api-key")
    kwargs = {"api_key": args.api_key}
    if args.base_url:
        kwargs["base_url"] = args.base_url
    return OpenAI(**kwargs)


def embed_batch(
    client,
    args: argparse.Namespace,
    limiter: RateLimiter,
    metrics: ConcurrencyMetrics,
    texts: List[str],
) -> np.ndarray:
    token_estimate = estimate_tokens(texts)
    last_error = None
    for attempt in range(1, args.retry + 1):
        try:
            limiter.acquire(token_estimate)
            kwargs = {"model": args.model, "input": texts}
            if args.dimensions > 0:
                kwargs["dimensions"] = args.dimensions
            metrics.on_request_start()
            try:
                response = client.embeddings.create(**kwargs)
            finally:
                metrics.on_request_end()
            data = sorted(response.data, key=lambda item: item.index)
            arr = np.array([item.embedding for item in data], dtype=np.float32)
            if args.normalize:
                norms = np.linalg.norm(arr, axis=1, keepdims=True)
                arr = arr / np.maximum(norms, 1e-12)
            return arr
        except Exception as exc:
            last_error = exc
            if attempt >= args.retry:
                raise
            time.sleep(args.retry_sleep * attempt)
    raise last_error  # type: ignore[misc]


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_jsonl)
    ids_path = Path(args.ids_out)
    npy_path = Path(args.embeddings_out)
    ids_path.parent.mkdir(parents=True, exist_ok=True)
    npy_path.parent.mkdir(parents=True, exist_ok=True)

    total = count_records(input_path, args.id_field, args.text_field)
    if total == 0:
        raise SystemExit("No text records found")

    if not args.resume or not ids_path.exists():
        ids_tmp = ids_path
        with ids_tmp.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["id", "text_chars", "source_jsonl"],
                delimiter="\t",
            )
            writer.writeheader()
            for rec_id, _, text_chars in iter_records(input_path, args.id_field, args.text_field):
                writer.writerow(
                    {
                        "id": rec_id,
                        "text_chars": text_chars,
                        "source_jsonl": str(input_path),
                    }
                )

    client = make_client(args)
    limiter = RateLimiter(args.target_tpm, args.target_rpm)
    metrics = ConcurrencyMetrics()
    start_time = time.monotonic()
    arr = None
    done_mask = None
    done = 0
    if args.resume and npy_path.exists():
        arr = np.load(npy_path, mmap_mode="r+")
        if arr.shape[0] != total:
            raise SystemExit(f"resume npy row count mismatch: npy={arr.shape[0]} input={total}")
        done_mask = existing_embedding_mask(arr)
        done = int(done_mask.sum())
        print(f"resume_done={done:,}")

    next_row = 0
    futures = {}
    progress = tqdm(total=total, initial=done, desc="embedding")

    def submit_batch(executor, indices: List[int], ids: List[str], texts: List[str]) -> None:
        future = executor.submit(embed_batch, client, args, limiter, metrics, texts)
        futures[future] = list(indices)
        metrics.on_submit(len(futures), len(ids))

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        batch_indices: List[int] = []
        batch_ids: List[str] = []
        batch_texts: List[str] = []
        for rec_id, text, _ in iter_records(input_path, args.id_field, args.text_field):
            if done_mask is not None and done_mask[next_row]:
                next_row += 1
                continue
            batch_indices.append(next_row)
            batch_ids.append(rec_id)
            batch_texts.append(text)
            next_row += 1
            if len(batch_texts) < args.batch_size:
                continue
            submit_batch(executor, batch_indices, batch_ids, batch_texts)
            batch_indices, batch_ids, batch_texts = [], [], []
            while len(futures) >= args.max_in_flight:
                completed, _ = wait(futures, return_when=FIRST_COMPLETED)
                for future in completed:
                    indices = futures.pop(future)
                    emb = future.result()
                    if arr is None:
                        arr = np.lib.format.open_memmap(
                            npy_path,
                            mode="w+",
                            dtype=np.float32,
                            shape=(total, emb.shape[1]),
                        )
                    arr[indices] = emb
                    progress.update(len(indices))
                    metrics.on_complete(len(indices))
        if batch_texts:
            submit_batch(executor, batch_indices, batch_ids, batch_texts)

        while futures:
            completed, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in completed:
                indices = futures.pop(future)
                emb = future.result()
                if arr is None:
                    arr = np.lib.format.open_memmap(
                        npy_path,
                        mode="w+",
                        dtype=np.float32,
                        shape=(total, emb.shape[1]),
                    )
                arr[indices] = emb
                progress.update(len(indices))
                metrics.on_complete(len(indices))

    progress.close()
    if arr is not None:
        arr.flush()
        dim = arr.shape[1]
    else:
        dim = 0
    print(f"records={total}")
    print(f"dim={dim}")
    print(f"ids_out={ids_path}")
    print(f"embeddings_out={npy_path}")
    elapsed = time.monotonic() - start_time
    metrics_payload = {
        "input_jsonl": str(input_path),
        "ids_out": str(ids_path),
        "embeddings_out": str(npy_path),
        "model": args.model,
        "dimensions_arg": args.dimensions,
        "normalize": bool(args.normalize),
        "records": total,
        "dim": dim,
        "batch_size": args.batch_size,
        "workers_config": args.workers,
        "max_in_flight_config": args.max_in_flight,
        "target_tpm": args.target_tpm,
        "target_rpm": args.target_rpm,
        "elapsed_seconds": elapsed,
        "records_per_second": total / elapsed if elapsed > 0 else 0.0,
        **metrics.as_dict(),
    }
    if args.metrics_out:
        metrics_path = Path(args.metrics_out)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(
            json.dumps(metrics_payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"metrics_out={metrics_path}")
    print(f"max_active_requests_observed={metrics_payload['max_active_requests_observed']}")
    print(f"max_pending_futures_observed={metrics_payload['max_pending_futures_observed']}")


if __name__ == "__main__":
    main()
