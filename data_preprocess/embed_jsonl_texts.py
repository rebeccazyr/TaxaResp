#!/usr/bin/env python3
"""Embed JSONL text records into .npy + ids.tsv tables.

Recommended SPECTER2 usage:
- paper_texts.jsonl: adapter=proximity, title-field=title, abstract-field=abstract
- task_node_requirements.jsonl: adapter=adhoc_query, text-field=requirement
- node_texts.jsonl: adapter=adhoc_query, text-field=text
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List, Tuple

import numpy as np
from tqdm import tqdm

from embedding_pipeline_utils import read_jsonl, save_embedding_table


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Embed JSONL texts")
    p.add_argument("--input-jsonl", required=True)
    p.add_argument("--ids-out", required=True)
    p.add_argument("--embeddings-out", required=True)
    p.add_argument("--backend", choices=("specter2", "sentence-transformers"), default="specter2")
    p.add_argument("--model", default="allenai/specter2_base")
    p.add_argument(
        "--adapter",
        default="proximity",
        choices=("proximity", "adhoc_query", "classification", "regression"),
        help="SPECTER2 adapter. proximity is allenai/specter2.",
    )
    p.add_argument("--sentence-transformer-model", default="allenai-specter")
    p.add_argument("--id-field", default="id")
    p.add_argument(
        "--composite-id-fields",
        default="",
        help="Comma-separated fields used to build ids, e.g. paper_id,node_id",
    )
    p.add_argument("--text-field", default="text")
    p.add_argument("--title-field", default="")
    p.add_argument("--abstract-field", default="")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, mps")
    p.add_argument("--normalize", action="store_true")
    return p.parse_args()


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_records(args: argparse.Namespace) -> Tuple[List[str], List[str], dict]:
    ids: List[str] = []
    texts: List[str] = []
    extra = {}
    for obj in read_jsonl(Path(args.input_jsonl)):
        if args.composite_id_fields:
            fields = [x.strip() for x in args.composite_id_fields.split(",") if x.strip()]
            vals = [str(obj.get(field, "")).strip() for field in fields]
            rec_id = "::".join(vals) if all(vals) else ""
        elif obj.get("paper_id") and obj.get("node_id"):
            rec_id = f"{obj['paper_id']}::{obj['node_id']}"
        else:
            rec_id = str(obj.get(args.id_field) or obj.get("paper_id") or obj.get("node_id") or "")
        if not rec_id:
            continue
        if args.title_field and args.abstract_field:
            title = str(obj.get(args.title_field) or "")
            abstract = str(obj.get(args.abstract_field) or "")
            text = title + "\n" + abstract
        else:
            text = str(
                obj.get(args.text_field)
                or obj.get("requirement")
                or obj.get("prompt")
                or ""
            )
        text = " ".join(text.split())
        if not text:
            continue
        ids.append(rec_id)
        texts.append(text)
        extra[rec_id] = {
            "text_chars": len(text),
            "source_jsonl": str(args.input_jsonl),
        }
    return ids, texts, extra


def embed_specter2(args: argparse.Namespace, texts: List[str]) -> np.ndarray:
    import torch
    from transformers import AutoTokenizer

    try:
        from adapters import AutoAdapterModel
    except ImportError as exc:
        raise SystemExit(
            "Missing package `adapters`. On the server run: pip install adapters transformers torch"
        ) from exc

    device = resolve_device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoAdapterModel.from_pretrained(args.model)
    adapter_name = "allenai/specter2" if args.adapter == "proximity" else f"allenai/specter2_{args.adapter}"
    model.load_adapter(adapter_name, source="hf", load_as=args.adapter, set_active=True)
    model.to(device)
    model.eval()

    out = []
    sep = tokenizer.sep_token or "[SEP]"
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), args.batch_size), desc="embedding"):
            batch = [t.replace("\n", f" {sep} ") for t in texts[i : i + args.batch_size]]
            inputs = tokenizer(
                batch,
                padding=True,
                truncation=True,
                return_tensors="pt",
                return_token_type_ids=False,
                max_length=args.max_length,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            emb = model(**inputs).last_hidden_state[:, 0, :].detach().cpu().numpy()
            out.append(emb)
    arr = np.vstack(out).astype(np.float32)
    if args.normalize:
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        arr = arr / np.maximum(norms, 1e-12)
    return arr


def embed_sentence_transformers(args: argparse.Namespace, texts: List[str]) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    device = resolve_device(args.device)
    model = SentenceTransformer(args.sentence_transformer_model, device=device)
    arr = model.encode(
        texts,
        batch_size=args.batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=args.normalize,
    )
    return arr.astype(np.float32)


def main() -> None:
    args = parse_args()
    ids, texts, extra = load_records(args)
    if not ids:
        raise SystemExit("No text records found to embed")

    if args.backend == "specter2":
        embeddings = embed_specter2(args, texts)
    else:
        embeddings = embed_sentence_transformers(args, texts)

    save_embedding_table(
        Path(args.ids_out),
        Path(args.embeddings_out),
        ids,
        embeddings,
        extra,
    )
    print(f"records={len(ids)}")
    print(f"dim={embeddings.shape[1]}")
    print(f"ids_out={args.ids_out}")
    print(f"embeddings_out={args.embeddings_out}")


if __name__ == "__main__":
    main()
