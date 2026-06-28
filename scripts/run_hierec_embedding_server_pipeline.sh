#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="${OUT_DIR:-output/hierec_embedding_server_inputs}"
EVAL_DIR="${EVAL_DIR:-output/hierec_embedding_team_size_eval}"
MAX_EXPERTS="${MAX_EXPERTS:-0}"
DEVICE="${DEVICE:-auto}"
BATCH_SIZE="${BATCH_SIZE:-32}"
EMBED_BACKEND="${EMBED_BACKEND:-openai-compatible}"
EMBED_MODEL="${EMBED_MODEL:-Qwen3-Embedding-4B-4bit-DWQ}"
EMBED_BASE_URL="${EMBED_BASE_URL:-http://127.0.0.1:7823/v1}"
EMBED_API_KEY="${EMBED_API_KEY:-${LOCAL_EMBEDDING_API_KEY:-}}"
GENERATE_TASK_REQUIREMENTS="${GENERATE_TASK_REQUIREMENTS:-0}"
LLM_BACKEND="${LLM_BACKEND:-together}"
LLM_MODEL="${LLM_MODEL:-openai/gpt-oss-120b}"
LLM_API_KEY="${LLM_API_KEY:-${TOGETHER_API_KEY:-}}"

if [[ "$EMBED_BACKEND" == "openai-compatible" && -z "$EMBED_API_KEY" ]]; then
  echo "Missing LOCAL_EMBEDDING_API_KEY or EMBED_API_KEY for openai-compatible embedding backend."
  exit 1
fi

python3 data_preprocess/prepare_hierec_embedding_inputs.py \
  --out-dir "$OUT_DIR" \
  --max-experts "$MAX_EXPERTS" \
  --max-profile-nodes 120 \
  --max-evidence-papers-per-node 5 \
  --ancestor-depth 5

if [[ ! -f "$OUT_DIR/task_node_requirements.jsonl" ]]; then
  if [[ "$GENERATE_TASK_REQUIREMENTS" == "1" ]]; then
    python3 data_preprocess/generate_task_node_requirements_openai.py \
      --prompts-jsonl "$OUT_DIR/task_node_prompts.jsonl" \
      --out-jsonl "$OUT_DIR/task_node_requirements.jsonl" \
      --backend "$LLM_BACKEND" \
      --model "$LLM_MODEL" \
      --api-key "$LLM_API_KEY" \
      --resume
  else
    echo "Missing $OUT_DIR/task_node_requirements.jsonl"
    echo "Generate it from $OUT_DIR/task_node_prompts.jsonl before continuing."
    echo "To generate with Together, set GENERATE_TASK_REQUIREMENTS=1 and TOGETHER_API_KEY."
    exit 1
  fi
fi

python3 data_preprocess/embed_jsonl_texts.py \
  --input-jsonl "$OUT_DIR/paper_texts.jsonl" \
  --ids-out "$OUT_DIR/paper_embedding_ids.tsv" \
  --embeddings-out "$OUT_DIR/paper_embeddings.npy" \
  --backend "$EMBED_BACKEND" \
  --model "$EMBED_MODEL" \
  --base-url "$EMBED_BASE_URL" \
  --api-key "$EMBED_API_KEY" \
  --title-field title \
  --abstract-field abstract \
  --batch-size "$BATCH_SIZE" \
  --normalize

python3 data_preprocess/embed_jsonl_texts.py \
  --input-jsonl "$OUT_DIR/node_texts.jsonl" \
  --ids-out "$OUT_DIR/node_embedding_ids.tsv" \
  --embeddings-out "$OUT_DIR/node_embeddings.npy" \
  --backend "$EMBED_BACKEND" \
  --model "$EMBED_MODEL" \
  --base-url "$EMBED_BASE_URL" \
  --api-key "$EMBED_API_KEY" \
  --text-field text \
  --batch-size "$BATCH_SIZE" \
  --normalize

python3 data_preprocess/embed_jsonl_texts.py \
  --input-jsonl "$OUT_DIR/task_node_requirements.jsonl" \
  --ids-out "$OUT_DIR/task_requirement_embedding_ids.tsv" \
  --embeddings-out "$OUT_DIR/task_requirement_embeddings.npy" \
  --backend "$EMBED_BACKEND" \
  --model "$EMBED_MODEL" \
  --base-url "$EMBED_BASE_URL" \
  --api-key "$EMBED_API_KEY" \
  --text-field requirement \
  --composite-id-fields paper_id,node_id \
  --batch-size "$BATCH_SIZE" \
  --normalize

python3 data_preprocess/build_expert_node_embeddings_from_cache.py \
  --expert-node-evidence-jsonl "$OUT_DIR/expert_node_evidence.jsonl" \
  --paper-ids "$OUT_DIR/paper_embedding_ids.tsv" \
  --paper-embeddings "$OUT_DIR/paper_embeddings.npy" \
  --node-ids "$OUT_DIR/node_embedding_ids.tsv" \
  --node-embeddings "$OUT_DIR/node_embeddings.npy" \
  --ids-out "$OUT_DIR/expert_node_embedding_ids.tsv" \
  --embeddings-out "$OUT_DIR/expert_node_embeddings.npy" \
  --ancestor-depth 5

python3 data_preprocess/build_task_node_embeddings_from_cache.py \
  --task-nodes-jsonl "$OUT_DIR/task_nodes.jsonl" \
  --requirement-ids "$OUT_DIR/task_requirement_embedding_ids.tsv" \
  --requirement-embeddings "$OUT_DIR/task_requirement_embeddings.npy" \
  --node-ids "$OUT_DIR/node_embedding_ids.tsv" \
  --node-embeddings "$OUT_DIR/node_embeddings.npy" \
  --ids-out "$OUT_DIR/task_node_embedding_ids.tsv" \
  --embeddings-out "$OUT_DIR/task_node_embeddings.npy" \
  --node-weight 0.25

python3 data_preprocess/evaluate_hierec_embedding_team_size.py \
  --task-nodes-jsonl "$OUT_DIR/task_nodes.jsonl" \
  --task-node-ids "$OUT_DIR/task_node_embedding_ids.tsv" \
  --task-node-embeddings "$OUT_DIR/task_node_embeddings.npy" \
  --expert-node-ids "$OUT_DIR/expert_node_embedding_ids.tsv" \
  --expert-node-embeddings "$OUT_DIR/expert_node_embeddings.npy" \
  --out-dir "$EVAL_DIR"
