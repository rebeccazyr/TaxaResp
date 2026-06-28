# HieRec Embedding Team Formation Server Pipeline

This pipeline prepares cached inputs locally or on a server, embeds texts with
the local OpenAI-compatible embedding server, builds expert/task node embeddings, and evaluates the fixed-budget
setting:

```text
for each task taxonomy node:
  write an LLM role description
  assign the best expert for the same node
deduplicate experts by best node score
select ground-truth team_size experts
evaluate precision/recall
```

## Install

```bash
pip install -r requirements_embedding_server.txt
```

For GPU, install the PyTorch build matching the server CUDA version first, then
install the rest of the requirements.

## 1. Prepare Data Caches

This scans `dblp.v12.json` once and writes paper text, expert evidence, task
nodes, taxonomy node texts, and LLM prompts.

```bash
python3 data_preprocess/prepare_hierec_embedding_inputs.py \
  --out-dir output/hierec_embedding_server_inputs \
  --max-experts 0 \
  --max-profile-nodes 120 \
  --max-evidence-papers-per-node 5 \
  --ancestor-depth 5
```

Use `--max-experts 100` for a small debug run. `--max-experts 0` means all
experts.

Outputs:

```text
output/hierec_embedding_server_inputs/paper_texts.jsonl
output/hierec_embedding_server_inputs/expert_node_evidence.jsonl
output/hierec_embedding_server_inputs/node_texts.jsonl
output/hierec_embedding_server_inputs/task_nodes.jsonl
output/hierec_embedding_server_inputs/task_node_prompts.jsonl
```

## 2. Generate LLM Task-Node Role Descriptions

Each row in `task_node_prompts.jsonl` asks the LLM:

```text
Given the task paper abstract and this taxonomy node subtree, what does this
specific expert role does this task require under this node?
```

Together helper using `openai/gpt-oss-120b`:

```bash
export TOGETHER_API_KEY="..."

python3 data_preprocess/generate_task_node_requirements_openai.py \
  --prompts-jsonl output/hierec_embedding_server_inputs/task_node_prompts.jsonl \
  --out-jsonl output/hierec_embedding_server_inputs/task_node_requirements.jsonl \
  --backend together \
  --model openai/gpt-oss-120b \
  --resume
```

To generate automatically inside `scripts/run_hierec_embedding_server_pipeline.sh`:

```bash
GENERATE_TASK_REQUIREMENTS=1 TOGETHER_API_KEY="..." \
  scripts/run_hierec_embedding_server_pipeline.sh
```

If using another LLM, produce the same JSONL shape. `requirement` is the text
embedded downstream; `role_description` is kept as an explicit alias:

```json
{"paper_id":"...","node_id":"...","requirement":"...","role_description":"...","key_capabilities":[],"evidence_from_abstract":[]}
```

OpenAI is still supported:

```bash
python3 data_preprocess/generate_task_node_requirements_openai.py \
  --prompts-jsonl output/hierec_embedding_server_inputs/task_node_prompts.jsonl \
  --out-jsonl output/hierec_embedding_server_inputs/task_node_requirements.jsonl \
  --backend openai \
  --model gpt-4.1-mini \
  --resume
```

## 3. Embed Paper Evidence, Taxonomy Nodes, and Task Requirements

Default local embedding server from `agents.md`:

```text
Base URL: http://127.0.0.1:7823/v1
Model: Qwen3-Embedding-4B-4bit-DWQ
Dimension: 2560
```

Set the local API key from your local machine/server config:

```bash
export LOCAL_EMBEDDING_API_KEY="..."
```

Smoke test:

```bash
curl http://127.0.0.1:7823/v1/embeddings \
  -H "Authorization: Bearer $LOCAL_EMBEDDING_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen3-Embedding-4B-4bit-DWQ","input":"hello world"}'
```

Paper evidence embedding:

```bash
python3 data_preprocess/embed_jsonl_texts.py \
  --input-jsonl output/hierec_embedding_server_inputs/paper_texts.jsonl \
  --ids-out output/hierec_embedding_server_inputs/paper_embedding_ids.tsv \
  --embeddings-out output/hierec_embedding_server_inputs/paper_embeddings.npy \
  --backend openai-compatible \
  --model Qwen3-Embedding-4B-4bit-DWQ \
  --base-url http://127.0.0.1:7823/v1 \
  --api-key "$LOCAL_EMBEDDING_API_KEY" \
  --title-field title \
  --abstract-field abstract \
  --batch-size 32 \
  --normalize
```

Taxonomy node label embedding:

```bash
python3 data_preprocess/embed_jsonl_texts.py \
  --input-jsonl output/hierec_embedding_server_inputs/node_texts.jsonl \
  --ids-out output/hierec_embedding_server_inputs/node_embedding_ids.tsv \
  --embeddings-out output/hierec_embedding_server_inputs/node_embeddings.npy \
  --backend openai-compatible \
  --model Qwen3-Embedding-4B-4bit-DWQ \
  --base-url http://127.0.0.1:7823/v1 \
  --api-key "$LOCAL_EMBEDDING_API_KEY" \
  --text-field text \
  --batch-size 64 \
  --normalize
```

Task-node LLM role-description embedding:

```bash
python3 data_preprocess/embed_jsonl_texts.py \
  --input-jsonl output/hierec_embedding_server_inputs/task_node_requirements.jsonl \
  --ids-out output/hierec_embedding_server_inputs/task_requirement_embedding_ids.tsv \
  --embeddings-out output/hierec_embedding_server_inputs/task_requirement_embeddings.npy \
  --backend openai-compatible \
  --model Qwen3-Embedding-4B-4bit-DWQ \
  --base-url http://127.0.0.1:7823/v1 \
  --api-key "$LOCAL_EMBEDDING_API_KEY" \
  --text-field requirement \
  --composite-id-fields paper_id,node_id \
  --batch-size 64 \
  --normalize
```

## 4. Build Expert-Node Embeddings

```bash
python3 data_preprocess/build_expert_node_embeddings_from_cache.py \
  --expert-node-evidence-jsonl output/hierec_embedding_server_inputs/expert_node_evidence.jsonl \
  --paper-ids output/hierec_embedding_server_inputs/paper_embedding_ids.tsv \
  --paper-embeddings output/hierec_embedding_server_inputs/paper_embeddings.npy \
  --node-ids output/hierec_embedding_server_inputs/node_embedding_ids.tsv \
  --node-embeddings output/hierec_embedding_server_inputs/node_embeddings.npy \
  --ids-out output/hierec_embedding_server_inputs/expert_node_embedding_ids.tsv \
  --embeddings-out output/hierec_embedding_server_inputs/expert_node_embeddings.npy \
  --ancestor-depth 5
```

## 5. Build Task-Node Embeddings

```bash
python3 data_preprocess/build_task_node_embeddings_from_cache.py \
  --task-nodes-jsonl output/hierec_embedding_server_inputs/task_nodes.jsonl \
  --requirement-ids output/hierec_embedding_server_inputs/task_requirement_embedding_ids.tsv \
  --requirement-embeddings output/hierec_embedding_server_inputs/task_requirement_embeddings.npy \
  --node-ids output/hierec_embedding_server_inputs/node_embedding_ids.tsv \
  --node-embeddings output/hierec_embedding_server_inputs/node_embeddings.npy \
  --ids-out output/hierec_embedding_server_inputs/task_node_embedding_ids.tsv \
  --embeddings-out output/hierec_embedding_server_inputs/task_node_embeddings.npy \
  --node-weight 0.25
```

## 6. Evaluate Fixed-Budget Team Formation

```bash
python3 data_preprocess/evaluate_hierec_embedding_team_size.py \
  --task-nodes-jsonl output/hierec_embedding_server_inputs/task_nodes.jsonl \
  --task-node-ids output/hierec_embedding_server_inputs/task_node_embedding_ids.tsv \
  --task-node-embeddings output/hierec_embedding_server_inputs/task_node_embeddings.npy \
  --expert-node-ids output/hierec_embedding_server_inputs/expert_node_embedding_ids.tsv \
  --expert-node-embeddings output/hierec_embedding_server_inputs/expert_node_embeddings.npy \
  --out-dir output/hierec_embedding_team_size_eval
```

Final outputs:

```text
output/hierec_embedding_team_size_eval/metrics_summary.tsv
output/hierec_embedding_team_size_eval/node_assignments.tsv
output/hierec_embedding_team_size_eval/predictions_team_size.tsv
```
