# HieRec Embedding Team Formation Server Pipeline

This pipeline prepares cached inputs locally or on a server, embeds texts with
SPECTER2, builds expert/task node embeddings, and evaluates the fixed-budget
setting:

```text
for each task taxonomy node:
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

## 2. Generate LLM Task-Node Requirements

Each row in `task_node_prompts.jsonl` asks the LLM:

```text
Given the task paper abstract and this taxonomy node subtree, what does this
specific task require under this node?
```

Optional OpenAI helper:

```bash
python3 data_preprocess/generate_task_node_requirements_openai.py \
  --prompts-jsonl output/hierec_embedding_server_inputs/task_node_prompts.jsonl \
  --out-jsonl output/hierec_embedding_server_inputs/task_node_requirements.jsonl \
  --model gpt-4.1-mini \
  --resume
```

If using a local LLM, produce the same JSONL shape:

```json
{"paper_id":"...","node_id":"...","requirement":"...","key_capabilities":[],"evidence_from_abstract":[]}
```

## 3. Embed Paper Evidence, Taxonomy Nodes, and Task Requirements

Paper evidence uses SPECTER2 `proximity`.

```bash
python3 data_preprocess/embed_jsonl_texts.py \
  --input-jsonl output/hierec_embedding_server_inputs/paper_texts.jsonl \
  --ids-out output/hierec_embedding_server_inputs/paper_embedding_ids.tsv \
  --embeddings-out output/hierec_embedding_server_inputs/paper_embeddings.npy \
  --backend specter2 \
  --adapter proximity \
  --title-field title \
  --abstract-field abstract \
  --batch-size 32 \
  --device auto \
  --normalize
```

Taxonomy node labels use SPECTER2 `adhoc_query` because they are short text.

```bash
python3 data_preprocess/embed_jsonl_texts.py \
  --input-jsonl output/hierec_embedding_server_inputs/node_texts.jsonl \
  --ids-out output/hierec_embedding_server_inputs/node_embedding_ids.tsv \
  --embeddings-out output/hierec_embedding_server_inputs/node_embeddings.npy \
  --backend specter2 \
  --adapter adhoc_query \
  --text-field text \
  --batch-size 64 \
  --device auto \
  --normalize
```

Task-node LLM requirements also use `adhoc_query`.

```bash
python3 data_preprocess/embed_jsonl_texts.py \
  --input-jsonl output/hierec_embedding_server_inputs/task_node_requirements.jsonl \
  --ids-out output/hierec_embedding_server_inputs/task_requirement_embedding_ids.tsv \
  --embeddings-out output/hierec_embedding_server_inputs/task_requirement_embeddings.npy \
  --backend specter2 \
  --adapter adhoc_query \
  --text-field requirement \
  --composite-id-fields paper_id,node_id \
  --batch-size 64 \
  --device auto \
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
