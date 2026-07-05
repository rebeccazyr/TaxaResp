# Stage-1 Server Handoff

This directory contains the KDD-local Stage-1 code and orchestration scripts.
The git commit intentionally excludes local data, generated outputs, logs,
caches, and checkpoints. Recreate or sync those artifacts on the server before
running training.

## Setup

From the repository root:

```bash
python -m venv opentf_venv
source opentf_venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install torch tqdm openai together
```

Then run KDD commands from `kdd/`:

```bash
cd kdd
```

If using OpenAI-compatible embedding or LLM APIs, provide either environment
variables or a `.env`, `.env.openai`, or `.env.together` file outside git:

```bash
export OPENAI_API_KEY=...
export OPENAI_BASE_URL=...   # optional for compatible local/proxy endpoints
export TOGETHER_API_KEY=...  # only needed for Together generation
```

## Required Artifacts

The canonical `scripts/train_stage1.py` defaults require these external files.
They are generated artifacts and are not tracked in git.

Shared metadata:

- `../data/dblp/FieldsOfStudy.txt`
- `../data/dblp/13.FieldOfStudyChildren.nt`

Train split:

- `outputs/role_training_splits_author_count3_jsd_l0_ge0p03186_highconf2/train_5000.jsonl`
- `outputs/role_training_splits_author_count3_jsd_l0_ge0p03186_highconf2/train_role_descriptions_llm_gptoss120b/stage1_task_node_role_descriptions.jsonl`
- `outputs/role_training_splits_author_count3_jsd_l0_ge0p03186_highconf2/train_role_descriptions_llm_gptoss120b/role_description_embedding_ids.tsv`
- `outputs/role_training_splits_author_count3_jsd_l0_ge0p03186_highconf2/train_role_descriptions_llm_gptoss120b/role_description_embeddings.npy`
- `outputs/role_training_splits_author_count3_jsd_l0_ge0p03186_highconf2/train_history_paper_embeddings_inputs/author_history_papers.tsv`
- `outputs/role_training_splits_author_count3_jsd_l0_ge0p03186_highconf2/train_history_paper_embeddings_inputs/history_paper_embedding_ids.tsv`
- `outputs/role_training_splits_author_count3_jsd_l0_ge0p03186_highconf2/train_history_paper_embeddings_inputs/history_paper_embeddings.npy`
- `outputs/role_training_splits_author_count3_jsd_l0_ge0p03186_highconf2/train_history_paper_embeddings_inputs/history_paper_fos_weights.tsv`

Dev split:

- `outputs/role_training_splits_author_count3_jsd_l0_ge0p03186_highconf2/test_500.jsonl`
- `outputs/role_training_splits_author_count3_jsd_l0_ge0p03186_highconf2/test_role_descriptions_llm_gptoss120b/stage1_task_node_role_descriptions.jsonl`
- `outputs/role_training_splits_author_count3_jsd_l0_ge0p03186_highconf2/test_role_descriptions_llm_gptoss120b/role_description_embedding_ids.tsv`
- `outputs/role_training_splits_author_count3_jsd_l0_ge0p03186_highconf2/test_role_descriptions_llm_gptoss120b/role_description_embeddings.npy`
- `outputs/role_training_splits_author_count3_jsd_l0_ge0p03186_highconf2/test_history_paper_embeddings_inputs/author_history_papers.tsv`
- `outputs/role_training_splits_author_count3_jsd_l0_ge0p03186_highconf2/test_history_paper_embeddings_inputs/history_paper_embedding_ids.tsv`
- `outputs/role_training_splits_author_count3_jsd_l0_ge0p03186_highconf2/test_history_paper_embeddings_inputs/history_paper_embeddings.npy`
- `outputs/role_training_splits_author_count3_jsd_l0_ge0p03186_highconf2/test_history_paper_embeddings_inputs/history_paper_fos_weights.tsv`

Official 2019 hist>=5 test split:

- `outputs/role_test_set_2019_author_count3_jsd_l0_ge0p03186_highconf2_histge5/test_500.jsonl`
- `outputs/role_test_set_2019_author_count3_jsd_l0_ge0p03186_highconf2_histge5/test_role_descriptions_llm_gptoss120b/stage1_task_node_role_descriptions.jsonl`
- `outputs/role_test_set_2019_author_count3_jsd_l0_ge0p03186_highconf2_histge5/test_role_descriptions_llm_gptoss120b/role_description_embedding_ids.tsv`
- `outputs/role_test_set_2019_author_count3_jsd_l0_ge0p03186_highconf2_histge5/test_role_descriptions_llm_gptoss120b/role_description_embeddings.npy`
- `outputs/role_test_set_2019_author_count3_jsd_l0_ge0p03186_highconf2_histge5/test_history_embedding_inputs/author_history_papers.tsv`
- `outputs/role_test_set_2019_author_count3_jsd_l0_ge0p03186_highconf2_histge5/test_history_embeddings_openai/history_paper_embedding_ids.tsv`
- `outputs/role_test_set_2019_author_count3_jsd_l0_ge0p03186_highconf2_histge5/test_history_embeddings_openai/history_paper_embeddings.npy`
- `outputs/role_test_set_2019_author_count3_jsd_l0_ge0p03186_highconf2_histge5/test_history_embedding_inputs/history_paper_fos_weights.tsv`

Expert profiles:

- `outputs/expert_profile_cutoffs/pre_2018_for_valid_2018/`
- `outputs/expert_profile_cutoffs/pre_2019_for_test_2019_2020/`

## Artifact Preflight

Run a one-paper CPU preflight before starting the full job:

```bash
python scripts/train_stage1.py \
  --epochs 1 \
  --max-train-papers 1 \
  --max-eval-papers 1 \
  --batch-size 1 \
  --device cpu \
  --out-dir outputs/stage1_training/server_preflight
```

If any required path is missing, `train_stage1.py` prints
`missing_required_stage1_artifacts=1` followed by the exact missing paths.

## Canonical Full Run

```bash
python scripts/train_stage1.py \
  --node-link-mode minimal_tree \
  --node-weight-mode unweighted \
  --negative-mode v1 \
  --negative-pool-mode untrained_topm \
  --untrained-negative-top-m 20 \
  --epochs 10 \
  --batch-size 8 \
  --out-dir outputs/stage1_training/task_expert_node_dev_and_test_v1_untrained_topm20
```

## Useful Regeneration Commands

If the server has raw DBLP and API credentials instead of synced artifacts,
the full command history is recorded in `AGENTS.md`. The most relevant scripts
are:

- `scripts/build_cutoff_expert_profiles.py`
- `scripts/build_temporal_validation_test_sets.py`
- `scripts/select_cross_domain_eval_candidates.py`
- `scripts/build_role_training_splits.py`
- `scripts/generate_stage1_task_node_role_descriptions.py`
- `scripts/export_stage1_smoke_history_paper_texts.py`
- `scripts/export_history_paper_fos_weights.py`
- `scripts/embed_jsonl_texts_openai_parallel.py`
- `scripts/train_stage1.py`
