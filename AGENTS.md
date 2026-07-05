# Agent Notes

## Project Goal

- OpeNTF is an open-source neural team formation benchmark/library for expert
  team recommendation, including preprocessing, training, testing, evaluation,
  temporal strategies, and fairness-aware reranking.

## Directory Rules

- Always inspect and clean the project structure before coding.
- Keep reusable library code in `src/`.
- Keep runnable shell/Python entrypoint scripts in `scripts/`.
- Keep raw, interim, and processed datasets under `data/`, using
  `data/raw/`, `data/interim/`, and `data/processed/` for new pipelines when
  practical.
- Keep generated results in `outputs/`. Existing legacy outputs currently live
  under `output/`; preserve those paths unless the pipeline is migrated
  deliberately.
- Keep logs in `logs/`. Move new run logs there instead of leaving them in the
  repository root.
- Keep temporary files, scratch artifacts, and local caches in `cache/`.
- Never commit raw or generated data unless explicitly instructed.
- Do not commit model checkpoints or machine-specific artifacts. Generic secrets
  files are ignored by default, but project-approved local keys documented in
  `AGENTS.md` may be committed.
- Always create or update `.gitignore` so it excludes data folders, generated
  outputs, logs, cache, `.DS_Store`, `__pycache__`, notebook checkpoints,
  virtual environments, model checkpoints, and secrets.
- Always keep this `AGENTS.md` updated as project-local memory with project
  goal, directory rules, data inventory, pipeline commands, important decisions,
  and TODOs.

## Data Inventory

- `kdd/`: clean workspace for new KDD-focused code and experiments. Use
  `kdd/src/` for reusable code, `kdd/scripts/` for runnable entrypoints, and
  `kdd/data/`, `kdd/outputs/`, `kdd/logs/`, and `kdd/cache/` for local data and
  generated artifacts.
- `kdd/data/dblp/dblp.v12.json`: full DBLP v12 JSON. The legacy path
  `data/dblp/dblp.v12.json` is kept as a compatibility symlink for existing
  scripts.
- `data/dblp/`: legacy DBLP publication/team formation data and taxonomy files.
- `data/uspt/`: USPTO patent data.
- `data/imdb/`: IMDb movie/cast data.
- `data/gith/`: GitHub repository/developer data.
- `output/`: legacy generated outputs, including expert profiles, embeddings,
  graph artifacts, taxonomy experiments, and domain-specific processed files.
- `logs/` and root `*.log`: run logs; place new logs under `logs/`.

## Pipeline Commands

- Environment setup:

```bash
python -m venv opentf_venv
source opentf_venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

- README quickstart:

```bash
cd src
python main.py "cmd=[prep, train, test, eval]" \
               "models.instances=[mdl.rnd.Rnd, mdl.fnn.Fnn, mdl.bnn.Bnn]" \
               data.domain=cmn.publication.Publication data.source=../data/dblp/toy.dblp.v12.json data.output=../output/dblp/toy.dblp.v12.json \
               ~data.filter \
               train.train_test_ratio=0.85 train.nfolds=3 train.save_per_epoch=3 \
               test.per_epoch=True test.topK=100 \
               eval.topk=\'2,5,10\'
```

- Local HiERec embedding-server pipeline:

```bash
scripts/run_hierec_embedding_server_pipeline.sh
```

- Taxonomy region-cut embedding evaluation on the 2020plus test set:

```bash
python data_preprocess/evaluate_embedding_taxonomy_region_cut.py \
  --task-nodes-jsonl output/hierec_embedding_server_inputs/task_nodes.jsonl \
  --task-node-ids output/all_expert_paper_embeddings/task_node_embedding_ids_strict_v2_no_label.tsv \
  --task-node-embeddings output/all_expert_paper_embeddings/task_node_embeddings_strict_v2_no_label.npy \
  --expert-node-ids output/all_expert_paper_embeddings/expert_node_embedding_ids_no_label.tsv \
  --expert-node-embeddings output/all_expert_paper_embeddings/expert_node_embeddings_no_label.npy \
  --out-dir output/embedding_taxonomy_region_cut_jsd_topm256_temp015_no_label \
  --top-m 256 \
  --distribution-temperature 0.15
```

- Taxonomy region-cut parameter tuning on the 2020plus test set:

```bash
python data_preprocess/tune_embedding_taxonomy_region_cut.py \
  --task-nodes-jsonl output/hierec_embedding_server_inputs/task_nodes.jsonl \
  --task-node-ids output/all_expert_paper_embeddings/task_node_embedding_ids_strict_v2_no_label.tsv \
  --task-node-embeddings output/all_expert_paper_embeddings/task_node_embeddings_strict_v2_no_label.npy \
  --expert-node-ids output/all_expert_paper_embeddings/expert_node_embedding_ids_no_label.tsv \
  --expert-node-embeddings output/all_expert_paper_embeddings/expert_node_embeddings_no_label.npy \
  --out-dir output/embedding_taxonomy_region_cut_jsd_topm_tuning_no_label \
  --top-m-grid 8,16,32,64,128,256 \
  --temperature-grid 0.01,0.02,0.03,0.05,0.08,0.1,0.15,0.2,0.3,0.5 \
  --repeat-grid unique,repeat \
  --objective mean_recall_at_team_size
```

- Weighted specialization score evaluation for groundtruth and aligned
  prediction TSVs:

```bash
python data_preprocess/compute_weighted_specscore.py \
  --task-nodes-jsonl output/hierec_embedding_server_inputs/task_nodes.jsonl \
  --task-node-ids output/all_expert_paper_embeddings/task_node_embedding_ids_strict_v2_no_label.tsv \
  --task-node-embeddings output/all_expert_paper_embeddings/task_node_embeddings_strict_v2_no_label.npy \
  --expert-node-ids output/all_expert_paper_embeddings/expert_node_embedding_ids_no_label.tsv \
  --expert-node-embeddings output/all_expert_paper_embeddings/expert_node_embeddings_no_label.npy \
  --out-dir output/specscore_weighted \
  --predictions-tsv output/hierec_embedding_team_formation_experiment_20/predictions_team_size.tsv \
  --prediction-label seqseq \
  --random-baseline \
  --random-seed 13
```

- Weighted specialization score evaluation for an OpenTF seq2seq token
  prediction file:

```bash
python data_preprocess/compute_weighted_specscore.py \
  --task-nodes-jsonl output/hierec_embedding_server_inputs/task_nodes.jsonl \
  --task-node-ids output/all_expert_paper_embeddings/task_node_embedding_ids_strict_v2_no_label.tsv \
  --task-node-embeddings output/all_expert_paper_embeddings/task_node_embeddings_strict_v2_no_label.npy \
  --expert-node-ids output/all_expert_paper_embeddings/expert_node_embedding_ids_no_label.tsv \
  --expert-node-embeddings output/all_expert_paper_embeddings/expert_node_embeddings_no_label.npy \
  --out-dir output/specscore_weighted_seq2seq_epoch15527 \
  --opentf-token-pred-csv output/test.fold0.epoch15527.pred.csv \
  --opentf-token-label seq2seq_fold0_epoch15527 \
  --indexes-pkl output/indexes.pkl
```

- Soft-groundtruth P/R evaluation for cached history-team, seq2seq, embedding
  BFS, owner-gain/responsibility cut, expert-distribution cut, and random
  baseline predictions:

```bash
python scripts/evaluate_soft_groundtruth_methods.py \
  --out-dir output/soft_groundtruth_evaluation
```

- Soft-groundtruth citation Louvain resolution grid evaluation:

```bash
python scripts/evaluate_soft_groundtruth_methods.py \
  --out-dir output/soft_groundtruth_evaluation_louvain_grid \
  --citation-community-method louvain \
  --citation-louvain-resolution-grid 1,2,5,10
```

- Embedding-similarity pred-gold expert pair distance distribution for selected
  methods:

```bash
python scripts/plot_embedding_similarity_pair_distribution.py \
  --out-dir output/selected_method_results \
  --threshold 0.30
```

- Full pairwise expert mean-paper embedding distance distribution matching the
  `Embedding Similarity P/R` expert representation:

```bash
python scripts/plot_expert_embedding_pair_distance_distribution.py \
  --out-dir output/selected_method_results \
  --threshold 0.30
```

- Full same-node task-node/expert-node embedding distance distribution:

```bash
python scripts/plot_task_expert_embedding_distance_distribution.py \
  --out-dir output/selected_method_results \
  --threshold 0.30
```

- Higher-resolution citation Louvain grid used to reduce random-baseline
  saturation:

```bash
python scripts/evaluate_soft_groundtruth_methods.py \
  --out-dir output/soft_groundtruth_evaluation_louvain_grid_high \
  --citation-community-method louvain \
  --citation-louvain-resolution-grid 10,20,50,100
```

- Global all-expert citation Louvain blocks and method metrics for all 13,815
  expert profiles:

```bash
python scripts/evaluate_global_louvain_blocks.py \
  --out-dir output/global_louvain_blocks \
  --resolutions 10,20,50,100
```

- Additional global all-expert citation Louvain high-resolution run:

```bash
python scripts/evaluate_global_louvain_blocks.py \
  --out-dir output/global_louvain_blocks_r200_500 \
  --resolutions 200,500
```

- Expert-node citation graph Louvain evaluation, where nodes are experts and
  weighted edges count citation links between their 2000-2019 historical papers:

```bash
python scripts/evaluate_expert_citation_louvain_blocks.py \
  --out-dir output/expert_citation_louvain_blocks \
  --resolutions 1,2,5,10,20,50,100
```

- 2020plus test-set groundtruth authors' dispersion across expert-node
  citation Louvain blocks at resolution 2. The script reuses an existing
  `expert_id -> block_id` cache when available and writes both membership and
  block-member tables for later reuse:

```bash
python scripts/analyze_test_author_citation_block_dispersion.py \
  --out-dir output/test_author_citation_block_dispersion_r2
```

- Prompt generation for direct, pre-completion FoS task nodes on the level-1
  direct-FoS cross-domain filtered test set. By default this writes prompts
  only for the first three papers and does not call an LLM; add `--generate`
  after checking prompts to call the configured backend:

```bash
python scripts/generate_direct_fos_node_role_descriptions.py
```

- Owner-gain taxonomy region-cut evaluation on the 2020plus test set:

```bash
python data_preprocess/evaluate_embedding_taxonomy_owner_gain_cut.py \
  --task-nodes-jsonl output/hierec_embedding_server_inputs/task_nodes.jsonl \
  --task-node-ids output/all_expert_paper_embeddings/task_node_embedding_ids_strict_v2_no_label.tsv \
  --task-node-embeddings output/all_expert_paper_embeddings/task_node_embeddings_strict_v2_no_label.npy \
  --expert-node-ids output/all_expert_paper_embeddings/expert_node_embedding_ids_no_label.tsv \
  --expert-node-embeddings output/all_expert_paper_embeddings/expert_node_embeddings_no_label.npy \
  --out-dir output/embedding_taxonomy_owner_gain_cut_topm256_no_label \
  --top-m 256
```

- Fill selected-method task coverage and team-structure metrics for
  Embedding BFS, responsibility cut, expert-distribution cut, Seq2seq, and
  Random mean 5:

```bash
python scripts/compute_selected_method_missing_metrics.py
```

- Direct test-abstract to aggregated-user embedding search, matching each
  task's prediction count to BFS Unique Assignment:

```bash
python scripts/evaluate_abstract_user_embedding_search.py \
  --out-dir output/abstract_user_embedding_search_qwen2560
```

- Direct test-paper abstract embedding search over each expert's all-paper mean
  abstract embedding, retrieving `top_k = groundtruth team size` and reporting
  exact-member P/R/F1:

```bash
python scripts/evaluate_test_abstract_expert_mean_search.py \
  --out-dir output/abstract_expert_mean_embedding_search
```

- Virtual-root role description generation, global expert mean-paper embedding
  construction, and root-role expert retrieval:

```bash
python scripts/generate_virtual_root_roles.py \
  --task-nodes-jsonl output/hierec_embedding_server_inputs/task_nodes.jsonl \
  --out-jsonl output/virtual_root_role_descriptions/root_role_texts.jsonl \
  --backend template

# LLM version matching task-node generation defaults:
python scripts/generate_virtual_root_roles.py \
  --task-nodes-jsonl output/hierec_embedding_server_inputs/task_nodes.jsonl \
  --out-jsonl output/virtual_root_role_descriptions/root_role_texts_llm_gptoss120b.jsonl \
  --backend together \
  --model openai/gpt-oss-120b \
  --temperature 0 \
  --max-tokens 2048 \
  --workers 8 \
  --max-in-flight 16

python scripts/build_expert_mean_paper_embeddings.py \
  --expert-papers-tsv output/all_expert_paper_embeddings/expert_papers.tsv \
  --paper-ids-tsv output/all_expert_paper_embeddings/paper_embedding_ids.tsv \
  --paper-embeddings output/all_expert_paper_embeddings/paper_embeddings.npy \
  --ids-out output/virtual_root_role_descriptions/expert_mean_paper_embedding_ids.tsv \
  --embeddings-out output/virtual_root_role_descriptions/expert_mean_paper_embeddings.npy

python data_preprocess/embed_jsonl_texts_openai_parallel.py \
  --input-jsonl output/virtual_root_role_descriptions/root_role_texts.jsonl \
  --ids-out output/virtual_root_role_descriptions/root_role_embedding_ids.tsv \
  --embeddings-out output/virtual_root_role_descriptions/root_role_embeddings.npy \
  --model text-embedding-3-small \
  --id-field id \
  --text-field text \
  --batch-size 128 \
  --workers 4 \
  --max-in-flight 4 \
  --normalize

python scripts/match_virtual_root_roles_to_experts.py \
  --root-role-ids output/virtual_root_role_descriptions/root_role_embedding_ids.tsv \
  --root-role-embeddings output/virtual_root_role_descriptions/root_role_embeddings.npy \
  --expert-ids output/virtual_root_role_descriptions/expert_mean_paper_embedding_ids.tsv \
  --expert-embeddings output/virtual_root_role_descriptions/expert_mean_paper_embeddings.npy \
  --out-tsv output/virtual_root_role_descriptions/virtual_root_expert_matches.tsv \
  --top-k 20 \
  --predictions-top1-tsv output/virtual_root_role_descriptions/virtual_root_top1_predictions.tsv
```

- Simplified tree-knapsack DP over selected region count only, using each
  taxonomy node's same-node rank-1 expert and a configurable virtual-root
  policy (`optional`, `forced`, or `none`; default `optional`):

```bash
python scripts/evaluate_tree_knapsack_dp_regions.py \
  --root-policy optional \
  --out-dir output/tree_knapsack_dp_regions_llm_root
```

- Full TaxaResp-DP evaluation with `DP_u[x,k]` state, counted virtual-root
  responsibility region, cached top-256 same-node candidates, and no
  unique-owner constraint:

```bash
python scripts/evaluate_taxaresp_dp.py \
  --out-dir output/taxaresp_dp_full_with_virtual_root_topm256_no_unique_owner
```

- Full TaxaResp-DP evaluation without virtual-root scoring or virtual-root
  expert candidates. The virtual root remains only as an internal connector for
  multiple taxonomy roots:

```bash
python scripts/evaluate_taxaresp_dp.py \
  --virtual-root-mode connector \
  --max-root-rank 0 \
  --out-dir output/taxaresp_dp_full_no_virtual_root_topm256_no_unique_owner \
  --method-label taxaresp_dp_full_no_virtual_root
```

## Important Decisions

- New reusable implementation should go under `src/`; use `scripts/` only for
  runnable orchestration.
- Preserve existing `output/` paths until callers/configs are migrated; use
  `outputs/` for new generated-result locations.
- The local embedding service runs outside the Codex sandbox. If a sandboxed
  command cannot reach it, rerun the embedding/smoke-test command with elevated
  permissions before assuming the service is down.
- `data_preprocess/evaluate_embedding_taxonomy_region_cut.py` implements the
  region-cut method: top-M same-node expert matching, softmax expert
  distributions per task node, JSD edge boundary scoring, highest-boundary edge
  cuts, and unique region-owner assignment via maximum weight bipartite
  matching. Responsibility overlap is reported only as an evaluation metric.
- A 2020plus grid search selected `top_m=256`, `temperature=0.15`, and unique
  owner matching as the best observed setting by mean recall at team size.
- `data_preprocess/compute_weighted_specscore.py` implements the weighted
  specialization metric: for each selected expert, normalize
  `node_importance * same-node embedding similarity` across task FoS nodes, then
  report the team mean pairwise JSD. Negative similarities are clamped to zero
  by default so the normalized values remain a probability distribution.
- `data_preprocess/evaluate_embedding_taxonomy_owner_gain_cut.py` replaces JSD
  cut scoring with greedy owner-assignment gain: each cut is selected by the
  marginal improvement in unique region-owner matching objective.
- `scripts/evaluate_soft_groundtruth_methods.py` evaluates cached predictions
  with history-team members as anchors but soft matching by maximum bipartite
  assignment: exact-history-member matching requires identical expert IDs;
  citation blocks default to Louvain communities over the DBLP citation graph
  induced by candidate experts' historical papers; user embedding matches use
  cosine distance between mean historical-paper embeddings; random baseline
  predictions default to five seeded samples from the non-random evaluated
  expert pool. Use `--citation-louvain-resolution-grid` to compare citation
  block granularity in one graph-build pass. In the high-resolution Louvain grid
  on the 2020plus evaluation set, random citation P/R dropped from 73.37 at
  resolution 10 to 46.76 at resolution 100, while embedding BFS remained
  91.16/89.58 at resolution 100.
- `scripts/evaluate_global_louvain_blocks.py` uses `python-igraph` to build
  global citation blocks from all 13,815 expert profiles' 2000-2019 historical
  papers. The full graph has 1,429,694 paper nodes and 11,603,774 citation
  edges. In the global r=10,20,50,100 run, average blocks per expert with
  history papers were 34.61, 38.76, 44.72, and 49.36 respectively; embedding
  BFS citation P/R was 98.66/97.08, 98.54/96.95, 97.14/95.55, and 95.18/93.66.
  In the additional r=200,500 run, average blocks per expert with history
  papers were 54.13 and 60.57; embedding BFS citation P/R was 93.95/92.37 and
  89.55/88.03, while random citation P/R dropped to 42.69 and 31.08.
- `scripts/evaluate_expert_citation_louvain_blocks.py` builds an expert-node
  graph with 13,815 expert nodes and weighted undirected edges counting
  citation links between experts' 2000-2019 historical papers. The graph has
  5,233,665 expert-expert edges. At resolutions 1,2,5,10,20,50,100, community
  counts were 15,24,70,218,574,1566,2862; random P/R dropped from 16.18 to
  0.11, and high-resolution metrics approach exact-member matching.
- `scripts/compute_selected_method_missing_metrics.py` completes left-slide
  metrics for the five selected methods in `output/selected_method_results/`.
  It defines soft specialty coverage as selected-team max
  `direct_weight_sum(expert, task_skill) / total_direct_profile_weight(expert)`,
  defines authority coverage as selected-team max direct FoS weight normalized
  by the global best expert for each task skill, and reports responsibility
  compactness with taxonomy-structure assignment. For responsibility cut and
  expert-distribution cut, compactness uses each method's generated `regions.tsv`
  region-owner assignments; compactness is
  `assigned nodes / taxonomy tree-closure nodes`, averaged over selected experts
  and tasks. Post-hoc Specialty/Authority node assignment is retained only as an
  auxiliary baseline for methods without native region partitions. Random mean 5
  uses seeds 13-17 from the same method/history expert pool as the
  soft-groundtruth random baseline.
- `scripts/evaluate_abstract_user_embedding_search.py` compares direct test
  paper abstract embeddings to per-user mean historical evidence-paper
  embeddings and evaluates exact-history-member matching with the same per-task
  prediction counts as BFS Unique Assignment. The current
  `output/hierec_embedding_server_inputs/paper_embeddings.npy` cache is
  incomplete: only the first 2,048 rows are nonzero, and all 262 test abstract
  rows are zero vectors. Regenerate that cache before trusting direct abstract
  search results.
- `scripts/evaluate_test_abstract_expert_mean_search.py` evaluates direct test
  paper abstract embeddings from
  `output/virtual_root_role_descriptions/task_paper_text_embeddings.npy`
  against each expert's all-paper mean abstract embedding from
  `output/virtual_root_role_descriptions/expert_mean_paper_embeddings.npy`,
  retrieving `top_k = groundtruth team size` and reporting exact-member P/R/F1.
  First run on 262 test tasks: macro P/R/F1 = 14.656489/14.656489/14.656489,
  micro P/R/F1 = 15.189873/15.189873/15.189873, with 132 hits out of 869
  selected/gold experts.
- Virtual-root owner retrieval should match a whole-task role description
  against a global expert representation, not a same-node expert profile.
  `scripts/generate_virtual_root_roles.py` writes one overall role per paper;
  `scripts/build_expert_mean_paper_embeddings.py` builds global expert vectors
  by averaging historical paper embeddings; and
  `scripts/match_virtual_root_roles_to_experts.py` retrieves top-k root owners.
  The LLM virtual-root generation path supports the same default provider/model
  as task-node role generation: Together backend, `openai/gpt-oss-120b`,
  temperature 0. Use `--max-tokens 2048` to avoid truncated JSON; in the first
  full LLM run, 8 workers generated all 262 root descriptions successfully.
  The current `output/all_expert_paper_embeddings` cache is 1536-dimensional
  and should be paired with `text-embedding-3-small` root-role embeddings. Do
  not mix it with the 2560-dimensional Qwen cache under
  `output/hierec_embedding_server_inputs` unless the expert mean table is
  regenerated in the same embedding space. In the first template-root run,
  root top1 matched a gold team member for 37/262 tasks; top20 covered at least
  one gold member for 129/262 tasks. In the first Together
  `openai/gpt-oss-120b` LLM-root run, root top1 matched 40/262 tasks and top20
  covered 130/262 tasks. DP should prefer the root top-k candidate pool over a
  hard top1 root owner.
- `scripts/evaluate_tree_knapsack_dp_regions.py` implements the simplified
  `DP_u[m]` tree-knapsack pruning variant. Its state tracks only the number of
  selected responsibility regions, not the expert set: keep child means the
  child subtree stays covered by the current selected region root; cut child
  merges the child's DP table into the parent table. The script now supports
  `--root-policy optional|forced|none`. With `optional`, `DP_virtual_root[0]=0`
  makes the virtual root selectable but not mandatory; if the virtual root is
  not selected, its children must be cut so no subtree is silently uncovered.
  The 2020plus `optional` run with default `root_weight=all_skill_sum` selected
  the virtual root for all 262 tasks and matched the old forced-root metrics:
  raw-region macro P/R 11.316794/11.208651, micro P/R 11.425206/11.162255;
  dedup-expert macro P/R 15.375318/11.208651, micro P/R 13.642757/11.162255.
  With `optional --root-weight one`, the virtual root was selected for 213/262
  tasks and exact hard-match metrics were: raw-region macro P/R
  9.236641/9.128499, micro P/R 10.416667/9.205984; dedup-expert macro P/R
  12.767176/9.128499, micro P/R 12.461059/9.205984.
- `scripts/evaluate_taxaresp_dp.py` implements the full TaxaResp-DP recurrence
  from the method note: `DP_u[x,k]` fixes the owner of the open region
  containing node `u`, keep transitions require the same owner across the
  parent-child edge, cut transitions use owner-free child values
  `B_c[l]=max_y DP_c[y,l]`, and repeat owners across different regions are
  allowed. The task-level candidate pool is the union of cached per-node
  top-256 same-node experts plus virtual-root top-k experts; an expert missing
  for a node contributes similarity 0 by default. The default mode now treats
  the virtual root as a counted responsibility node with weight
  `all_skill_sum`; use `--virtual-root-mode connector` to preserve the previous
  zero-weight non-counted connector behavior. The first counted-root 2020plus
  run wrote `output/taxaresp_dp_full_with_virtual_root_topm256_no_unique_owner`:
  262 tasks, macro P/R 13.759542/13.651399, micro P/R
  14.016490/13.693901, dedup micro P/R 14.727723/13.693901, and 41 duplicate
  region-owner assignments. The virtual root appeared in one region for all
  262 tasks, including 18 root-only regions. Three tasks have
  `team_size > task_nodes + 1`, so the feasible region count is
  `min(team_size, task_nodes + 1)` for those tasks.
- The no-virtual-root TaxaResp-DP run uses `--virtual-root-mode connector` plus
  `--max-root-rank 0`, so `__task_root__` can still appear in cut-edge or
  augmented-region outputs as a zero-score connector, but it does not appear in
  region `node_ids`, does not contribute utility, and does not add root-retrieval
  experts to the candidate pool. The 2020plus run wrote
  `output/taxaresp_dp_full_no_virtual_root_topm256_no_unique_owner`: 262 tasks,
  macro P/R 14.554707/14.300254, micro P/R 14.779499/14.269275, dedup micro P/R
  15.422886/14.269275, and 35 duplicate region-owner assignments.
- The no-virtual-root TaxaResp-DP paper-citation-block soft-groundtruth
  evaluation wrote `output/soft_groundtruth_taxaresp_dp_no_virtual_root_thr023_res2`,
  using embedding cosine-distance threshold 0.23 and paper citation-block
  Louvain resolution 2. Under unified dedup/max-bipartite evaluation, Exact P/R
  is 15.597964/14.300254, Embedding Similarity P/R is 79.891858/75.057252, and
  paper Citation Block P/R is 98.791349/93.416031. Do not mix this 98% paper
  citation-block metric with the selected-method "Expert Citation P/R" table,
  which uses expert-node citation graph Louvain blocks.
- For the selected-method table convention, Embedding Similarity P/R uses
  cosine-distance threshold 0.2309, while Expert Citation P/R uses expert-node
  citation graph Louvain at resolution 2. A rerun including no-virtual-root
  TaxaResp-DP wrote `output/soft_groundtruth_taxaresp_dp_no_virtual_root_thr02309`
  for Exact/Embedding and
  `output/expert_citation_louvain_blocks_taxaresp_dp_no_virtual_root_r2` for
  expert-graph citation. In that same expert-graph rerun, no-virtual-root
  TaxaResp-DP had Exact P/R 15.597964/14.300254, Embedding Similarity P/R
  80.273537/75.438931, and Expert Citation P/R 61.011450/57.296438.
- `scripts/analyze_test_author_citation_block_dispersion.py` analyzes the
  2020plus groundtruth teams' author dispersion over expert-node citation
  Louvain blocks. The first run reused
  `output/selected_method_results/expert_graph_louvain_resolution2_membership.tsv`
  and wrote `output/test_author_citation_block_dispersion_r2`. That cached
  resolution-2 membership has 13,815 experts in 26 blocks. Across 262 test
  tasks and 869 gold authors, no authors were missing from the cache; 158 tasks
  had all authors in one block, 86 spanned two blocks, 17 spanned three blocks,
  and 1 spanned four blocks. The mean distinct block count per task was
  1.469465648855.

## TODOs

- Consider migrating legacy `output/` paths to `outputs/` in a coordinated
  pipeline/config update.
- Consider moving root-level historical `*.log` files into `logs/` after
  confirming no scripts depend on their current paths.
- Keep `data_preprocess/` cleanup on the backlog: reusable modules should move
  into `src/`, while runnable commands should move into `scripts/`.
- Connect virtual-root top-k owner candidates to the RR-DP pruning stage and
  evaluate exact, embedding-similarity, and citation P/R against the previous
  virtual-root aggregate-proxy owner.

## Local Embedding API

- The embedding API runs outside the Codex sandbox. If sandboxed commands cannot
  reach `127.0.0.1:7823`, rerun the embedding/smoke-test command with elevated
  permissions outside the sandbox instead of assuming the service is down.
- Base URL: `http://127.0.0.1:7823/v1`
- API key: `yiruiisthebest`
- Embedding model: `Qwen3-Embedding-4B-4bit-DWQ`
- Embedding dimension: `2560`

Smoke test:

```bash
curl http://127.0.0.1:7823/v1/embeddings \
  -H "Authorization: Bearer yiruiisthebest" \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen3-Embedding-4B-4bit-DWQ","input":"hello world"}'
```
