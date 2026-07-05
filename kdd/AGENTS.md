# Agent Notes

## Project Goal

- KDD-focused workspace for OpeNTF experiments on temporal expert team
  recommendation.

## Directory Rules

- Keep reusable implementation in `src/`.
- Keep runnable entrypoints in `scripts/`.
- Keep local data under `data/`, generated artifacts under `outputs/`, logs
  under `logs/`, and scratch/cache files under `cache/`.
- Do not commit raw DBLP data, generated profile outputs, logs, caches,
  checkpoints, virtual environments, or secrets.

## Data Inventory

- `data/dblp/dblp.v12.json`: full DBLP v12 JSON used for KDD temporal profile
  construction.
- `outputs/temporal_task_splits_full/`: full validation/test task splits
  exported directly from DBLP by year, without filtering authors through a
  legacy expert id list.
- `outputs/temporal_task_splits_full/validation_2018_all_authors_hist_ge5.jsonl`:
  60,680-paper 2018 validation subset where every paper author has at least
  five pre-2018 historical papers.
- `outputs/cross_domain_eval_selection/author_count3_jsd_l0_ge0p03186_highconf2/`:
  current role-aware training set selected from the 2018 hist>=5 pool by
  `author_count >= 3`, `author_jsd_l0_mean >= 0.03186`, and
  `high_conf_direct_node_count >= 2`.
- `outputs/role_training_splits_author_count3_jsd_l0_ge0p03186_highconf2/`:
  paper-disjoint train/test split sampled from the current role-aware training
  pool with groundtruth paper authors as teams: `train_5000.*` and
  `test_500.*`. Groundtruth-author pre-2018 history paper texts and OpenAI
  `text-embedding-3-small` normalized embeddings are under
  `train_history_paper_embeddings_inputs/` and
  `test_history_paper_embeddings_inputs/`.
- `outputs/cross_domain_eval_selection/test_2019_author_count3_jsd_l0_ge0p03186_highconf2/`:
  2019-only test candidate pool selected from
  `outputs/temporal_task_splits_full/test_2019.jsonl` by
  `author_count >= 3`, pre-2019 `author_jsd_l0_mean >= 0.03186`, and
  `high_conf_direct_node_count >= 2`.
- `outputs/role_test_set_2019_author_count3_jsd_l0_ge0p03186_highconf2/`:
  500-paper 2019 role-aware test set sampled from the 2019 selected-paper pool
  with each paper's groundtruth DBLP authors as the target team:
  `test_500.jsonl`, `test_500.tsv`, and `test_500_ids.tsv`.
  Complete template task-node role descriptions are under
  `test_role_descriptions/`; their OpenAI `text-embedding-3-small` normalized
  embeddings are under `test_role_embeddings_template_openai/`.
- `outputs/cross_domain_eval_selection/test_2019_author_count3_jsd_l0_ge0p03186_highconf2_histge5/`:
  2019 role-aware test candidate pool after additionally requiring every
  groundtruth paper author to have at least five pre-2019 DBLP history papers.
- `outputs/role_test_set_2019_author_count3_jsd_l0_ge0p03186_highconf2_histge5/`:
  500-paper 2019 role-aware test set sampled from the hist>=5 candidate pool.
  This is the preferred 2019 test split when expert-history embeddings are
  required for every groundtruth author. Pre-2019 groundtruth-author history
  paper texts are under `test_history_embedding_inputs/`, and their OpenAI
  `text-embedding-3-small` normalized embeddings are under
  `test_history_embeddings_openai/`.
- `outputs/expert_citation_graph_valid2018_pre2018/`: current expert citation
  graph for authors in the `validation_2018_all_authors_hist_ge5.jsonl` subset,
  with edges induced only by DBLP citation relationships among papers with
  `year < 2018`.
- `../data/dblp/expert_id_name.tsv`: optional legacy expert id list. Use it
  only by explicitly passing `--expert-tsv` when a deliberate subset run is
  needed.
- `../data/dblp/FieldsOfStudy.txt`: default legacy FoS id/name map used by
  `scripts/build_cutoff_expert_profiles.py` when no local FoS map exists.

## Canonical Role-Aware Splits

- Current training set:
  `outputs/role_training_splits_author_count3_jsd_l0_ge0p03186_highconf2/train_5000.jsonl`
  with 5,000 papers sampled from the 2018 validation-derived selected pool.
  Use the paired `train_5000.tsv` and `train_5000_ids.tsv` for inspection.
  Template role descriptions are in `train_role_descriptions/`; LLM role
  descriptions are in `train_role_descriptions_llm_gptoss120b/`; pre-2018
  groundtruth-author history text/embedding files are in
  `train_history_paper_embeddings_inputs/`.
- Internal development holdout, not a final temporal test set:
  `outputs/role_training_splits_author_count3_jsd_l0_ge0p03186_highconf2/test_500.jsonl`
  with 500 paper-disjoint papers sampled from the same 2018 selected pool as
  the training set. Use this for quick dev checks or ablations tied to the
  training pool; do not report it as the main 2019 test result.
- Formal 2019 test set, hist>=5 version:
  `outputs/role_test_set_2019_author_count3_jsd_l0_ge0p03186_highconf2_histge5/test_500.jsonl`
  with 500 papers sampled after additionally requiring every groundtruth paper
  author to have at least five pre-2019 DBLP history papers. Prefer this test
  split when the evaluation needs groundtruth-author history embeddings for all
  gold authors. Pre-2019 history texts are in `test_history_embedding_inputs/`;
  OpenAI `text-embedding-3-small` normalized history embeddings are in
  `test_history_embeddings_openai/`. LLM task-node role descriptions and
  OpenAI `text-embedding-3-small` role embeddings are under
  `test_role_descriptions_llm_gptoss120b/`.
- In short: report the 2019 hist>=5 split as the official test set for current
  role-aware experiments. Treat the 2018 `role_training_splits.../test_500`
  file as a development holdout.

## Pipeline Commands

- Stage-1 server handoff:

```bash
cat README_STAGE1_SERVER.md

python scripts/train_stage1.py \
  --epochs 1 \
  --max-train-papers 1 \
  --max-eval-papers 1 \
  --batch-size 1 \
  --device cpu \
  --out-dir outputs/stage1_training/server_preflight
```

Use `README_STAGE1_SERVER.md` as the compact checklist for cloning the KDD
workspace on a server. It lists the code-only git handoff policy, runtime
dependencies, required generated artifacts, one-paper preflight command, and
canonical full Stage-1 training command.

- Build leakage-free temporal expert profiles for validation/test:

```bash
python scripts/build_cutoff_expert_profiles.py \
  --dblp-json data/dblp/dblp.v12.json \
  --out-dir outputs/expert_profile_cutoffs
```

By default this uses every DBLP author id observed in qualifying papers. Pass
`--expert-tsv ../data/dblp/expert_id_name.tsv` only for a deliberate legacy
expert-subset run.

Default outputs:

- `outputs/expert_profile_cutoffs/pre_2018_for_valid_2018/`: expert profiles
  from papers with `2000 <= year < 2018`, for the 2018 validation set.
- `outputs/expert_profile_cutoffs/pre_2019_for_test_2019_2020/`: expert
  profiles from papers with `2000 <= year < 2019`, for the 2019-2020 test set.

Use `--min-year 0` to include all parseable DBLP years before each cutoff, or
repeat `--cutoff LABEL:YEAR` to generate additional strict pre-year profiles.

- Build full temporal validation/test task sets, without any expert-id
  allowlist:

```bash
python scripts/build_temporal_validation_test_sets.py \
  --dblp-json data/dblp/dblp.v12.json \
  --out-dir outputs/temporal_task_splits_full
```

Default outputs:

- `outputs/temporal_task_splits_full/validation_2018.jsonl`: all DBLP records
  with `year == 2018`, preserving all DBLP authors.
- `outputs/temporal_task_splits_full/test_2019_2020.jsonl`: all DBLP records
  with `year in {2019, 2020}`, preserving all DBLP authors.
- `outputs/temporal_task_splits_full/_summary.tsv`: split-level counts for
  records, author edges, unique author ids, records with authors, and records
  with FoS.

- Build the 2018-validation author citation graph from pre-2018 evidence:

```bash
python scripts/build_validation_author_citation_graph.py \
  --dblp-json data/dblp/dblp.v12.json \
  --validation-jsonl outputs/temporal_task_splits_full/validation_2018_all_authors_hist_ge5.jsonl \
  --cutoff-year 2018 \
  --out-dir outputs/expert_citation_graph_valid2018_pre2018
```

Default outputs:

- `outputs/expert_citation_graph_valid2018_pre2018/nodes.tsv`: all author ids
  observed in the filtered 2018 validation subset, plus validation/historical
  paper counts and directed citation in/out weights.
- `outputs/expert_citation_graph_valid2018_pre2018/edges_directed.tsv`: weighted
  directed author citation edges induced by pre-2018 paper citations. Edge
  weights are Cartesian-normalized: one paper citation contributes total weight
  1, split across author pairs as `1 / (source_author_count * target_author_count)`.
- `outputs/expert_citation_graph_valid2018_pre2018/edges_undirected.tsv`:
  reciprocal directed weights collapsed for undirected graph algorithms such as
  Louvain.
- `outputs/expert_citation_graph_valid2018_pre2018/graph.sqlite`: SQLite copy of
  the directed weighted edge table.

- Compute degree-normalized citation proximity for candidate author/team pairs:

```bash
python scripts/compute_validation_author_team_proximity.py \
  --graph-dir outputs/expert_citation_graph_valid2018_pre2018 \
  --validation-jsonl outputs/temporal_task_splits_full/validation_2018_all_authors_hist_ge5.jsonl \
  --candidates-tsv path/to/candidates.tsv \
  --out-tsv outputs/path/to/candidate_team_proximity.tsv \
  --normalization degree
```

- Prepare small Stage-1 pilot datasets from the filtered 2018 validation split:

```bash
python scripts/prepare_stage1_pilot_samples.py
```

Default outputs under `outputs/stage1_pilot_samples/`:

- `smoke_200.jsonl`, `pilot_1000.jsonl`, `dev_5000.jsonl`: augmented task
  records for dataloader/training smoke tests.
- `*_llm_inputs.jsonl`: compact title/abstract/FoS/author metadata files for
  LLM task-node description generation.
- `*_ids.tsv`: quick inspection tables with team size, r=20 community ids,
  coarse stratification block ids, and author history counts.
- `sample_summary.tsv` and `manifest.json`: reproducibility metadata and
  stratification counts.

- Generate KDD-local Stage-1 task-node role-description prompts from the
  smoke/pilot/dev samples. By default this writes prompts only and does not call
  an LLM; add `--generate` after inspecting prompt quality:

```bash
python scripts/generate_stage1_task_node_role_descriptions.py \
  --sample-jsonl outputs/stage1_pilot_samples/smoke_200.jsonl \
  --out-dir outputs/stage1_task_node_role_descriptions_smoke_200
```

The script treats filtered paper FoS labels as preliminary task nodes, keeps FoS
labels with weight at least 0.4, does not apply generic-FoS blacklist filtering,
does not cap nodes per paper by default, then uses the original node-role prompt style from
`../scripts/generate_completed_task_node_role_descriptions.py`: all nodes for
one paper are prompted together with the `ONE FACET PER NODE` and `NODES MUST
DIFFER` constraints.

- Export unique pre-2018 history paper texts for the authors in the Stage-1
  smoke sample:

```bash
python scripts/export_stage1_smoke_history_paper_texts.py
```

Default outputs under `outputs/stage1_smoke_embedding_inputs/`:

- `history_paper_texts.jsonl`: unique pre-2018 history papers with title,
  reconstructed abstract, and embedding text.
- `author_history_papers.tsv`: smoke author to pre-2018 history-paper mapping.
- `summary.tsv`: scan and coverage counts.

- Embed smoke role descriptions and history paper texts with OpenAI-compatible
  parallel embedding script. The script records observed concurrency in
  `--metrics-out`.

```bash
set -a; source ../.env.openai; set +a; python scripts/embed_jsonl_texts_openai_parallel.py \
  --input-jsonl outputs/stage1_task_node_role_descriptions_smoke_200/stage1_task_node_role_descriptions.jsonl \
  --ids-out outputs/stage1_smoke_embeddings/role_description_embedding_ids.tsv \
  --embeddings-out outputs/stage1_smoke_embeddings/role_description_embeddings.npy \
  --metrics-out outputs/stage1_smoke_embeddings/role_description_embedding_metrics.json \
  --model text-embedding-3-small \
  --id-field id \
  --text-field role_description \
  --batch-size 128 \
  --workers 8 \
  --max-in-flight 8 \
  --target-tpm 0 \
  --target-rpm 0 \
  --normalize

set -a; source ../.env.openai; set +a; python scripts/embed_jsonl_texts_openai_parallel.py \
  --input-jsonl outputs/stage1_smoke_embedding_inputs/history_paper_texts.jsonl \
  --ids-out outputs/stage1_smoke_embeddings/history_paper_embedding_ids.tsv \
  --embeddings-out outputs/stage1_smoke_embeddings/history_paper_embeddings.npy \
  --metrics-out outputs/stage1_smoke_embeddings/history_paper_embedding_metrics.json \
  --model text-embedding-3-small \
  --id-field id \
  --text-field text \
  --batch-size 128 \
  --workers 8 \
  --max-in-flight 8 \
  --target-tpm 0 \
  --target-rpm 0 \
  --normalize
```

- Run the minimal Stage-1 smoke trainer. The default V1 command uses frozen
  OpenAI role/history embeddings, computes alpha inside each batch as
  `relu(cosine(role, history))` with per-author top-k hard zeroing, trains only
  two linear projection heads, and writes logs/checkpoints under
  `outputs/stage1_smoke_training/`.

```bash
python scripts/train_stage1_smoke.py \
  --negative-mode v1 \
  --epochs 3 \
  --batch-size 8 \
  --out-dir outputs/stage1_smoke_training/v1
```

Common PU variants:

```bash
python scripts/train_stage1_smoke.py \
  --negative-mode v2 \
  --pi0 0.5 \
  --epochs 2 \
  --batch-size 8 \
  --out-dir outputs/stage1_smoke_training/v2_pi05

python scripts/train_stage1_smoke.py \
  --negative-mode v3 \
  --citation-graph-dir outputs/expert_citation_graph_valid2018_pre2018 \
  --epochs 3 \
  --batch-size 8 \
  --out-dir outputs/stage1_smoke_training/v3

python scripts/train_stage1_smoke.py \
  --negative-mode v4 \
  --citation-graph-dir outputs/expert_citation_graph_valid2018_pre2018 \
  --epochs 3 \
  --batch-size 8 \
  --out-dir outputs/stage1_smoke_training/v4
```

- Select candidate cross-domain evaluation tasks. First use direct FoS label
  counts to find multi-facet papers, then use the gold authors' pre-cutoff
  direct-FoS histories to require that different task labels are covered by
  different authors:

```bash
python scripts/select_cross_domain_eval_candidates.py \
  --validation-jsonl outputs/stage1_pilot_samples/smoke_200.jsonl \
  --out-dir outputs/cross_domain_eval_selection_smoke_200
```

- Build the 5,000-paper training split and 500-paper test split from the
  role-aware selected-paper pool. The split is paper-disjoint, stratified by
  author-count bucket, and uses each paper's groundtruth DBLP authors as the
  target team:

```bash
python scripts/build_role_training_splits.py
```

Default outputs are under
`outputs/role_training_splits_author_count3_jsd_l0_ge0p03186_highconf2/`.

- Export pre-2018 history paper texts for the 5,000-paper train split and the
  500-paper test split groundtruth author sets, then embed them with
  `text-embedding-3-small`:

```bash
python scripts/export_stage1_smoke_history_paper_texts.py \
  --sample-jsonl outputs/role_training_splits_author_count3_jsd_l0_ge0p03186_highconf2/train_5000.jsonl \
  --dblp-json data/dblp/dblp.v12.json \
  --cutoff-year 2018 \
  --out-dir outputs/role_training_splits_author_count3_jsd_l0_ge0p03186_highconf2/train_history_paper_embeddings_inputs \
  --progress-every 500000

python scripts/export_stage1_smoke_history_paper_texts.py \
  --sample-jsonl outputs/role_training_splits_author_count3_jsd_l0_ge0p03186_highconf2/test_500.jsonl \
  --dblp-json data/dblp/dblp.v12.json \
  --cutoff-year 2018 \
  --out-dir outputs/role_training_splits_author_count3_jsd_l0_ge0p03186_highconf2/test_history_paper_embeddings_inputs \
  --progress-every 500000

python scripts/embed_jsonl_texts_openai_parallel.py \
  --input-jsonl outputs/role_training_splits_author_count3_jsd_l0_ge0p03186_highconf2/train_history_paper_embeddings_inputs/history_paper_texts.jsonl \
  --ids-out outputs/role_training_splits_author_count3_jsd_l0_ge0p03186_highconf2/train_history_paper_embeddings_inputs/history_paper_embedding_ids.tsv \
  --embeddings-out outputs/role_training_splits_author_count3_jsd_l0_ge0p03186_highconf2/train_history_paper_embeddings_inputs/history_paper_embeddings.npy \
  --metrics-out outputs/role_training_splits_author_count3_jsd_l0_ge0p03186_highconf2/train_history_paper_embeddings_inputs/history_paper_embedding_metrics.json \
  --model text-embedding-3-small \
  --id-field id \
  --text-field text \
  --batch-size 128 \
  --workers 8 \
  --max-in-flight 8 \
  --target-tpm 4000000 \
  --normalize \
  --resume

python scripts/embed_jsonl_texts_openai_parallel.py \
  --input-jsonl outputs/role_training_splits_author_count3_jsd_l0_ge0p03186_highconf2/test_history_paper_embeddings_inputs/history_paper_texts.jsonl \
  --ids-out outputs/role_training_splits_author_count3_jsd_l0_ge0p03186_highconf2/test_history_paper_embeddings_inputs/history_paper_embedding_ids.tsv \
  --embeddings-out outputs/role_training_splits_author_count3_jsd_l0_ge0p03186_highconf2/test_history_paper_embeddings_inputs/history_paper_embeddings.npy \
  --metrics-out outputs/role_training_splits_author_count3_jsd_l0_ge0p03186_highconf2/test_history_paper_embeddings_inputs/history_paper_embedding_metrics.json \
  --model text-embedding-3-small \
  --id-field id \
  --text-field text \
  --batch-size 128 \
  --workers 8 \
  --max-in-flight 8 \
  --normalize
```

- Run formal Stage-1 training on the canonical role-aware splits. This trains
  on `train_5000`, evaluates on the 2018 internal dev holdout, and evaluates on
  the official 2019 hist>=5 test split:

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

The Stage-1 implementation now matches the task-node/expert-node objective:
role-side vectors are frozen role-description embeddings, while expert-side
vectors are node-conditioned alpha-weighted means of frozen history-paper
embeddings. By default, alpha comes from each history paper's direct FoS
weights expanded to the minimal-tree connector nodes used by the untrained
Top-M retrieval experiments (`--node-link-mode minimal_tree
--node-weight-mode unweighted`), with hard-zero masking for unrelated papers;
it is not computed from role/history embedding cosine. Use
`scripts/export_history_paper_fos_weights.py` to create
`history_paper_fos_weights.tsv` for each split's existing
`history_paper_embedding_ids.tsv`.

The minimal-tree v3-retrieval role-description text files exist under
`outputs/expanded_fos_role_descriptions_minimal_tree_*_v3_retrieval_gpt5mini/`,
and the dev/formal-test role embedding caches are complete for the current
JSONL files. Train minimal-tree v3 role embeddings have not been generated in
this workspace note, so keep canonical train paths unless the train embedding
cache is created or explicitly override train/dev/official-test role paths
together.

Stage-1 training now defaults to frozen-retrieval hard negatives rather than
batch-only or random negatives. For each task, the denominator includes all
gold authors plus the union of the per-node frozen role/expert-node top-M
non-gold authors selected by `--untrained-negative-top-m`; use
`--negative-pool-mode batch` only to reproduce older in-batch-negative runs.
Use `--negative-pool-mode mixed` to add sampled random pool negatives on top of
the frozen top-M hard negatives.

The first task-node/expert-node V1 run with both validation splits wrote
`outputs/stage1_training/task_expert_node_dev_and_test_v1`: 4,467 train tasks,
500 dev tasks, 500 official-test tasks, 31,339 train nodes used with 7,316
skipped for empty gold alpha support, 3,474 dev nodes used with 828 skipped,
and 3,391 official-test nodes used with 738 skipped. Epoch-10 top1 was train
0.929289, dev 0.832182, and official-test 0.837806. Best logged dev top1 was
epoch 8 at 0.835636; best logged official-test top1 was epoch 7 at 0.847243.
This run used the earlier batch-only negative pool and should be treated as the
pre-global-negative baseline.

The older `outputs/stage1_training/dev_as_test_v1` run used role/history
cosine-derived alpha and should be treated only as a superseded smoke result,
not as the Stage-1 objective result.

- Evaluate the trained Stage-1 projection model by full split-pool node
  assignment. For each split, the candidate pool is all unique groundtruth
  authors appearing in that split; each task node is assigned to the highest
  scoring candidate expert with nonzero alpha support for that node, then a
  paper-level deduplicated predicted team is compared against groundtruth
  authors:

```bash
python scripts/evaluate_stage1_full_node_assignment.py \
  --out-dir outputs/stage1_full_node_assignment/task_expert_node_v1
```

The first full split-pool assignment run wrote
`outputs/stage1_full_node_assignment/task_expert_node_v1`. Dev used 1,888
candidate authors, assigned 4,275 nodes, skipped 27 nodes with no valid
candidate, and reported raw node-assignment precision 0.207018, dedup macro
P/R/F1 0.176054/0.320160/0.218391, and dedup micro P/R/F1
0.159420/0.311695/0.210948. Official test used 1,898 candidate authors,
assigned 4,101 nodes, skipped 28 nodes, and reported raw node-assignment
precision 0.205803, dedup macro P/R/F1 0.181878/0.300801/0.216325, and dedup
micro P/R/F1 0.156603/0.291133/0.203657.

The independent argmax node assignment above allows multiple nodes to collapse
onto the same expert before deduplication. A corrected one-expert-per-node
Hungarian assignment run wrote
`outputs/stage1_full_node_assignment/task_expert_node_v1_hungarian`, using
`--assignment-mode hungarian`. Dev assigned 4,275 nodes, skipped 27, and reported
raw node precision 0.172164, dedup macro P/R/F1 0.178263/0.387684/0.237069,
and dedup micro P/R/F1 0.172164/0.379186/0.236808. Official test assigned
4,100 nodes, skipped 29, and reported raw node precision 0.171220, dedup macro
P/R/F1 0.181031/0.369756/0.234750, and dedup micro P/R/F1
0.171220/0.359815/0.232028. Hungarian removes duplicate expert collapse and
raises recall versus independent argmax, but still trails the whole-paper to
whole-expert mean baseline.

Adding one virtual root row to the same Hungarian assignment, with root query
equal to the target paper title/abstract embedding and root expert vectors equal
to each candidate author's mean history-paper embedding, wrote
`outputs/stage1_full_node_assignment/task_expert_node_v1_hungarian_with_root`
using `--assignment-mode hungarian --include-root-node`. Dev assigned 4,775
nodes including roots, skipped 27, and reported raw precision 0.175916, dedup
macro P/R/F1 0.181703/0.441545/0.250720, and dedup micro P/R/F1
0.175916/0.432767/0.250149. Official test assigned 4,600 nodes including
roots, skipped 29, and reported raw precision 0.175435, dedup macro P/R/F1
0.183845/0.424596/0.249382, and dedup micro P/R/F1
0.175435/0.413634/0.246375. The root row improves recall and F1 over
node-only Hungarian, but still trails the whole-paper to whole-expert mean
baseline.

- Whole-paper to whole-expert mean-embedding search baseline. Query embeddings
  are OpenAI `text-embedding-3-small` embeddings of each target paper's
  title/abstract; candidate expert embeddings are per-split mean normalized
  history-paper embeddings for all unique authors in that split. Retrieve
  `top_k = task node count` experts per paper and compare against groundtruth
  authors:

```bash
python scripts/export_stage1_split_paper_texts.py \
  --sample-jsonl outputs/role_training_splits_author_count3_jsd_l0_ge0p03186_highconf2/test_500.jsonl \
  --out-jsonl outputs/paper_to_expert_mean_embedding/dev_paper_texts.jsonl

python scripts/export_stage1_split_paper_texts.py \
  --sample-jsonl outputs/role_test_set_2019_author_count3_jsd_l0_ge0p03186_highconf2_histge5/test_500.jsonl \
  --out-jsonl outputs/paper_to_expert_mean_embedding/test_paper_texts.jsonl

python scripts/evaluate_paper_to_expert_mean_embedding_search.py ...
```

The first run wrote `outputs/paper_to_expert_mean_embedding/`. Dev used 1,888
candidate authors, mean top-k 8.604, and reported macro P/R/F1
0.256949/0.547680/0.338530 and micro P/R/F1
0.245235/0.543534/0.337979. Official test used 1,898 candidate authors, mean
top-k 8.258, and reported macro P/R/F1 0.290540/0.575988/0.371831 and micro
P/R/F1 0.267861/0.566889/0.363816.

Formal train/dev/official-test artifacts are now present. The official hist>=5
test role-description file has 4,129 role rows for 500 papers; Together
`openai/gpt-oss-120b` generated 499 papers, while paper `2991104668` used the
script's template backend fallback after repeated invalid JSON escapes from
math/code text. The paired official-test role embedding array has shape
`(4129, 1536)` with normalized OpenAI `text-embedding-3-small` vectors.

- Filter the 2019 role-aware candidate pool to papers where every groundtruth
  author has at least five pre-2019 history papers, then sample the strict
  500-paper test set:

```bash
python scripts/filter_selected_papers_by_author_history_count.py \
  --input-jsonl outputs/cross_domain_eval_selection/test_2019_author_count3_jsd_l0_ge0p03186_highconf2/selected_papers.jsonl \
  --features-tsv outputs/cross_domain_eval_selection/test_2019_author_count3_jsd_l0_ge0p03186_highconf2/selected_papers.tsv \
  --cutoff-year 2019 \
  --min-history-papers 5 \
  --out-dir outputs/cross_domain_eval_selection/test_2019_author_count3_jsd_l0_ge0p03186_highconf2_histge5 \
  --progress-every 1000000

python scripts/build_role_training_splits.py \
  --input-jsonl outputs/cross_domain_eval_selection/test_2019_author_count3_jsd_l0_ge0p03186_highconf2_histge5/selected_papers.jsonl \
  --features-tsv outputs/cross_domain_eval_selection/test_2019_author_count3_jsd_l0_ge0p03186_highconf2_histge5/selected_papers.tsv \
  --out-dir outputs/role_test_set_2019_author_count3_jsd_l0_ge0p03186_highconf2_histge5 \
  --train-size 0 \
  --test-size 500 \
  --seed 13
```

- Generate or finish the missing LLM role artifacts needed by formal Stage-1
  training. The role embeddings must use the same `text-embedding-3-small`
  space as the cached history embeddings:

```bash
python scripts/embed_jsonl_texts_openai_parallel.py \
  --input-jsonl outputs/role_training_splits_author_count3_jsd_l0_ge0p03186_highconf2/train_role_descriptions_llm_gptoss120b/stage1_task_node_role_descriptions.jsonl \
  --ids-out outputs/role_training_splits_author_count3_jsd_l0_ge0p03186_highconf2/train_role_descriptions_llm_gptoss120b/role_description_embedding_ids.tsv \
  --embeddings-out outputs/role_training_splits_author_count3_jsd_l0_ge0p03186_highconf2/train_role_descriptions_llm_gptoss120b/role_description_embeddings.npy \
  --metrics-out outputs/role_training_splits_author_count3_jsd_l0_ge0p03186_highconf2/train_role_descriptions_llm_gptoss120b/role_description_embedding_metrics.json \
  --model text-embedding-3-small \
  --id-field id \
  --text-field role_description \
  --batch-size 128 \
  --workers 8 \
  --max-in-flight 8 \
  --target-tpm 4000000 \
  --normalize \
  --resume

python scripts/generate_stage1_task_node_role_descriptions.py \
  --sample-jsonl outputs/role_test_set_2019_author_count3_jsd_l0_ge0p03186_highconf2_histge5/test_500.jsonl \
  --out-dir outputs/role_test_set_2019_author_count3_jsd_l0_ge0p03186_highconf2_histge5/test_role_descriptions_llm_gptoss120b \
  --generate \
  --backend together \
  --model openai/gpt-oss-120b \
  --temperature 0 \
  --max-tokens 2048 \
  --workers 8 \
  --progress-every 50

python scripts/embed_jsonl_texts_openai_parallel.py \
  --input-jsonl outputs/role_test_set_2019_author_count3_jsd_l0_ge0p03186_highconf2_histge5/test_role_descriptions_llm_gptoss120b/stage1_task_node_role_descriptions.jsonl \
  --ids-out outputs/role_test_set_2019_author_count3_jsd_l0_ge0p03186_highconf2_histge5/test_role_descriptions_llm_gptoss120b/role_description_embedding_ids.tsv \
  --embeddings-out outputs/role_test_set_2019_author_count3_jsd_l0_ge0p03186_highconf2_histge5/test_role_descriptions_llm_gptoss120b/role_description_embeddings.npy \
  --metrics-out outputs/role_test_set_2019_author_count3_jsd_l0_ge0p03186_highconf2_histge5/test_role_descriptions_llm_gptoss120b/role_description_embedding_metrics.json \
  --model text-embedding-3-small \
  --id-field id \
  --text-field role_description \
  --batch-size 128 \
  --workers 8 \
  --max-in-flight 8 \
  --target-tpm 4000000 \
  --normalize \
  --resume
```

## Important Decisions

- Validation tasks are year 2018 and must use expert evidence strictly before
  2018.
- Stage-1 should train against the split-level author pool, not only authors
  that co-occur in the same mini-batch. The current default is
  `--node-link-mode minimal_tree --node-weight-mode unweighted
  --negative-pool-mode untrained_topm --untrained-negative-top-m 20`, which
  first retrieves high-scoring non-gold experts with frozen minimal-tree
  role/expert-node embeddings and then trains the projection heads to rank gold
  authors above those hard candidates. Older `task_expert_node_dev_and_test_v1`
  metrics used batch-only direct-node negatives and are now a baseline rather
  than the preferred training setup.
- The full-validation filter `author_count >= 3 AND author_jsd_l0_mean >= 0.019`
  over `validation_2018_all_authors_hist_ge5.jsonl` wrote
  `outputs/cross_domain_eval_selection/author_jsd_l0_ge0p019_author_ge3/`.
  It keeps 26,852 of 60,680 papers (44.25%), with 103,836 author appearances
  and 53,490 unique authors. A manual full-FoS audit sample of 100 papers from
  this filtered set is in
  `outputs/cross_domain_eval_selection/author_jsd_l0_ge0p019_author_ge3/sample100_manual_cross_domain_annotations.tsv`
  and had label counts: 17 yes, 22 borderline, 61 no. This filter is therefore
  useful as a broad candidate pool but too noisy to serve directly as a clean
  cross-domain evaluation set.
- A separate random 100-paper audit was sampled directly from the full
  `validation_2018_all_authors_hist_ge5.jsonl` pool, without `author_jsd`,
  `direct_l2`, or `author_count>=3` filtering, using
  `scripts/sample_full_fos_cross_domain_check.py --min-direct-l2 0 --min-authors 1 --sample-size 100 --seed 73`.
  Outputs are under
  `outputs/cross_domain_eval_selection/whole_validation_hist_ge5_sample100/`.
  Manual role-based-vs-common-interest annotations are generated by
  `scripts/write_whole_validation_sample100_role_annotations.py` and written to
  `sample100_role_based_collaboration_annotations.tsv`: 11 `role_based`, 11
  `borderline`, and 78 `common_interest`. The sample contains 24 single-author
  papers and 76 multi-author papers. Treat this as the unfiltered 2018
  hist>=5 validation baseline, not as a filtered cross-domain candidate set.
  Filter-method comparison on this sample uses
  `scripts/build_sample100_author_jsd_features.py` plus
  `scripts/compare_whole_validation_sample100_filters.py`; outputs are
  `sample100_author_jsd_all_levels.tsv`, `filter_method_comparison.tsv`, and
  `filter_method_comparison_top.tsv` in the same directory. With strict
  `role_based` as positive, the all-negative baseline has 0.890 accuracy but
  F1 0, so accuracy alone is misleading. The best F1 rule in the grid was
  `author_jsd_direct_mean>=0.41697 AND author_jsd_l1_mean>=0.167751`, selecting
  15 papers with precision 0.400, recall 0.545, F1 0.462, and accuracy 0.860.
  The best balanced-accuracy rule was
  `author_jsd_l0_mean>=0.0364052 AND author_jsd_l0_min>=0.0109167`, selecting
  36 papers with precision 0.278, recall 0.909, F1 0.426, balanced accuracy
  0.808, and accuracy 0.730. The previously considered `direct_l2_count>=3`
  is poor on this unfiltered sample: strict precision 0.033, recall 0.091,
  F1 0.049, accuracy 0.610.
  A constrained three-part rule search that always requires `author_count>=3`
  and then adds one author-JSD condition plus one taxonomy-node-distribution
  condition is implemented in
  `scripts/compare_author_count3_jsd_taxnode_filters.py`; outputs are
  `filter_author_count3_jsd_taxnode_grid.tsv` and
  `filter_author_count3_jsd_taxnode_top.tsv`. On this 100-paper sample, the
  best strict-role F1 rule was
  `author_count>=3 AND author_jsd_l1_mean>=0.17491 AND tax_direct_any_count>=2`
  with selected=9, TP=5, FP=4, precision 0.556, recall 0.455, F1 0.500, and
  accuracy 0.900. The best relaxed role-or-borderline F1 rule was
  `author_count>=3 AND author_jsd_l0_mean>=0.0318615 AND tax_direct_l2_count>=1`
  with selected=25, TP=13, FP=12, precision 0.520, recall 0.591, F1 0.553,
  and accuracy 0.790. `direct_l2_count>=3` remains too strict for this
  purpose; prefer `direct_l2_count>=1` or `direct_any_count>=2` as the
  taxonomy-node condition when author-history JSD is also present.
  Applying the more interpretable candidate rule
  `author_count>=3 AND author_jsd_l0_mean>=0.03186 AND high_conf_direct_node_count>=2`
  to the full 60,680-paper 2018 validation set, where
  `high_conf_direct_node_count` counts direct FoS nodes with weight >= 0.5,
  wrote
  `outputs/cross_domain_eval_selection/author_count3_jsd_l0_ge0p03186_highconf2/`.
  It keeps 13,456 papers (22.18%), with 52,289 author appearances and 33,129
  unique authors. Mean authors per paper increases from 2.782 before filtering
  to 3.886 after filtering.
  Treat all 13,456 selected papers as the current role-aware training set.
  The author-count distribution is: 3 authors = 6,350 papers, 4 authors =
  4,062, 5 authors = 1,935, 6 authors = 731, 7 authors = 233, and 8+ authors =
  145. Do not use this filtered set as a clean evaluation set; keep held-out
  evaluation on later temporal test tasks such as the 2020plus subset.
  The sampled 5,000/500 paper-disjoint train/test split under
  `outputs/role_training_splits_author_count3_jsd_l0_ge0p03186_highconf2/`
  uses groundtruth paper authors as teams. Its pre-2018 groundtruth-author
  history export covers every sampled author: train has 15,745 authors,
  666,298 unique history papers, and 873,982 author-history edges; test has
  1,888 authors, 108,464 unique history papers, and 124,483 edges. The
  normalized OpenAI `text-embedding-3-small` history embeddings have shapes
  `(666298, 1536)` for train and `(108464, 1536)` for test, with zero missing
  rows.
  Applying the same interpretable rule to 2019 papers with pre-2019 author
  histories wrote
  `outputs/cross_domain_eval_selection/test_2019_author_count3_jsd_l0_ge0p03186_highconf2/`.
  It keeps 88,245 of 296,890 papers (29.72%), with 410,246 author appearances
  and 257,401 unique authors. Mean authors per paper is 4.649 after filtering.
  A stratified 500-paper test set sampled with seed 13 wrote
  `outputs/role_test_set_2019_author_count3_jsd_l0_ge0p03186_highconf2/`.
  The test set has 2,310 author appearances, 2,298 unique authors, mean team
  size 4.620, and author-count buckets: 3 authors = 148 papers, 4 authors =
  139, 5 authors = 98, 6 authors = 60, 7 authors = 26, and 8+ authors = 29.
  The target team for every task is exactly the paper's groundtruth DBLP author
  list.
  The complete template role-description run produced 4,118 node descriptions
  for all 500 papers with no duplicate ids. The embedding run used
  `text-embedding-3-small`, wrote a `(4118, 1536)` float32 matrix, and normalized
  rows to unit length. A Together `openai/gpt-oss-120b` LLM role-description
  attempt is partially present under `test_role_descriptions_llm_gptoss120b/`,
  but Together returned `402 credit_limit`; rerun that command after credits are
  available before using the LLM file as a complete source.
  The original 2019 `test_500` did not require every author to have
  `history_paper_count >= 5`; only 1,807 of 2,298 sampled groundtruth authors
  had pre-2019 history text. The strict hist>=5 rerun wrote
  `outputs/cross_domain_eval_selection/test_2019_author_count3_jsd_l0_ge0p03186_highconf2_histge5/`
  and kept 9,773 of 88,245 candidate papers (11.07%). Its sampled test set
  under
  `outputs/role_test_set_2019_author_count3_jsd_l0_ge0p03186_highconf2_histge5/`
  has 500 papers, 1,951 author appearances, 1,898 unique authors, mean team size
  3.902, and every sampled groundtruth author has at least five pre-2019 DBLP
  history papers. The strict test set history export found all 1,898
  groundtruth authors, 109,028 unique pre-2019 history papers with text, and
  125,744 author-history-paper edges. The history embedding run used
  `text-embedding-3-small`, wrote
  `test_history_embeddings_openai/history_paper_embeddings.npy` with shape
  `(109028, 1536)`, and normalized every row to unit length. The ids file
  order matches `test_history_embedding_inputs/history_paper_texts.jsonl`.
  A random 100-paper audit sample from this filtered set, using seed 97, wrote
  `sample100_audit.jsonl`, `sample100_compact.tsv`, and
  `sample100_manual_role_annotations.tsv` in the same directory. Manual
  inspection labeled 34 `role_based`, 23 `borderline`, and 43
  `common_interest`; therefore this rule is useful as a role-based
  collaboration candidate pool, but not as a fully clean evaluation set without
  additional review or stricter filtering.
  Re-labeling the same sample by the Stage-1 training criterion "does the paper
  have separable role/facet supervision, including within-domain technical
  division of labor" wrote `sample100_role_internal_annotations.tsv`: 36
  `role_based_clear`, 54 `role_based_internal`, and 10
  `weak_or_common_interest`, so 90/100 are usable as role-aware Stage-1
  candidates. Keep this criterion separate from a stricter cross-domain
  evaluation criterion.
- For that same 100-paper manual audit sample, direct/L0/L1/L2 author-history
  JSD features were recomputed in
  `outputs/cross_domain_eval_selection/author_jsd_l0_ge0p019_author_ge3/sample100_author_jsd_all_levels.tsv`,
  and threshold grid results are in
  `outputs/cross_domain_eval_selection/author_jsd_l0_ge0p019_author_ge3/sample100_author_jsd_threshold_grid.tsv`.
  Strict-yes best JSD-only rule was `author_jsd_l0_mean>=0.0603261`: selected
  16 papers, precision 0.562, recall 0.529, F1 0.545, balanced accuracy 0.723.
  The next useful strict-yes rule was `author_jsd_l1_mean>=0.161422`: selected
  24, precision 0.417, recall 0.588, F1 0.488, balanced accuracy 0.710.
  For yes-or-borderline, best F1 was `author_jsd_l2_mean>=0.388193`: selected
  51, precision 0.569, recall 0.744, F1 0.644, balanced accuracy 0.691. If
  precision is preferred for yes-or-borderline, `author_jsd_l0_mean>=0.0524481`
  selected 24 with precision 0.792, recall 0.487, F1 0.603, balanced accuracy
  0.703.
- Adding `direct_l2_count>=3` to the same 100-paper audit sample performed
  poorly. Results are in
  `outputs/cross_domain_eval_selection/author_jsd_l0_ge0p019_author_ge3/sample100_direct_l2_ge3_author_jsd_composition_grid.tsv`.
  The base `direct_l2_count>=3` selected 33 papers but only 3 strict-yes
  examples (precision 0.091, recall 0.176, F1 0.120). The best strict-yes
  composition was only
  `direct_l2_count>=3 AND author_jsd_l1_min>=0.100606`, selecting 6 papers
  with precision 0.333, recall 0.118, F1 0.174. For yes-or-borderline, the
  best composition reached only F1 0.277. Do not use direct-L2 count as a hard
  AND filter for this candidate set; it discards many manually identified
  cross-domain examples and keeps many same-community technical papers.
- Smoke-200 cross-domain classifier comparison is reproducible with
  `python scripts/build_smoke200_cross_domain_all_features.py` followed by
  `python scripts/compare_cross_domain_classifiers_smoke.py`. The first script
  writes
  `outputs/cross_domain_eval_selection_smoke_200/smoke_200_cross_domain_all_features.tsv`
  with FoS projected branch counts plus author-history JSD at direct/L0/L1/L2
  levels; the second writes
  `outputs/cross_domain_eval_selection_smoke_200/classifier_method_comparison_smoke_200.tsv`
  and
  `outputs/cross_domain_eval_selection_smoke_200/classifier_family_summary_smoke_200.tsv`.
  Against the full manual smoke labels (45 yes, 52 borderline, 103 no), strict
  yes detection is class-imbalanced: all-no already gives 0.775 accuracy, so
  use F1/balanced accuracy instead of raw accuracy. Best single-feature strict
  yes rules by F1/balanced accuracy were `r20_block_count>=4` (F1 0.432,
  balanced accuracy 0.627), `direct_l2_count>=3` (F1 0.425, balanced accuracy
  0.623), `level1_projected_count>=8` (F1 0.424, balanced accuracy 0.621),
  `author_jsd_l0_mean>=0.0260786` (F1 0.418, balanced accuracy 0.605), and
  `author_jsd_l1_mean>=0.0877824` (F1 0.404, balanced accuracy 0.585).
  `author_jsd_direct_mean>=0.29342` has high strict-yes recall (0.867) but low
  precision (0.248), so projected L0/L1 JSD is preferable to direct-node JSD as
  an author-background signal. For relaxed yes-or-borderline, F1-optimal
  thresholds tend to overselect, so use balanced accuracy: `level1_projected_count>=8`
  was strongest (balanced accuracy 0.668, precision 0.678, recall 0.608),
  followed by `author_jsd_l2_min<=0.268617` (balanced accuracy 0.617).
- Smoke-200 composite AND-rule grid was written to
  `outputs/cross_domain_eval_selection_smoke_200/composite_and_rule_grid_smoke_200.tsv`.
  For strict yes, the best F1/balanced-accuracy composite was
  `r20_block_count>=4 AND author_jsd_l1_mean>=0.0877824` (selected 69, precision
  0.406, recall 0.622, F1 0.491, balanced accuracy 0.679). The best more
  interpretable FoS+JSD composites were
  `level1_projected_count>=9 AND author_jsd_l1_mean>=0.0894819` (selected 43,
  precision 0.488, recall 0.467, F1 0.477, balanced accuracy 0.662) and
  `direct_l2_count>=3 AND author_jsd_l1_mean>=0.0859385` (selected 61,
  precision 0.410, recall 0.556, F1 0.472, balanced accuracy 0.662). For
  relaxed yes-or-borderline, the best balanced-accuracy composite was
  `level1_projected_count>=8 AND author_jsd_l1_mean>=0.0522062` (selected 84,
  precision 0.702, recall 0.608, F1 0.652, balanced accuracy 0.683). Fixed
  intuitive rule `level1_projected_count>=8 AND author_jsd_l1_mean>=0.0878`
  had relaxed precision 0.730, recall 0.474, F1 0.575, balanced accuracy 0.655.
- Test tasks are years 2019-2020 and must use expert evidence strictly before
  2019.
- Dev and official-test history paper FoS weight tables were exported with
  `scripts/export_history_paper_fos_weights.py`. Dev matched all 108,464
  history papers and wrote 996,811 FoS rows; official hist>=5 test matched all
  109,028 history papers and wrote 984,836 FoS rows. A canonical Stage-1
  one-paper preflight using `scripts/train_stage1.py --epochs 1
  --max-train-papers 1 --max-eval-papers 1 --batch-size 1 --device cpu`
  successfully loaded train, dev, and official test paths and completed one
  epoch.
- Temporal validation/test task sets should be built from DBLP by year and must
  not depend on `expert_id_name.tsv`; that file is only an optional legacy
  candidate-pool subset.
- The 2018 validation expert citation graph defines experts as all authors in
  the filtered 2018 validation subset where every paper author has at least
  five pre-2018 historical papers. Its citation evidence is leakage-free:
  author-author edges use only paper citation links where the citing paper and
  cited paper are both before 2018. Self-loops are skipped by default. To avoid
  inflating large-team citations, each paper citation contributes fractional
  author-pair edge weight `1 / (n_source_authors * n_target_authors)`.
- Citation proximity should use degree normalization: raw proximity from a
  candidate author to a target team is the sum of citation edge weights to that
  team's authors, then `src/citation_proximity.py` divides by the candidate
  author's weighted degree by default.
- `scripts/partition_validation_author_citation_blocks.py` can select
  `best_resolution.tsv` by `--best-metric standard_modularity_gamma1`. On the
  Cartesian-normalized five-history-paper filtered 2018 subset, the refined Louvain grid
  `0.6,0.7,0.8,0.9,1,1.1,1.25,1.5,1.75,2` selected resolution `1.1`, with
  `standard_modularity_gamma1 = 0.570355290231`, 157 blocks, and 124 singleton
  blocks.
- For interpretable stage-1 citation communities, prefer the finer
  `outputs/validation_author_citation_louvain_blocks/membership_resolution_20.tsv`
  over the modularity-best `resolution=1.1` membership. The r=20 graph has 724
  communities, largest community 1,387 authors, median size 45, and much less
  severe mega-block concentration. Community labels generated from pre-2018
  history are under `outputs/interpretable_citation_communities_r20/`,
  especially `community_labels_area_v3.tsv`. Labels are heuristic, using venue
  patterns plus de-noised FoS/top-author evidence; inspect them before treating
  them as hard semantic classes.
- First-round Stage-1 training samples are selected from
  `validation_2018_all_authors_hist_ge5.jsonl` with 2-6 authors, reconstructable
  abstract, non-empty FoS, author history counts, and citation memberships. The
  default sampler stratifies by author count and coarse r=1.1 citation-block
  dispersion, while retaining r=20 interpretable community ids in each sample.
  Current default run produced 41,680 eligible papers and wrote 200/1,000/5,000
  paper samples to `outputs/stage1_pilot_samples/`.
- The first smoke task-node prompt generation run wrote
  `outputs/stage1_task_node_role_descriptions_smoke_200/stage1_task_node_role_prompts.jsonl`
  for 200 papers and 1,718 FoS-derived nodes using the original node-role prompt
  wording, `min_fos_weight=0.4`, no generic-FoS filtering, and no per-paper node cap. This is suitable for
  an engineering smoke test, but the FoS-derived node set still contains some
  medium-broad or noisy labels, so do not treat it as final high-quality task
  decomposition without additional filtering or manual/LLM QC.
- The first smoke LLM generation used Together `openai/gpt-oss-120b` with
  `--workers 100` and wrote
  `outputs/stage1_task_node_role_descriptions_smoke_200/stage1_task_node_role_descriptions.jsonl`.
  Final validation: 200 papers, 1,718 expected FoS-derived nodes, 1,718
  description rows, 200 raw per-paper LLM responses, no missing/extra/duplicate
  `(paper_id, node_id)` outputs. The script includes a conservative node-id
  repair for minor LLM spelling/diacritic changes, accepting only a unique
  high-similarity match to an expected node id.
- The first smoke embedding run used OpenAI `text-embedding-3-small` for both
  task-node role descriptions and pre-2018 history papers. It wrote
  `outputs/stage1_smoke_embeddings/role_description_embeddings.npy` with shape
  `(1718, 1536)` and
  `outputs/stage1_smoke_embeddings/history_paper_embeddings.npy` with shape
  `(45182, 1536)`. Both runs used normalized vectors; sampled L2 norms were
  approximately 1.0. The history text export scanned 4,894,081 DBLP records,
  covered all 706 smoke authors, and found 45,182 unique history papers with
  51,657 author-history edges. With `--workers 8 --max-in-flight 8`, both role
  and history embedding runs observed `max_active_requests_observed=8` and
  `max_pending_futures_observed=8`.
- `src/stage1_smoke_training.py` and `scripts/train_stage1_smoke.py` implement
  the first minimal Stage-1 objective. The trainer computes alpha dynamically
  per batch from frozen role/history embeddings, forms
  `z_{e|i}` by alpha-weighted averaging over each author's pre-2018 history,
  trains only `g_phi` and `g_psi`, supports V1-V4 negative weights, and now
  defaults to minimal-tree expert-node construction plus frozen top-M retrieval
  hard negatives instead of direct-node in-batch-only negatives.
  V3/V4
  scan `edges_undirected.tsv` once but keep only the relevant smoke-author
  subgraph plus relevant authors' full weighted degree, avoiding loading the
  whole 481MB graph into a dense Python adjacency. V3 automatic thresholding
  estimates the requested quantile over positive proximity values only; if no
  positive proximity exists it uses `inf`, so zero-proximity negatives are not
  accidentally all protected.
- First verified smoke training runs: V1 on all 200 tasks for 3 epochs wrote
  `outputs/stage1_smoke_training/v1`, used 1,553 train nodes and 165 eval
  nodes with zero skipped-positive nodes, and ended at train loss 0.273405 /
  eval loss 0.501815 / eval top1-gold accuracy 0.745455. V2 with `pi0=0.5`
  for 2 epochs wrote `outputs/stage1_smoke_training/v2_pi05` and ended at
  eval loss 0.320082 / eval top1-gold accuracy 0.709091. V3 and V4 citation
  proximity paths were debug-verified on small `--max-papers` runs.
- Cross-domain eval filtering should be calibrated on small samples before full
  validation selection. On `smoke_200`, the first FoS-count rule
  `min_fos_weight=0.5`, `author_count>=2`, and
  `direct_l2_count>=5 OR direct_l3_count>=3 OR direct_l4_count>=2` selected 44
  candidate papers. Adding history-dispersion with
  `covered_label_count>=3`, `coverage_frac>=0.35`,
  `distinct_cover_authors>=2`, and `top_author_label_share<=0.75` kept 30
  papers. Manual inspection suggested this was still slightly loose, so the
  preferred high-precision smoke rule is `coverage_frac>=0.5` and
  `top_author_label_share<=2/3`, keeping 23 papers at
  `outputs/cross_domain_eval_selection_smoke_200/selected_high_precision_dispersion.tsv`.
  A very strict variant requiring the stronger FoS-count rule plus dispersion
  kept 10 papers at
  `outputs/cross_domain_eval_selection_smoke_200/selected_fos_strict_and_dispersion.tsv`.
- Manual smoke cross-domain collaboration annotations are generated by
  `scripts/write_smoke200_cross_domain_annotations.py` and written to
  `outputs/cross_domain_eval_selection_smoke_200/smoke_200_manual_cross_domain_annotations.tsv`.
  The annotation uses three labels: `yes` for clear cross-domain collaboration,
  `borderline` for multi-facet but mostly same broad area, and `no` for
  single-domain/same-method/insufficient evidence. Current smoke_200 counts are
  45 `yes`, 52 `borderline`, and 103 `no`; only `yes` rows have
  `use_for_cross_domain_eval=1`.
- `scripts/evaluate_untrained_taxonomy_aggregated_expert_nodes.py` evaluates a
  no-training role-to-expert-node baseline where each expert-node vector is the
  unweighted mean of linked history-paper embeddings. In `direct` mode, papers
  link only to direct FoS labels; in `ancestor` mode, each direct FoS links
  through all `hasParent` ancestors until no parent remains. The default also
  adds a virtual `__root__` assignment row whose task query is the whole-paper
  embedding and whose expert vector is the mean of all that expert's history
  paper embeddings. On the 500-paper dev/test split-pool Hungarian evaluation,
  direct+root wrote
  `outputs/untrained_taxonomy_aggregated_expert_nodes/direct_unweighted_with_root`
  and scored dev micro P/R/F1 21.947644/53.992787/31.209053 and official-test
  24.173913/56.996412/33.949015. Full ancestor-to-root+root wrote
  `outputs/untrained_taxonomy_aggregated_expert_nodes/ancestor_unweighted_full_to_root_with_root`
  and scored dev 22.094241/54.353426/31.417510 and official-test
  24.086957/56.791389/33.826897. Thus the upward aggregation is essentially
  tied with the direct baseline and does not improve the official 2019 test in
  this exact-match setup.
- The same untrained direct+root baseline was rerun with node-side history
  papers averaged by FoS `weight` from `history_paper_fos_weights.tsv`; root
  remains the unweighted mean over all expert history papers. The weighted run
  wrote
  `outputs/untrained_taxonomy_aggregated_expert_nodes/direct_weighted_with_root`
  and scored dev micro P/R/F1 21.905759/53.889748/31.149494 and official-test
  24.021739/56.637622/33.735308. This is slightly worse than unweighted
  direct+root on both splits, so current FoS weights should not be used for the
  no-training direct+root baseline.
- `scripts/generate_stage1_task_node_role_descriptions.py` now supports
  expanded FoS role generation without changing the default direct-node
  behavior: pass `--node-scope expanded` to include LLM root roles and expanded
  FoS nodes. `--expanded-node-policy all_ancestors` preserves the earlier full
  direct+all-ancestor closure, while `--expanded-node-policy minimal_tree`
  keeps all direct FoS nodes and adds only connector ancestors selected to form
  a compact paper-local taxonomy tree/forest under `__root__`. The smoke
  all-ancestor run on the first 3 dev papers wrote
  `outputs/expanded_fos_role_descriptions_llm_root_dev_smoke3_llm_gptoss120b_promptv2`
  with 79 roles: 3 root, 26 direct, and 50 ancestor roles. Many full-closure
  ancestor roles were unsupported by abstract evidence. The minimal-tree smoke
  run wrote
  `outputs/expanded_fos_role_descriptions_llm_root_dev_smoke3_minimal_tree_v2_llm_gptoss120b`
  with 41 roles: 3 root, 26 direct, and 12 connector roles; connector empty
  evidence dropped from 35/50 to 3/12, and obvious risky connectors dropped from
  10 to 2 on this small sample. Remaining risky connectors include taxonomy
  edges such as `Computer vision <- Image stitching` and
  `Forensic engineering <- Test method`, so downstream evaluations should report
  connector quality and not treat MAG taxonomy edges as fully clean.
- The dev-500 minimal-tree expanded-role run generated 499/500 papers and 7,040
  role descriptions under
  `outputs/expanded_fos_role_descriptions_llm_root_dev_500_minimal_tree_gptoss120b`;
  paper `2963217755` is still missing 13 role rows. The generated roles were
  embedded with OpenAI `text-embedding-3-small`, normalized, producing
  `role_description_embedding_ids.tsv` and `role_description_embeddings.npy`
  with shape `(7040, 1536)`. A first no-training evaluation using these
  minimal-tree task nodes, LLM root role query, and expert-side direct+ancestor
  links restricted to the minimal-tree node set wrote
  `outputs/untrained_taxonomy_aggregated_expert_nodes_minimal_tree_llm_root/ancestor_unweighted_full_to_root`.
  On the 499 covered dev papers it assigned 6,514 rows, skipped 526 no-candidate
  nodes, and scored micro P/R/F1 16.165183/54.362416/24.920128 with 1,053 hits
  out of 1,937 gold authors. This keeps recall comparable to direct+root but
  precision drops because every minimal-tree node is assigned; this setting
  should feed Stage-2 cutting/selection rather than assigning all nodes directly.
- Stage-1 candidate-generation recall should be evaluated by per-node Top-M
  union rather than Hungarian assignment. `scripts/evaluate_untrained_taxonomy_topm_union.py`
  writes such metrics. On dev, direct-node roles with expert direct links reached
  micro recall 35.86/61.00/68.83/76.66/82.53/88.82 at per-node M
  1/3/5/10/20/50, with average candidate-pool sizes
  6.95/18.32/29.40/55.51/101.47/214.02. The same direct roles with expert
  ancestor links reached 37.45/62.13/69.24/78.00/83.15/89.44 at similar pool
  sizes. Minimal-tree LLM-root roles reached 46.52/66.55/73.41/80.28/86.68/92.36
  at average pool sizes 9.79/25.12/39.67/73.53/132.37/274.71. Whole-paper to
  expert-mean Top-K reached 46.83/57.44/68.93/73.93/80.68/86.66/92.74 at
  K=5/10/25/40/74/132/275. Thus minimal-tree role union improves over direct
  node union, but at comparable candidate-pool sizes whole-paper Top-K remains
  slightly stronger or tied for exact-member recall.
- Minimal-tree expanded-role prompt variants were added without changing the
  default prompt: `--expanded-prompt-variant v2_boundary|retrieval_dense|keyword_query|balanced_recall`.
  On the same 20 dev papers sampled from the generated `retrieval_dense` run
  (`outputs/prompt_variant_eval_dev20/shared_sample/dev20_same_papers.jsonl`),
  per-node `m=1` Top-M union with expert ancestor links gave: existing
  `v2_boundary` P/R/F1 46.601942/71.641791/56.470588 with avg pool 5.15;
  `retrieval_dense` 53.488372/68.656716/60.130719 with avg pool 4.30;
  `keyword_query` 53.012048/65.671642/58.666667 with avg pool 4.15; and
  `balanced_recall` 44.761905/70.149254/54.651163 with avg pool 5.25. By raw
  recall, the old `v2_boundary` prompt remains best on this small sample. By
  recall per average candidate (`R/avg_pool`), `retrieval_dense` is best
  (~0.1597 vs old ~0.1391), but it loses two exact gold hits on 67 gold authors.
- `scripts/generate_stage1_task_node_role_descriptions.py` now also supports
  `--expanded-prompt-variant direct_style`, which applies the original
  direct-FoS prompt used for the completed train-5000 LLM run to expanded
  minimal-tree nodes. The node set can include `__root__`, direct FoS nodes, and
  minimal-tree connector ancestors, but the prompt text is the old flat
  "single technical facet / discriminate between nodes" wording and emits the
  same compact JSON fields as the direct-node prompt. Expert-side evaluation
  scripts now also support `--link-mode minimal_tree`, where each historical
  paper links only to its own compact direct+connector minimal-tree nodes rather
  than every ancestor up to root. On the same 20-paper dev prompt-variant sample
  with per-node `m=1` Top-M union, old `v2_boundary` + full-ancestor link had
  P/R/F1 46.601942/71.641791/56.470588 with avg pool 5.15; `direct_style` +
  full-ancestor link improved raw recall to 73.134328 with P/F1 44.545455/55.367232
  and avg pool 5.50; old `v2_boundary` + minimal-tree link scored
  43.243243/71.641791/53.932584 with avg pool 5.55; `direct_style` +
  minimal-tree link scored 39.316239/68.656716/50.000000 with avg pool 5.85.
  Thus the old direct-style prompt may help recall, but strict expert-side
  minimal-tree linking reduced candidate coverage on this small sample.
- GPT-5 mini role-description generation uses the OpenAI backend with
  `model=gpt-5-mini`; the local generator sends `max_completion_tokens` for
  OpenAI `gpt-5*` models because they reject the older `max_tokens` parameter.
  The formal 2019 hist>=5 minimal-tree test set was generated for both prompt
  variants on all 500 papers / 6,832 nodes:
  `outputs/expanded_fos_role_descriptions_minimal_tree_official_test_500_v2_boundary_gpt5mini`
  and
  `outputs/expanded_fos_role_descriptions_minimal_tree_official_test_500_direct_style_gpt5mini`.
- Official 2019 hist>=5 Top-M union recall with minimal-tree expert-node
  construction was evaluated for `m=1,3,5,10,20,50` and summarized under
  `outputs/topm_union_minimal_tree_official_test_500_comparison/`. With
  `gpt-5-mini` role descriptions, `v2_boundary` beat `direct_style` at every
  m. Across available methods, `gpt-5-mini/v2_boundary` was best at m=1
  (micro R 46.08), while `gpt-oss-120b/direct_style` was best from m=3 onward
  (micro R 69.09/76.22/83.70/89.03/93.95 for m=3/5/10/20/50).
- The same three official 2019 hist>=5 minimal-tree Top-M union methods were
  rerun with `--node-weight-mode weighted`, so expert-node vectors use
  FoS-weighted history-paper means normalized by the summed weights. The curve
  outputs are under
  `outputs/topm_union_minimal_tree_official_test_500_weighted_comparison/`.
  Micro recall for `gpt-5-mini/v2_boundary` was
  45.52/68.07/75.14/82.68/88.31/93.44 at m=1/3/5/10/20/50;
  `gpt-5-mini/direct_style` was 43.21/66.17/73.55/81.45/87.54/92.93; and
  `gpt-oss-120b/direct_style` was 45.11/69.25/76.47/83.85/89.24/93.85.
  Weight-normalized expert-node construction leaves the ranking pattern mostly
  unchanged: `gpt-5-mini/v2_boundary` is still best at m=1, while
  `gpt-oss-120b/direct_style` is best from m=3 onward.
  A six-line unweighted-vs-weighted comparison plot/table was written to
  `outputs/topm_union_minimal_tree_official_test_500_six_line_comparison/`.
- `scripts/evaluate_root_to_expert_mean_topk.py` evaluates the root-role
  baseline: use each task's generated `__root__` role embedding as the query
  and compare it against each candidate author's mean pre-2019 history-paper
  embedding. Same-K and candidate-pool-matched baseline tables were written to
  `outputs/topm_union_minimal_tree_official_test_500_comparison/`. Under
  comparable candidate-pool sizes, root-mean retrieval is stronger at the
  smallest pool (K=10, micro R 55.72 for `gpt-oss-120b/direct_style` vs 46.08
  best node-union m=1), but minimal-tree node union is stronger from m=3 onward
  among currently evaluated methods.
- `scripts/evaluate_stage1_projected_whole_mean_search.py` isolates the trained
  Stage-1 projection space by doing whole-paper to whole-expert retrieval:
  average all raw node role embeddings for each task, apply trained `g_phi`,
  average each candidate author's full history-paper embeddings, apply trained
  `g_psi`, then retrieve top-k where `k = task node count`. On the official
  2019 hist>=5 test set with checkpoint
  `outputs/stage1_training/task_expert_node_dev_and_test_v1/checkpoint_last.pt`,
  this wrote
  `outputs/stage1_projected_whole_mean_search/task_expert_node_v1_mean_nodes`.
  Micro P/R/F1 was 0.174376/0.369042/0.236842, compared with raw mean baseline
  0.267861/0.566889/0.363816 and Stage-1 Hungarian node assignment
  0.171220/0.359815/0.232028. This suggests the learned projection space itself
  is hurting whole-paper matching, not only the node-granularity retrieval.
- `scripts/generate_stage1_task_node_role_descriptions.py` now includes
  `--expanded-prompt-variant v3_retrieval`. It keeps the `v2_boundary` tree-aware
  schema (`subtree_scope`, `covered_direct_fos`, `distinctive_boundary`) but
  rewrites the prompt objective so `role_description` is a dense expert-profile
  retrieval query: direct FoS nodes are the primary retrieval surface, root is a
  whole-paper query, and connector/ancestor nodes are conservative subtree
  queries grounded in `covered_direct_fos` and paper evidence.
- A dev20 same-paper smoke test for `v3_retrieval` used Together
  `openai/gpt-oss-120b`, wrote
  `outputs/expanded_fos_role_prompt_variants_dev20_same/v3_retrieval`, embedded
  with OpenAI `text-embedding-3-small`, and wrote
  `outputs/prompt_variant_eval_dev20_same/prompt_variant_comparison_with_v3.tsv`.
  With ancestor expert links at m=1, `v3_retrieval` tied `v2_boundary` recall
  at 71.64 but improved precision/F1 to 51.06/59.63; `direct_style` still had
  the highest ancestor-link recall at 73.13. With minimal-tree expert links at
  m=1, `v3_retrieval` was best among tested minimal-tree-link prompts:
  P/R/F1 49.51/76.12/60.00 versus `v2_boundary` 43.24/71.64/53.93 and
  `direct_style` 39.32/68.66/50.00.
- The `v3_retrieval` prompt was run on the formal 2019 hist>=5 official test
  split with `gpt-5-mini` and 100-way LLM concurrency. One paper
  (`2955967180`) repeatedly returned 21/22 expected nodes, so the final
  comparison used the first 499 completed papers and wrote role descriptions
  and embeddings under
  `outputs/expanded_fos_role_descriptions_minimal_tree_official_test_500_v3_retrieval_gpt5mini`.
  Strict minimal-tree expert-node Top-M union evaluation for those 499 papers
  wrote
  `outputs/topm_union_minimal_tree_official_test_499_v3_retrieval_gpt5mini/minimal_tree_link`.
  Micro recall for m=1/3/5/10/20/50 was
  48.79/71.03/77.61/84.33/89.52/93.73. Compared against the existing 500-paper
  curves for `gpt-5-mini/v2_boundary`, `gpt-5-mini/direct_style`, and
  `gpt-oss-120b/direct_style`, v3 is best from m=1 through m=20; at m=50,
  `gpt-oss-120b/direct_style` remains slightly higher (93.95 vs 93.73).
  The correct no-node baseline for this curve uses each task's original
  whole-paper embedding, not the generated `__root__` role-description
  embedding, and matches it directly to each candidate author's mean
  history-paper embedding with `K=m` for m=1/3/5/10/20/50. On the same 499
  papers, its micro recall was 17.51/39.14/49.20/60.86/69.65/80.43. The mixed
  499-vs-500 comparison table and curves, including the paper-embedding
  baseline overlay, are under
  `outputs/topm_union_minimal_tree_official_test_v3_499_vs_prior500_comparison`.
- A full `v3_retrieval` minimal-tree role-description generation pass was
  started for the canonical train/dev/test splits using `gpt-5-mini`:
  train output directory
  `outputs/expanded_fos_role_descriptions_minimal_tree_train_5000_v3_retrieval_gpt5mini`,
  dev output directory
  `outputs/expanded_fos_role_descriptions_minimal_tree_dev_500_v3_retrieval_gpt5mini`,
  and formal-test output directory
  `outputs/expanded_fos_role_descriptions_minimal_tree_official_test_500_v3_retrieval_gpt5mini`.
  The earlier run paused when OpenAI returned `insufficient_quota`, then was
  resumed after quota was restored. Stubborn papers that repeatedly returned
  one fewer JSON object than expected were resolved by preserving full-prompt
  outputs when possible and, for `2955967180`, generating only the missing
  connector node (`79974875`, Cloud computing) with a single-node v3 prompt.
  Final completion is train 5,000/5,000 papers with 70,982 role rows, dev
  500/500 papers with 7,053 role rows, and formal test 500/500 papers with
  6,832 role rows. On 2026-07-05, the OpenAI `text-embedding-3-small`
  normalized role-description embedding caches were generated with
  `--text-field role_description`: train has shape `(70982, 1536)`, dev has
  shape `(7053, 1536)`, and formal test has shape `(6832, 1536)`. The
  corresponding `role_description_embedding_ids.tsv` files match the current
  JSONL record order and all three splits have zero duplicate role ids.
- The direct-style minimal-tree prompt was run on the formal 2019 hist>=5
  official test-500 split with 100-way LLM concurrency after an initial resume
  from lower-concurrency runs. Output directory:
  `outputs/expanded_fos_role_descriptions_direct_style_minimal_tree_official_test_500_gptoss120b`.
  The run completed all 500 papers and 6,832 role rows; embeddings were built
  with OpenAI `text-embedding-3-small`, normalized. Per-node `m=1` Top-M union
  evaluation wrote
  `outputs/topm_union_direct_style_minimal_tree_official_test_500/ancestor_link`
  and `outputs/topm_union_direct_style_minimal_tree_official_test_500/minimal_tree_link`.
  With expert full-ancestor links, official-test P/R/F1 was
  17.288408/44.182471/24.852242 with avg pool 9.972, 862 hits out of 1,951 gold
  authors, and 4,986 total predicted unique authors. With strict expert
  minimal-tree links, P/R/F1 was 17.024235/43.926192/24.538296 with avg pool
  10.068, 857 hits, and 5,034 total predicted unique authors. Full-ancestor
  linking remains slightly better on this test run.
