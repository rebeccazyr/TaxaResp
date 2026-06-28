# Experiment Results Summary

Generated from the current `output/` and `logs/` directories on 2026-06-07.
This file is an index and interpretation layer only; no experiment artifacts were
moved or modified.

## Scope

The repository currently contains two major groups of experiment outputs:

1. Classic OpeNTF toy-domain cross-validation results under
   `output/{dblp,imdb,gith,uspt}/toy.../splits.f3.r0.85/`.
2. New taxonomy / HieRec / embedding-based expert-finding experiments under
   `output/taxonomy_team_formation_experiment/`,
   `output/hierec_embedding_server_inputs/`,
   `output/hierec_expert_node_embeddings*/`, and
   `output/hierec_embedding_team_formation_experiment_20/`.

## Output Inventory

| Path | Files | Size | Meaning |
| --- | ---: | ---: | --- |
| `output/dblp/` | 1,032 | 89M | DBLP domain preprocessing, toy model runs, plots |
| `output/imdb/` | 879 | 97M | IMDb domain preprocessing, toy model runs, plots |
| `output/gith/` | 1,085 | 183M | GitHub domain preprocessing, toy model runs, plots |
| `output/uspt/` | 740 | 101M | USPTO domain preprocessing, toy model runs, plots |
| `output/expert_profile/` | 13,816 | 1.1G | Full-period direct FoS profiles for experts |
| `output/expert_profile_year_bins/` | 69,078 | 2.8G | Expert FoS profiles split by time bins |
| `output/expert_profile_year_bins_drift/` | 2 | 16M | Level-wise temporal drift summaries |
| `output/expert_graph/` | 3 | 20M | Expert coauthor / communication graph |
| `output/taxonomy_team_formation_experiment/` | 14 | 4.4M | Taxonomy/direct scoring team-formation evaluations |
| `output/hierec_embedding_server_inputs/` | 10 | 3.6G | Full HieRec embedding cache inputs and `.npy` embeddings |
| `output/hierec_embedding_server_inputs_debug_1/` | 6 | 43M | One-expert debug cache |
| `output/hierec_expert_node_embeddings*/` | 9 | 14M | Sample expert-node embedding runs |
| `output/hierec_embedding_team_formation_experiment_20/` | 3 | 496K | HieRec fixed-budget team-size evaluation over 20 experts |
| `output/groundtruth_expert_paper_embeddings/` | 0 | 0B | Empty placeholder |

## Classic OpeNTF Toy Runs

Final cross-validation aggregate files:

- `output/dblp/toy.dblp.v12.json/splits.f3.r0.85/test.pred.eval.mean.agg.csv`
- `output/imdb/toy.title.basics.tsv/splits.f3.r0.85/test.pred.eval.mean.agg.csv`
- `output/gith/toy.repos.csv/splits.f3.r0.85/test.pred.eval.mean.agg.csv`
- `output/uspt/toy.patent.tsv/splits.f3.r0.85/test.pred.eval.mean.agg.csv`

Best model by the main retrieval metrics is consistently the GCN run:
`gcn.b1000.e100.ns5.lr0.001.es5.spe10.d128.add.stm.h128.nn30-20`.

| Dataset | P@5 | Recall@5 | NDCG@5 | MAP@5 | AUC-ROC |
| --- | ---: | ---: | ---: | ---: | ---: |
| DBLP toy | 0.373333 | 0.844444 | 0.771647 | 0.702037 | 0.822671 |
| IMDb toy | 0.333333 | 0.390432 | 0.495996 | 0.300231 | 0.878640 |
| GitHub toy | 0.366667 | 0.618519 | 0.692772 | 0.599630 | 0.674871 |
| USPTO toy | 0.366667 | 0.777778 | 0.732891 | 0.643519 | 0.897674 |

Non-GCN baselines are materially weaker in these aggregate files. For example,
the best non-GCN P@5 values are 0.253333 on DBLP, 0.066667 on IMDb, 0.122222 on
GitHub, and 0.133333 on USPTO.

## Taxonomy Team-Formation Experiment

Main files:

- `output/taxonomy_team_formation_experiment/metrics_summary.tsv`
- `output/taxonomy_team_formation_experiment/direct_weighting_comparison.tsv`
- `output/taxonomy_team_formation_experiment/direct_team_size_metrics.tsv`
- `output/taxonomy_team_formation_experiment/recursive_rule_threshold_sweep.tsv`

Core comparison over 262 tasks:

| Method | Recall@team_size | P@2 | P@5 | P@10 | P@20 | MRR first actual member |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| direct | 0.108206 | 0.122137 | 0.091603 | 0.063740 | 0.043321 | 0.218262 |
| taxonomy | 0.089122 | 0.104962 | 0.083969 | 0.055725 | 0.037405 | 0.204072 |
| direct_taxonomy_blend | 0.107570 | 0.124046 | 0.091603 | 0.062977 | 0.044466 | 0.220407 |
| subtree_cover | 0.047519 | 0.068702 | 0.036641 | 0.020992 | 0.010687 | 0.128609 |

Best direct-weighting variant:

| Variant | Recall@team_size | Precision@team_size | P@2 | Recall@2 | NDCG@2 | MAP@2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `direct_log_sum_with_idf_cosine` | 0.129071 | 0.129071 | 0.146947 | 0.089885 | 0.153424 | 0.133588 |

Interpretation: direct FoS evidence is currently stronger than subtree-cover
taxonomy traversal. The log-sum + IDF + cosine variant is the best taxonomy-era
scoring result found in the current artifacts.

## HieRec Embedding Pipeline

Full cache summary from `output/hierec_embedding_server_inputs/summary.tsv`:

| Metric | Value |
| --- | ---: |
| tasks | 262 |
| task_nodes | 5,576 |
| experts | 13,815 |
| expert_direct_nodes | 1,656,638 |
| taxonomy_nodes | 32,855 |
| requested_papers | 1,304,787 |
| loaded_papers | 1,304,787 |

Debug cache `output/hierec_embedding_server_inputs_debug_1/` contains the same
262 tasks but only 1 expert, 20 direct nodes, 1,228 taxonomy nodes, and 295
loaded papers.

Sample expert-node embedding runs:

| Path | Experts | Unique taxonomy nodes | Expert-node embeddings | Dim | Method |
| --- | ---: | ---: | ---: | ---: | --- |
| `output/hierec_expert_node_embeddings/` | 100 | 3,875 | 14,169 | 64 | TFIDF_SVD node semantic + child attention |
| `output/hierec_expert_node_embeddings_example_1/` | 1 | 64 | 64 | 64 | TFIDF_SVD node semantic + direct paper attention + child attention |
| `output/hierec_expert_node_embeddings_abstract_sample/` | 3 | 48 | 61 | 64 | TFIDF_SVD node semantic + direct paper attention + child attention |
| `output/hierec_expert_node_embeddings_abstract_20/` | 20 | 753 | 1,517 | 64 | TFIDF_SVD node semantic + direct paper attention + child attention |

Fixed-budget HieRec result from
`output/hierec_embedding_team_formation_experiment_20/metrics_summary.tsv`:

| Tasks | Experts | Embedding dim | Avg assigned nodes | Avg unique assigned experts | Avg selected experts | Macro P@team_size | Macro R@team_size | Micro P@team_size | Micro R@team_size |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 262 | 20 | 64 | 16.904580 | 7.614504 | 3.179389 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |

This is an important negative result. With the current 20-expert HieRec run,
none of the selected experts match the ground-truth teams. This should be
treated as a pipeline or candidate-pool issue before comparing it with the
classic OpeNTF or taxonomy results.

## Expert Profiles and Drift

Full-period profile summary from `output/expert_profile/_summary.tsv`:

| Metric | Value |
| --- | ---: |
| experts | 13,815 |
| average papers per expert | 159.96 |
| average direct FoS nodes per expert | 407.02 |
| max papers for one expert | 1,367 |
| max direct FoS nodes for one expert | 2,001 |

Time-bin profile averages from `output/expert_profile_year_bins/_summary.tsv`:

| Bin | Experts | Avg papers | Avg direct FoS nodes |
| --- | ---: | ---: | ---: |
| train_2000_2004 | 13,815 | 16.46 | 69.70 |
| train_2005_2009 | 13,815 | 32.16 | 125.60 |
| valid_2010_2014 | 13,815 | 45.14 | 166.55 |
| test_2015_2019 | 13,815 | 52.54 | 172.57 |

Temporal drift summary from
`output/expert_profile_year_bins_drift/summary_by_level.tsv`:

| Transition | L1 mean | L2 mean | L3 mean | L4 mean |
| --- | ---: | ---: | ---: | ---: |
| 2000-2004 -> 2005-2009 | 0.174932 | 0.301641 | 0.594857 | 0.638815 |
| 2005-2009 -> 2010-2014 | 0.049521 | 0.161665 | 0.497158 | 0.559014 |
| 2010-2014 -> 2015-2019 | 0.027243 | 0.135979 | 0.462374 | 0.562196 |

The higher-level L1/L2 distributions stabilize after 2005, while deeper L3/L4
topic distributions remain much more volatile.

## Expert Communication Graph

From `output/expert_graph/communication_stats.txt`:

| Metric | Value |
| --- | ---: |
| experts_total | 13,815 |
| parsed_papers | 4,894,081 |
| matched_papers_with_2plus_experts | 476,504 |
| matched_expert_mentions | 2,209,889 |
| edges_total | 160,768 |

Graph files:

- `output/expert_graph/communication_nodes.tsv`
- `output/expert_graph/communication_edges.tsv`

## Logs

Available log files:

- `logs/build_all_expert_profiles.log`: full expert-profile construction log.
- `logs/paper_embeddings.log`: embedding-server paper embedding log.
- `logs/paper_embeddings.pid`: recorded embedding process PID.

## Items To Recheck

1. `output/groundtruth_expert_paper_embeddings/` is empty.
2. HieRec fixed-budget evaluation over 20 experts has zero precision and recall.
   Likely causes include too small a candidate pool, ID mismatch between
   predictions and ground truth, or task-node / expert-node embedding mismatch.
3. The strongest current non-neural taxonomy result is
   `direct_log_sum_with_idf_cosine`, not `subtree_cover`.
4. Classic toy runs are only for toy datasets; full-domain folders mainly
   contain preprocessing artifacts and plots, not final model comparisons.
