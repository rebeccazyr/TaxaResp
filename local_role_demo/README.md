# Local Role Assignment Demo

This local web demo visualizes
`embedding_bfs_unique_assign_each_node_then_top_team_size_by_weighted_score` using existing cached
outputs. It does not call an LLM or encode new embeddings in the browser flow.

The page reads the precomputed ranking from
`output/embedding_bfs_unique_assignment_no_label/predictions_team_size.tsv`, shows the
method metrics, then visualizes for each selected task:

- task taxonomy nodes in BFS order
- existing no-label task-node embedding IDs
- same-node expert assignment from `node_assignments.tsv`
- cosine similarity, node importance, and weighted score
- final top `team_size` experts selected by weighted score

The older `/api/analyze` endpoint is still present for the LLM demo, but the
default page uses only `/api/method/*`.

By default it builds a quick local expert index from
`output/expert_profile/*_direct_fos_nodes.tsv`. The full HieRec evidence JSONL
can also be used with `--source jsonl`, but that file is large and slow to load.

Start it from the repository root:

```bash
python3 local_role_demo/server.py --port 8765
```

Optional full-index mode:

```bash
python3 local_role_demo/server.py --source jsonl --port 8765
```

Open:

```text
http://127.0.0.1:8765
```

No Together API key is needed for the method visualizer.
