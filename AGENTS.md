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

- `data/dblp/`: DBLP publication/team formation data.
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

## Important Decisions

- New reusable implementation should go under `src/`; use `scripts/` only for
  runnable orchestration.
- Preserve existing `output/` paths until callers/configs are migrated; use
  `outputs/` for new generated-result locations.
- The local embedding service runs outside the Codex sandbox. If a sandboxed
  command cannot reach it, rerun the embedding/smoke-test command with elevated
  permissions before assuming the service is down.

## TODOs

- Consider migrating legacy `output/` paths to `outputs/` in a coordinated
  pipeline/config update.
- Consider moving root-level historical `*.log` files into `logs/` after
  confirming no scripts depend on their current paths.
- Keep `data_preprocess/` cleanup on the backlog: reusable modules should move
  into `src/`, while runnable commands should move into `scripts/`.

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
