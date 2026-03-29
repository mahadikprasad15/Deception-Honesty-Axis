# Deception-Honesty-Axis

Resumable scripts for an assistant-axis-style pilot on deception roles with `meta-llama/Llama-3.2-3B-Instruct`.

## What This Repo Does

The initial pipeline covers:

- source sync from the upstream `assistant-axis` repo
- rollout generation for `default` plus 6 deception roles
- pooled all-layer activation extraction
- per-role vector aggregation
- centered PCA analysis against the `default` anchor
- audit and HF artifact sync utilities

The first pass is configured for:

- first `2` instruction variants per selected role
- first `100` extraction questions
- append-only corpus storage so later expansion to `5 x 240` reuses the same directories

## Repo Layout

- `data/` stores pinned source inputs and manifests only
- `artifacts/corpora/...` stores the growing rollout and activation corpus
- `artifacts/runs/...` stores vector and PCA analysis runs
- `scripts/` contains the stage entrypoints
- `src/deception_honesty_axis/` contains shared helpers
- `notebooks/` contains the Colab bootstrap notebook

## Setup

```bash
python3 -m pip install -r requirements.txt
```

For editable local development:

```bash
python3 -m pip install -e .
```

## First Commands

Sync the upstream questions and role instructions:

```bash
PYTHONPATH=src python3 scripts/sync_source_data.py --config configs/experiments/deception_v1_llama32_3b.json
```

Audit the target work units before generation:

```bash
PYTHONPATH=src python3 scripts/audit_corpus.py --config configs/experiments/deception_v1_llama32_3b.json
```

Generate the missing rollouts:

```bash
PYTHONPATH=src python3 scripts/generate_rollouts.py --config configs/experiments/deception_v1_llama32_3b.json --batch-size 2
```

Extract pooled activations for the rollout items that do not have them yet:

```bash
PYTHONPATH=src python3 scripts/extract_activations.py --config configs/experiments/deception_v1_llama32_3b.json
```

Build role vectors and run PCA:

```bash
RUN_ID="$(python3 - <<'PY'\nfrom deception_honesty_axis.common import make_run_id\nprint(make_run_id())\nPY\n)"
PYTHONPATH=src python3 scripts/build_role_vectors.py --config configs/experiments/deception_v1_llama32_3b.json --run-id "$RUN_ID"
PYTHONPATH=src python3 scripts/run_pca_analysis.py --config configs/experiments/deception_v1_llama32_3b.json --vectors-run-dir "artifacts/runs/role-vectors/meta-llama-llama-3-2-3b-instruct/assistant-axis/deception-v1/all-responses/$RUN_ID" --run-id "$RUN_ID"
```

## Resume Behavior

- rollout generation skips any `item_id` already indexed in `artifacts/corpora/.../indexes/rollouts.jsonl`
- activation extraction skips any `item_id` already indexed in `artifacts/corpora/.../indexes/activations.jsonl`
- expanding the subset later is a config change plus rerun; old shards stay in place and only the missing delta is appended

## Runtime Notes

- Use Colab for bootstrap, source sync, and smoke tests.
- Use Vast.ai for the long rollout and activation jobs.
- Use the private HF dataset repo in the config as the shared artifact backup.

See `docs/runtime/colab.md` and `docs/runtime/vast.md` for the concrete workflow.
