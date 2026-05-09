# Deception-Honesty-Axis

Research code for building role/persona-derived behavioral axes and evaluating how those axes transfer to deception and honesty-related benchmarks.

The repository provides a reproducible pipeline for:

- generating role-conditioned model responses,
- extracting pooled response activations,
- building role-vector and PCA-based axis artifacts,
- packaging reusable role-axis bundles,
- evaluating zero-shot, trained, and activation-row transfer probes,
- producing structured metrics and paper-facing figures.

Long-running jobs are designed to be resumable and artifact-first: corpora, run manifests, checkpoints, logs, metrics, and plots are written under `artifacts/`.

## Project Structure

```text
configs/
  experiments/   role-axis corpus and activation configs
  probes/        transfer, projection, and activation-row evaluation configs
  variants/      curated source-role/question specs
  paper/         figure-generation configs
data/            tracked role manifests, role instructions, and question files
docs/            runtime notes and experiment runbooks
scripts/         command-line pipeline stages and wrappers
src/             shared Python package
tests/           unit tests for config, artifact, transfer, and plotting utilities
artifacts/       generated outputs; usually not committed
```

## Installation

Python 3.12 or newer is expected.

```bash
python3 -m pip install -r requirements.txt
python3 -m pip install -e .
```

Most commands are run from the repository root with:

```bash
export PYTHONPATH=src
```

## Quick Checks

Run the test suite:

```bash
PYTHONPATH=src pytest
```

Preview an end-to-end role-axis run without executing the child stages:

```bash
PYTHONPATH=src python3 scripts/run_variant_pipeline.py \
  --experiment-config configs/experiments/quantity_axis_v2_llama32_3b.json \
  --probe-config configs/probes/role_axis_transfer_quantity_v2_cumulative_pc_sweep.json \
  --batch-size 2 \
  --progress-every 10 \
  --dry-run
```

Audit the corpus work units for a config:

```bash
PYTHONPATH=src python3 scripts/audit_corpus.py \
  --config configs/experiments/quantity_axis_v2_llama32_3b.json
```

## Main Pipeline

The main role-axis workflow is orchestrated by `scripts/run_variant_pipeline.py`.

```bash
PYTHONPATH=src python3 scripts/run_variant_pipeline.py \
  --experiment-config configs/experiments/quantity_axis_v2_llama32_3b.json \
  --probe-config configs/probes/role_axis_transfer_quantity_v2_cumulative_pc_sweep.json \
  --batch-size 2 \
  --progress-every 10
```

With both configs provided, the wrapper runs the standard sequence:

```text
rollouts -> activations -> role_vectors -> pca -> axis_bundle -> transfer -> postprocess
```

Useful options:

- `--run-id <id>` uses a fixed run id for coordinated runs.
- `--stages ...` runs only selected stages, for example `role_vectors pca axis_bundle transfer postprocess`.
- `--force-stage <stage>` reruns a stage whose outputs already exist.
- `--dry-run` prints planned commands and output locations.

The wrapper writes its own manifest and logs under:

```text
artifacts/runs/variant-pipeline/<model>/<dataset>/<role_set>/<run_id>/
```

## Evaluation Workflows

Role-axis transfer evaluates directions derived from role-conditioned activations on external activation datasets. The primary scripts are:

- `scripts/build_role_axis_bundle.py`
- `scripts/evaluate_role_axis_transfer.py`
- `scripts/postprocess_role_axis_transfer.py`
- `scripts/compare_role_axis_zero_shot_runs.py`

Activation-row baselines evaluate dataset-derived activation probes against the same target datasets:

- `scripts/evaluate_activation_row_transfer.py`
- `scripts/evaluate_activation_row_transfer_subspace_baselines.py`
- `scripts/evaluate_activation_row_transfer_pc_projection.py`
- `scripts/evaluate_activation_row_transfer_pc_projection_sweep.py`
- `scripts/evaluate_activation_row_transfer_multi_pc_projection.py`
- `scripts/compare_activation_row_transfer_runs.py`

Paper and analysis helpers include:

- `scripts/build_paper_axis_figures.py`
- `scripts/plot_role_axis_pc_scatter.py`
- `scripts/plot_role_axis_score_histograms.py`
- `scripts/analyze_behavior_axis_pc_clusters.py`
- `scripts/analyze_dataset_geometry.py`
- `scripts/analyze_greedy_search_evolution.py`

## Artifacts

Generated corpora are stored as append-only shards:

```text
artifacts/corpora/<model>/<dataset>/<role_set>/
  meta/
  indexes/
  rollouts/
  activations/
  checkpoints/
  logs/
```

Analysis and evaluation runs use:

```text
artifacts/runs/<experiment>/<model>/<dataset>/<variant-or-role-set>/<run_id>/
  meta/run_manifest.json
  meta/*_status.json
  checkpoints/
  results/
  logs/
```

Resume behavior is driven by indexes, status files, and checkpoints. Re-running a stage generally skips completed items or completed metric combinations and appends only the missing work.

## Artifact Sync

Large outputs can be synced through the configured Hugging Face dataset repository.

```bash
PYTHONPATH=src python3 scripts/sync_hf_artifacts.py \
  --direction push \
  --local-dir artifacts
```

Selective pulls are recommended in Colab and other small-disk environments:

```bash
PYTHONPATH=src python3 scripts/sync_hf_artifacts.py \
  --direction pull \
  --repo-id Prasadmahadik/deception-honesty-axis-artifacts \
  --local-dir artifacts \
  --allow-patterns "runs/role-axis-bundles/**" "corpora/**/meta/**" "corpora/**/indexes/**"
```

## Runtime Notes

- Full rollout and activation jobs require a GPU environment suitable for the configured model.
- Some transfer configs reference Drive-backed activation roots used in Colab workflows.
- Fixed `--run-id` values are useful when coordinating multi-stage runs across machines.
- Generated experiment outputs should stay under `artifacts/`.

Detailed runbooks:

- `docs/runtime/colab.md`
- `docs/runtime/vast.md`
- `docs/colab_8b_pipeline.md`
- `docs/experiments/four_axis_zero_shot_baselines.md`
- `docs/experiments/synthetic_v2_iteration.md`
