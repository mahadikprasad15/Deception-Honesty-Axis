# Synthetic V2 Iteration

This iteration introduces two explicit synthetic candidate pools:

- `synthetic_v1_pruned`
- `synthetic_v2_orthogonal`

The workflow is split into:

- a loud, resume-first pipeline wrapper
- a fixed-track greedy search runner

## Pipeline wrapper

Run a variant end-to-end from the repo root:

```bash
PYTHONPATH=src python3 scripts/run_variant_pipeline.py \
  --experiment-config configs/experiments/synthetic_v1_pruned_llama32_3b.json \
  --probe-config configs/probes/role_axis_transfer_synthetic_v1_pruned_all_datasets_zeroshot.json \
  --batch-size 2 \
  --progress-every 10
```

Swap the configs to run the orthogonal pool:

```bash
PYTHONPATH=src python3 scripts/run_variant_pipeline.py \
  --experiment-config configs/experiments/synthetic_v2_orthogonal_llama32_3b.json \
  --probe-config configs/probes/role_axis_transfer_synthetic_v2_orthogonal_all_datasets_zeroshot.json \
  --batch-size 2 \
  --progress-every 10
```

Useful flags:

- `--run-id <id>` to keep all analysis stages aligned to a known run id
- `--stages role_vectors pca axis_bundle transfer postprocess` to reuse an existing corpus
- `--dry-run` to print the planned commands and detected outputs

Wrapper manifests and logs are written under:

```text
artifacts/runs/variant-pipeline/<model>/<dataset>/<variant>/<run_id>/
```

## Fixed-track search

The fixed-track search keeps one probe and one layer fixed for the entire run.

### `contrast_zero_shot @ 14`

```bash
PYTHONPATH=src python3 scripts/run_fixed_track_search.py \
  --experiment-config configs/experiments/synthetic_v1_pruned_llama32_3b.json \
  --probe-config configs/probes/role_axis_transfer_synthetic_v1_pruned_all_datasets_zeroshot.json \
  --method contrast_zero_shot \
  --layer-spec 14 \
  --progress-every 10
```

### `pc1_zero_shot @ 14`

```bash
PYTHONPATH=src python3 scripts/run_fixed_track_search.py \
  --experiment-config configs/experiments/synthetic_v2_orthogonal_llama32_3b.json \
  --probe-config configs/probes/role_axis_transfer_synthetic_v2_orthogonal_all_datasets_zeroshot.json \
  --method pc1_zero_shot \
  --layer-spec 14 \
  --progress-every 10
```

Useful flags:

- `--run-id <id>` to resume or compare a specific search run
- `--resume` to continue from `checkpoints/search_state.json`
- `--min-honest`, `--min-deceptive`, `--min-questions` to change pruning floors

Search artifacts are written under:

```text
artifacts/runs/fixed-track-search/<model>/<dataset>/<variant>/<method>__layer-14/<run_id>/
```

Key outputs:

- `results/baseline_per_dataset.csv`
- `results/final_per_dataset.csv`
- `results/step_history.csv`
- `results/candidate_moves/step_###.csv`
- `results/final_selection.json`
- `plots/baseline_vs_final_auroc.png`
- `plots/delta_vs_baseline.png`
- `plots/search_trajectory.png`
