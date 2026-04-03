# Synthetic Pilot V1

`synthetic_pilot_v1` is a synthetic-only role-axis variant built from:

- `default` anchor
- 8 synthetic deceptive roles
- 6 synthetic honest roles
- 25 synthetic shared questions
- 2 prompt variants per role for the initial pilot

The zero-shot transfer probe evaluates:

- `contrast_zero_shot`
- `pc1_zero_shot`
- `pc2_zero_shot`
- `pc3_zero_shot`

on all 9 deception datasets at layers `14`, `21`, and `28`.

## Configs

- experiment config:
  - `configs/experiments/synthetic_pilot_v1_llama32_3b.json`
- transfer config:
  - `configs/probes/role_axis_transfer_synthetic_pilot_v1_all_datasets_zeroshot.json`

## Expected work units

- roles including anchor: `15`
- prompt variants used: `2`
- questions: `25`
- total work units: `750`

## Pipeline

Run from the repo root with `PYTHONPATH=src`.

```bash
RUN_ID="$(PYTHONPATH=src python3 - <<'PY'\nfrom deception_honesty_axis.common import make_run_id\nprint(make_run_id())\nPY\n)"
```

### 1. Generate rollouts

```bash
PYTHONPATH=src python3 scripts/generate_rollouts.py \
  --config configs/experiments/synthetic_pilot_v1_llama32_3b.json \
  --batch-size 2
```

### 2. Extract activations

```bash
PYTHONPATH=src python3 scripts/extract_activations.py \
  --config configs/experiments/synthetic_pilot_v1_llama32_3b.json
```

### 3. Build role vectors

```bash
PYTHONPATH=src python3 scripts/build_role_vectors.py \
  --config configs/experiments/synthetic_pilot_v1_llama32_3b.json \
  --run-id "$RUN_ID"
```

Vectors land under:

```text
artifacts/runs/role-vectors/<model>/<dataset>/synthetic-pilot-v1/all-responses/$RUN_ID
```

### 4. Run PCA

```bash
PYTHONPATH=src python3 scripts/run_pca_analysis.py \
  --config configs/experiments/synthetic_pilot_v1_llama32_3b.json \
  --vectors-run-dir "artifacts/runs/role-vectors/meta-llama-llama-3-2-3b-instruct/assistant-axis/synthetic-pilot-v1/all-responses/$RUN_ID" \
  --run-id "$RUN_ID"
```

Selected-layer PCA scatters are written for `PC1-PC2`, `PC1-PC3`, and `PC2-PC3`.

### 5. Build the reusable role-axis bundle

```bash
PYTHONPATH=src python3 scripts/build_role_axis_bundle.py \
  --config configs/probes/role_axis_transfer_synthetic_pilot_v1_all_datasets_zeroshot.json \
  --vectors-run-dir "artifacts/runs/role-vectors/meta-llama-llama-3-2-3b-instruct/assistant-axis/synthetic-pilot-v1/all-responses/$RUN_ID" \
  --run-id "$RUN_ID"
```

### 6. Evaluate zero-shot transfer

```bash
PYTHONPATH=src python3 scripts/evaluate_role_axis_transfer.py \
  --config configs/probes/role_axis_transfer_synthetic_pilot_v1_all_datasets_zeroshot.json \
  --axis-bundle-run-dir "artifacts/runs/role-axis-bundles/meta-llama-llama-3-2-3b-instruct/assistant-axis/synthetic-pilot-v1/mean_response/$RUN_ID" \
  --run-id "$RUN_ID"
```

### 7. Regenerate summaries and plots from saved transfer results

```bash
PYTHONPATH=src python3 scripts/postprocess_role_axis_transfer.py \
  --transfer-run-dir "artifacts/runs/role-axis-transfer/meta-llama-llama-3-2-3b-instruct/assistant-axis/synthetic-pilot-v1/completion_mean/$RUN_ID"
```

## Result artifacts

The transfer run writes:

- `results/pairwise_metrics.csv`
- `results/per_example_scores.jsonl`
- `results/summary_by_method.csv`
- `results/fit_summary.csv`
- `results/plots/*heatmap.png`
- `results/plots/role_axis_transfer_by_layer__auroc.png`
- `results/plots/zero_shot_grouped_bars__auroc__14.png`
- `results/plots/zero_shot_grouped_bars__auroc__21.png`
- `results/plots/zero_shot_grouped_bars__auroc__28.png`
- `results/plots/zero_shot_grouped_bars__auroc__overview.png`

The grouped-bar plots are the primary synthetic-pilot comparison view:

- x-axis = datasets
- y-axis = AUROC
- bars = `contrast`, `PC1`, `PC2`, `PC3`
