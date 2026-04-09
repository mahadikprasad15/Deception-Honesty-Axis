# Four-Axis Zero-Shot Baselines

This baseline pass keeps the earlier `Deception-Honesty-Axis` workflow intact and
only evaluates zero-shot directions:

- `contrast_zero_shot`
- `pc1_zero_shot`

There is no greedy search and no learned target-side probe in this pass.

## Pipeline shape

Each axis uses the same existing wrapper stages:

1. `generate_rollouts.py`
2. `extract_activations.py`
3. `build_role_vectors.py`
4. `run_pca_analysis.py`
5. `build_role_axis_bundle.py`
6. `evaluate_role_axis_transfer.py`
7. `postprocess_role_axis_transfer.py`

The wrapper script remains the entrypoint:

```bash
PYTHONPATH=src python3 scripts/run_variant_pipeline.py \
  --experiment-config configs/experiments/quality_axis_v1_llama32_3b.json \
  --probe-config configs/probes/role_axis_transfer_quality_v1_all_datasets_zeroshot.json \
  --batch-size 2 \
  --progress-every 10
```

Swap the config pair to run the other axes:

- `quantity_axis_v1_llama32_3b.json`
- `relation_axis_v1_llama32_3b.json`
- `manner_axis_v1_llama32_3b.json`

and

- `role_axis_transfer_quantity_v1_all_datasets_zeroshot.json`
- `role_axis_transfer_relation_v1_all_datasets_zeroshot.json`
- `role_axis_transfer_manner_v1_all_datasets_zeroshot.json`

## Resume-friendly usage

Use a fixed `--run-id` if you want all four axes to line up under the same run
token:

```bash
PYTHONPATH=src python3 scripts/run_variant_pipeline.py \
  --experiment-config configs/experiments/relation_axis_v1_llama32_3b.json \
  --probe-config configs/probes/role_axis_transfer_relation_v1_all_datasets_zeroshot.json \
  --run-id 20260409T000000Z-four-axis \
  --batch-size 2 \
  --progress-every 10
```

If rollouts and activations already exist, reuse them:

```bash
PYTHONPATH=src python3 scripts/run_variant_pipeline.py \
  --experiment-config configs/experiments/manner_axis_v1_llama32_3b.json \
  --probe-config configs/probes/role_axis_transfer_manner_v1_all_datasets_zeroshot.json \
  --run-id 20260409T000000Z-four-axis \
  --stages role_vectors pca axis_bundle transfer postprocess
```

## Expected outputs

The transfer stage writes:

- `results/pairwise_metrics.csv`
- `results/summary_by_method.csv`
- `results/plots/zero_shot_grouped_bars__auroc__14.png`
- `results/plots/zero_shot_grouped_bars__auroc__overview.png`

These are the baseline artifacts to compare the four axes before any greedy
pruning.
