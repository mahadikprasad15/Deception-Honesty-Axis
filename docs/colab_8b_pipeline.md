# Colab 8B Pipeline Runbook

This runbook assumes Google Drive is mounted at `/content/drive` and the dataset activations are stored under:

```text
/content/drive/MyDrive/Efficacy-of-ensemble-of-attention-probes/data/activations_fullprompt/
```

The 8B model-specific activation root expected by the configs is:

```text
/content/drive/MyDrive/Efficacy-of-ensemble-of-attention-probes/data/activations_fullprompt/meta-llama_Meta-Llama-3.1-8B-Instruct
```

## Setup

```bash
git clone https://github.com/Prasadmahadik/Deception-Honesty-Axis.git
cd Deception-Honesty-Axis
pip install -e .
pip install huggingface_hub safetensors scikit-learn matplotlib pandas
```

In Colab, mount Drive before running transfer evaluations:

```python
from google.colab import drive
drive.mount("/content/drive")
```

Pull existing artifacts from HF:

```bash
python scripts/sync_hf_artifacts.py \
  --direction pull \
  --repo-id Prasadmahadik/deception-honesty-axis-artifacts \
  --local-dir artifacts
```

## Build 8B Role Axes

Use one shared run id per axis so role vectors, PCA, bundles, and zero-shot transfer line up.

```bash
RUN_ID_8B_QV2=llama31-8b-quantity-v2
RUN_ID_8B_SYC=llama31-8b-sycophancy-pilot-v1
```

QuantityV2:

```bash
python scripts/run_variant_pipeline.py \
  --experiment-config configs/experiments/quantity_axis_v2_llama31_8b_instruct.json \
  --probe-config configs/probes/role_axis_transfer_quantity_v2_cumulative_pc_sweep_llama31_8b_instruct.json \
  --run-id "$RUN_ID_8B_QV2" \
  --stages rollouts activations role_vectors pca axis_bundle transfer postprocess
```

Sycophancy pilot v1:

```bash
python scripts/run_variant_pipeline.py \
  --experiment-config configs/experiments/sycophancy_pilot_v1_llama31_8b_instruct.json \
  --probe-config configs/probes/role_axis_transfer_sycophancy_pilot_v1_cumulative_pc_sweep_llama31_8b_instruct.json \
  --run-id "$RUN_ID_8B_SYC" \
  --stages rollouts activations role_vectors pca axis_bundle transfer postprocess
```

## Raw Transfer, Persona-PC Transfer, And Baselines

Run raw activation probes:

```bash
python scripts/evaluate_activation_row_transfer.py \
  --config configs/probes/activation_row_transfer_deception_quantity_v2_llama31_8b_instruct.json \
  --run-id llama31-8b-deception-raw

python scripts/evaluate_activation_row_transfer.py \
  --config configs/probes/activation_row_transfer_sycophancy_pilot_v1_llama31_8b_instruct.json \
  --run-id llama31-8b-sycophancy-raw
```

Run projected probes for `k=1,2,3`:

```bash
python scripts/evaluate_activation_row_transfer_pc_projection_sweep.py \
  --config configs/probes/activation_row_transfer_deception_quantity_v2_llama31_8b_instruct.json \
  --axis-bundle-run-dir artifacts/runs/role-axis-bundles/meta-llama-meta-llama-3-1-8b-instruct/assistant-axis/quantity-axis-v2-cumulative-pc-sweep/mean_response/$RUN_ID_8B_QV2 \
  --k-values 1 2 3 \
  --run-id llama31-8b-deception-qv2-pc-sweep

python scripts/evaluate_activation_row_transfer_pc_projection_sweep.py \
  --config configs/probes/activation_row_transfer_sycophancy_pilot_v1_llama31_8b_instruct.json \
  --axis-bundle-run-dir artifacts/runs/role-axis-bundles/meta-llama-meta-llama-3-1-8b-instruct/assistant-axis/sycophancy-pilot-v1-cumulative-pc-sweep/mean_response/$RUN_ID_8B_SYC \
  --k-values 1 2 3 \
  --run-id llama31-8b-sycophancy-pc-sweep
```

Run dataset-PCA and random orthonormal subspace baselines:

```bash
python scripts/evaluate_activation_row_transfer_subspace_baselines.py \
  --config configs/probes/activation_row_transfer_deception_quantity_v2_llama31_8b_instruct.json \
  --k-values 1 2 3 \
  --random-seeds 11 23 37 41 53 \
  --run-id llama31-8b-deception-subspace-baselines

python scripts/evaluate_activation_row_transfer_subspace_baselines.py \
  --config configs/probes/activation_row_transfer_sycophancy_pilot_v1_llama31_8b_instruct.json \
  --k-values 1 2 3 \
  --random-seeds 11 23 37 41 53 \
  --run-id llama31-8b-sycophancy-subspace-baselines
```

Analyze deltas against raw runs after substituting the resolved raw/subspace run directories:

```bash
python scripts/analyze_activation_row_transfer_subspace_baselines.py \
  --baseline-run-dir PATH_TO_RAW_TRANSFER_RUN \
  --subspace-run-dir PATH_TO_SUBSPACE_BASELINE_RUN \
  --run-id llama31-8b-deception-subspace-analysis
```

## Upload Artifacts

```bash
python scripts/sync_hf_artifacts.py \
  --direction push \
  --repo-id Prasadmahadik/deception-honesty-axis-artifacts \
  --local-dir artifacts
```
