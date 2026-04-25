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

Pull only the HF artifacts needed for this run. Do not pull the whole artifact repo into Colab.

```bash
python scripts/sync_hf_artifacts.py \
  --direction pull \
  --repo-id Prasadmahadik/deception-honesty-axis-artifacts \
  --local-dir artifacts \
  --allow-patterns \
    "runs/role-axis-bundles/meta-llama-llama-3-2-3b-instruct/assistant-axis/quantity-axis-v2-cumulative-pc-sweep/**" \
    "runs/role-axis-bundles/meta-llama-llama-3-2-3b-instruct/assistant-axis/sycophancy-pilot-v1-cumulative-pc-sweep/**" \
    "runs/external-sycophancy-activation-extraction/**" \
    "corpora/meta-llama-llama-3-2-3b-instruct/assistant-axis/quantity-axis-v2/rollouts/**" \
    "corpora/meta-llama-llama-3-2-3b-instruct/assistant-axis/quantity-axis-v2/indexes/**" \
    "corpora/meta-llama-llama-3-2-3b-instruct/assistant-axis/quantity-axis-v2/meta/**" \
    "corpora/meta-llama-llama-3-2-3b-instruct/assistant-axis/sycophancy-pilot-v1/rollouts/**" \
    "corpora/meta-llama-llama-3-2-3b-instruct/assistant-axis/sycophancy-pilot-v1/indexes/**" \
    "corpora/meta-llama-llama-3-2-3b-instruct/assistant-axis/sycophancy-pilot-v1/meta/**"
```

The role/question manifests and config JSON files are tracked in git under `data/` and `configs/`, so they do not need to be pulled from HF.

## Build 8B Role Axes

The selective HF pull above fetches the fixed 3B rollouts for both axes. To reuse those exact prompts/responses
for 8B activation extraction, materialize them under the 8B corpus path:

```bash
RUN_ID_8B_QV2=llama31-8b-quantity-v2
RUN_ID_8B_SYC=llama31-8b-sycophancy-pilot-v1

SRC_MODEL=meta-llama-llama-3-2-3b-instruct
DST_MODEL=meta-llama-meta-llama-3-1-8b-instruct

for AXIS in quantity-axis-v2 sycophancy-pilot-v1; do
  SRC="artifacts/corpora/$SRC_MODEL/assistant-axis/$AXIS"
  DST="artifacts/corpora/$DST_MODEL/assistant-axis/$AXIS"
  mkdir -p "$DST"
  cp -a "$SRC/rollouts" "$DST/"
  cp -a "$SRC/indexes" "$DST/"
  if [ -d "$SRC/meta" ]; then cp -a "$SRC/meta" "$DST/"; fi
done
```

Then run the 8B pipeline from activation extraction onward. Omit the `rollouts` stage unless you intentionally want
fresh 8B generations instead of fixed-text 8B activations.

QuantityV2:

```bash
python scripts/run_variant_pipeline.py \
  --experiment-config configs/experiments/quantity_axis_v2_llama31_8b_instruct.json \
  --probe-config configs/probes/role_axis_transfer_quantity_v2_cumulative_pc_sweep_llama31_8b_instruct.json \
  --run-id "$RUN_ID_8B_QV2" \
  --stages activations role_vectors pca axis_bundle transfer postprocess
```

Sycophancy pilot v1:

```bash
python scripts/run_variant_pipeline.py \
  --experiment-config configs/experiments/sycophancy_pilot_v1_llama31_8b_instruct.json \
  --probe-config configs/probes/role_axis_transfer_sycophancy_pilot_v1_cumulative_pc_sweep_llama31_8b_instruct.json \
  --run-id "$RUN_ID_8B_SYC" \
  --stages activations role_vectors pca axis_bundle transfer postprocess
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
