# Colab 8B Pipeline Runbook

This runbook has two jobs:

1. Build and evaluate the new 8B QuantityV2 / Sycophancy pilot axes.
2. Re-run the stronger transfer baselines for the existing 3B setup using already-cached 3B artifacts.

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
cd /content
git clone https://github.com/mahadikprasad15/Deception-Honesty-Axis.git
git clone https://github.com/mahadikprasad15/Efficacy-of-ensemble-of-attention-probes.git

export DHA_ROOT=/content/Deception-Honesty-Axis
export EFFICACY_ROOT=/content/Efficacy-of-ensemble-of-attention-probes

cd "$DHA_ROOT"
pip install -e .
pip install huggingface_hub safetensors scikit-learn matplotlib pandas

cd "$EFFICACY_ROOT"
pip install -r requirements.txt
```

In Colab, mount Drive before running transfer evaluations:

```python
from google.colab import drive
drive.mount("/content/drive")
```

Pull only the HF artifacts needed for this run. Do not pull the whole artifact repo into Colab.

```bash
cd "$DHA_ROOT"
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

## 3B Baselines From Existing Activations

The new dataset-PCA and random-subspace baselines are model-agnostic. For 3B, do not regenerate deception dataset
activations if the existing Drive cache is present. Use the existing 3B completion splits directly:

```bash
cd "$DHA_ROOT"
python scripts/evaluate_activation_row_transfer.py \
  --config configs/probes/activation_row_transfer_deception_quantity_v2_llama32_3b.json \
  --run-id llama32-3b-deception-raw

python scripts/evaluate_activation_row_transfer_subspace_baselines.py \
  --config configs/probes/activation_row_transfer_deception_quantity_v2_llama32_3b.json \
  --k-values 1 2 3 \
  --random-seeds 11 23 37 41 53 \
  --run-id llama32-3b-deception-subspace-baselines
```

For 3B persona-PC projected deception probes, use the pulled 3B QuantityV2 bundle. Resolve the exact bundle run dir
from the selective HF pull:

```bash
cd "$DHA_ROOT"
find artifacts/runs/role-axis-bundles/meta-llama-llama-3-2-3b-instruct/assistant-axis/quantity-axis-v2-cumulative-pc-sweep \
  -path "*/results/axis_bundle.pt" -print
```

Then pass the parent run directory to:

```bash
python scripts/evaluate_activation_row_transfer_pc_projection_sweep.py \
  --config configs/probes/activation_row_transfer_deception_quantity_v2_llama32_3b.json \
  --axis-bundle-run-dir PATH_TO_3B_QUANTITY_V2_BUNDLE_RUN_DIR \
  --k-values 1 2 3 \
  --run-id llama32-3b-deception-qv2-pc-sweep
```

## Generate 8B Dataset Activations

The 8B deception transfer evaluations need Drive-backed dataset activations before any zero-shot, raw-probe,
persona-PC, dataset-PCA, or random-subspace evaluation can run.

Use the Efficacy repo to cache the six deception completion splits into the exact root expected by this repo's
8B configs:

This is the same loader route used by the Efficacy repo's 3B principled cross-dataset runbook:

- `Deception-AILiar`: `DeceptionAILiarDataset`, pregenerated paired honest/deceptive completions.
- `Deception-Roleplaying`: `DeceptionRoleplayingDataset` with `--use_gold_completions`.
- `Deception-InstructedDeception`: explicit typed-message loader.
- `Deception-Mask`: explicit typed-message loader with fail-closed Mask file resolution.
- `Deception-ConvincingGame`: typed-message loader through `--dataset_file`, using the `convincing-game__*` source file.
- `Deception-HarmPressureChoice`: typed-message loader through `--dataset_file`, using the `harm-pressure-choice__*` source file.

Run this only after confirming the typed-deception source files exist under `EFFICACY_ROOT/data/apollo_raw`.

```python
from pathlib import Path
import json
import subprocess
import sys
import time

EFFICACY_ROOT = Path("/content/Efficacy-of-ensemble-of-attention-probes")
MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"
MODEL_DIR = MODEL.replace("/", "_")
ACTS_ROOT = Path("/content/drive/MyDrive/Efficacy-of-ensemble-of-attention-probes/data/activations_fullprompt")
RAW_ROOT = EFFICACY_ROOT / "data" / "apollo_raw"

SOURCE_DATASETS = [
    "Deception-ConvincingGame",
    "Deception-HarmPressureChoice",
    "Deception-InstructedDeception",
    "Deception-Mask",
    "Deception-AILiar",
    "Deception-Roleplaying",
]
SPLITS = ["train", "test"]
L_PRIME = 32
T_PRIME = 64
CACHE_BATCH_SIZE = 2

def has_manifest_and_shards(path: Path) -> bool:
    return (path / "manifest.jsonl").exists() and any(path.glob("shard_*.safetensors"))

def resolve_cache_dataset_args(dataset: str) -> list[str]:
    if dataset == "Deception-ConvincingGame":
        patterns = ["convincing-game__*.jsonl", "convincing-game__*.parquet", "convincing-game__*.json"]
        base_dataset = "Deception-InstructedDeception"
    elif dataset == "Deception-HarmPressureChoice":
        patterns = ["harm-pressure-choice__*.jsonl", "harm-pressure-choice__*.parquet", "harm-pressure-choice__*.json"]
        base_dataset = "Deception-InstructedDeception"
    else:
        return ["--dataset", dataset]

    matches = []
    for pattern in patterns:
        matches.extend(sorted(RAW_ROOT.rglob(pattern)))
    if not matches:
        raise FileNotFoundError(f"Missing typed-deception source file for {dataset} under {RAW_ROOT}")
    dataset_file = matches[0]
    print(f"[dataset] {dataset} uses {dataset_file}")
    return ["--dataset", base_dataset, "--dataset_file", str(dataset_file)]

def run_stream(label: str, cmd: list[str]) -> None:
    print(f"[start] {label}")
    print("[cmd]", " ".join(cmd))
    t0 = time.time()
    proc = subprocess.Popen(
        cmd,
        cwd=str(EFFICACY_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="")
    rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"{label} failed with exit code {rc}")
    print(f"[done] {label} elapsed={time.time() - t0:.1f}s")

total_units = len(SOURCE_DATASETS) * len(SPLITS)
unit_idx = 0
for dataset in SOURCE_DATASETS:
    for split in SPLITS:
        unit_idx += 1
        out_dir = ACTS_ROOT / MODEL_DIR / f"{dataset}-completion" / split
        label = f"[8b dataset activations {unit_idx}/{total_units}] {dataset}-completion {split}"
        if has_manifest_and_shards(out_dir):
            print(f"[skip] {label} already exists at {out_dir}")
            continue

        cmd = [
            sys.executable,
            "-u",
            "scripts/data/cache_deception_activations.py",
            "--model", MODEL,
            "--split", split,
            "--dataset_output_name", f"{dataset}-completion",
            "--output_dir", str(ACTS_ROOT),
            "--L_prime", str(L_PRIME),
            "--T_prime", str(T_PRIME),
            "--batch_size", str(CACHE_BATCH_SIZE),
        ]
        cmd.extend(resolve_cache_dataset_args(dataset))
        if dataset == "Deception-Roleplaying":
            cmd.append("--use_gold_completions")

        run_stream(label, cmd)
```

This cell will generate model completions and Cerebras labels for datasets that do not already have pregenerated
gold completions. Ensure `HF_TOKEN` and `CEREBRAS_API_KEY` are set in the Colab environment before running it.

After caching, sanity-check the 8B activation cache before running transfer evaluations:

```python
from pathlib import Path
import json
from collections import Counter

root = Path("/content/drive/MyDrive/Efficacy-of-ensemble-of-attention-probes/data/activations_fullprompt")
model_dir = "meta-llama_Meta-Llama-3.1-8B-Instruct"
for dataset in SOURCE_DATASETS:
    for split in SPLITS:
        path = root / model_dir / f"{dataset}-completion" / split
        manifest = path / "manifest.jsonl"
        shards = sorted(path.glob("shard_*.safetensors"))
        rows = [json.loads(line) for line in manifest.open()] if manifest.exists() else []
        labels = Counter(int(row["label"]) for row in rows if "label" in row)
        print(dataset, split, "rows=", len(rows), "labels=", dict(labels), "shards=", len(shards))
        assert rows and shards, path
        assert set(labels).issubset({0, 1}) and len(labels) == 2, (path, labels)
```

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
cd "$DHA_ROOT"
python scripts/run_variant_pipeline.py \
  --experiment-config configs/experiments/quantity_axis_v2_llama31_8b_instruct.json \
  --probe-config configs/probes/role_axis_transfer_quantity_v2_cumulative_pc_sweep_llama31_8b_instruct.json \
  --run-id "$RUN_ID_8B_QV2" \
  --stages activations role_vectors pca axis_bundle transfer postprocess
```

Sycophancy pilot v1:

```bash
cd "$DHA_ROOT"
python scripts/run_variant_pipeline.py \
  --experiment-config configs/experiments/sycophancy_pilot_v1_llama31_8b_instruct.json \
  --probe-config configs/probes/role_axis_transfer_sycophancy_pilot_v1_cumulative_pc_sweep_llama31_8b_instruct.json \
  --run-id "$RUN_ID_8B_SYC" \
  --stages activations role_vectors pca axis_bundle transfer postprocess
```

## Raw Transfer, Persona-PC Transfer, And Baselines

Run raw activation probes:

```bash
cd "$DHA_ROOT"
python scripts/evaluate_activation_row_transfer.py \
  --config configs/probes/activation_row_transfer_deception_quantity_v2_llama31_8b_instruct.json \
  --run-id llama31-8b-deception-raw

python scripts/evaluate_activation_row_transfer.py \
  --config configs/probes/activation_row_transfer_sycophancy_pilot_v1_llama31_8b_instruct.json \
  --run-id llama31-8b-sycophancy-raw
```

The sycophancy activation-transfer configs require sycophancy activation rows for the matching model. For 3B, those
are pulled from HF under `runs/external-sycophancy-activation-extraction/**`. For 8B, run
`scripts/extract_external_sycophancy_activations.py` for each sycophancy source at layer 16 before running the
8B sycophancy raw/projected/subspace commands, or update the sycophancy transfer config paths to point at the
generated `results/records.jsonl` files.

Run projected probes for `k=1,2,3`:

```bash
cd "$DHA_ROOT"
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
cd "$DHA_ROOT"
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
cd "$DHA_ROOT"
python scripts/analyze_activation_row_transfer_subspace_baselines.py \
  --baseline-run-dir PATH_TO_RAW_TRANSFER_RUN \
  --subspace-run-dir PATH_TO_SUBSPACE_BASELINE_RUN \
  --run-id llama31-8b-deception-subspace-analysis
```

## Upload Artifacts

```bash
cd "$DHA_ROOT"
python scripts/sync_hf_artifacts.py \
  --direction push \
  --repo-id Prasadmahadik/deception-honesty-axis-artifacts \
  --local-dir artifacts
```
