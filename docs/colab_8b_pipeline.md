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
/content/drive/MyDrive/Efficacy-of-ensemble-of-attention-probes/data/activations_fullprompt/meta-llama_Meta-Llama-3-8B-Instruct
```

## Setup

```bash
%%bash
set -euo pipefail
cd /content
git clone https://github.com/mahadikprasad15/Deception-Honesty-Axis.git
git clone https://github.com/mahadikprasad15/Efficacy-of-ensemble-of-attention-probes.git

export DHA_ROOT=/content/Deception-Honesty-Axis
export EFFICACY_ROOT=/content/Efficacy-of-ensemble-of-attention-probes

cd "$DHA_ROOT"
pip install -e .
pip install huggingface_hub safetensors scikit-learn matplotlib pandas

cd "$EFFICACY_ROOT"
git checkout residual-analysis
git pull origin residual-analysis
pip install -r requirements.txt
```

In Colab, mount Drive before running transfer evaluations:

```python
from google.colab import drive
drive.mount("/content/drive")
```

Pull only the HF artifacts needed for this run. Do not pull the whole artifact repo into Colab.

```bash
%%bash
set -euo pipefail
DHA_ROOT=/content/Deception-Honesty-Axis
cd "$DHA_ROOT"
python "$DHA_ROOT/scripts/sync_hf_artifacts.py" \
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
%%bash
set -euo pipefail
DHA_ROOT=/content/Deception-Honesty-Axis
cd "$DHA_ROOT"
python "$DHA_ROOT/scripts/evaluate_activation_row_transfer.py" \
  --config configs/probes/activation_row_transfer_deception_quantity_v2_llama32_3b.json \
  --run-id llama32-3b-deception-raw

python "$DHA_ROOT/scripts/evaluate_activation_row_transfer_subspace_baselines.py" \
  --config configs/probes/activation_row_transfer_deception_quantity_v2_llama32_3b.json \
  --k-values 1 2 3 \
  --random-seeds 11 23 37 41 53 \
  --run-id llama32-3b-deception-subspace-baselines
```

For 3B persona-PC projected deception probes, use the pulled 3B QuantityV2 bundle. Resolve the exact bundle run dir
from the selective HF pull:

```bash
%%bash
set -euo pipefail
DHA_ROOT=/content/Deception-Honesty-Axis
cd "$DHA_ROOT"
find artifacts/runs/role-axis-bundles \
  -path "*/results/axis_bundle.pt" \
  | grep -E "quantity.*v2|quantity-axis-v2" \
  | sort
```

Then pass the parent run directory to:

```bash
%%bash
set -euo pipefail
DHA_ROOT=/content/Deception-Honesty-Axis
cd "$DHA_ROOT"
AXIS_BUNDLE_PT=$(find artifacts/runs/role-axis-bundles \
  -path "*/results/axis_bundle.pt" \
  | grep -E "quantity.*v2|quantity-axis-v2" \
  | sort \
  | tail -n 1)

if [ -z "$AXIS_BUNDLE_PT" ]; then
  echo "No local 3B QuantityV2 axis bundle found. Re-run the HF pull with broader role-axis-bundle patterns."
  exit 1
fi

AXIS_BUNDLE_RUN_DIR=$(dirname "$(dirname "$AXIS_BUNDLE_PT")")
echo "Using axis bundle run dir: $AXIS_BUNDLE_RUN_DIR"

python "$DHA_ROOT/scripts/evaluate_activation_row_transfer_pc_projection_sweep.py" \
  --config configs/probes/activation_row_transfer_deception_quantity_v2_llama32_3b.json \
  --axis-bundle-run-dir "$AXIS_BUNDLE_RUN_DIR" \
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

Run this only after confirming the typed-deception source files exist under the Drive-backed `RAW_ROOT`.

```python
from pathlib import Path
import json
import subprocess
import sys
import time

EFFICACY_ROOT = Path("/content/Efficacy-of-ensemble-of-attention-probes")
MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
MODEL_DIR = MODEL.replace("/", "_")
ACTS_ROOT = Path("/content/drive/MyDrive/Efficacy-of-ensemble-of-attention-probes/data/activations_fullprompt")
RAW_ROOT = Path("/content/drive/MyDrive/Efficacy-of-ensemble-of-attention-probes/data/apollo_raw")

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
    typed_sources = {
        "Deception-ConvincingGame": (
            "Deception-ConvincingGame",
            ["convincing-game__*.jsonl", "convincing-game__*.parquet", "convincing-game__*.json"],
        ),
        "Deception-HarmPressureChoice": (
            "Deception-HarmPressureChoice",
            ["harm-pressure-choice__*.jsonl", "harm-pressure-choice__*.parquet", "harm-pressure-choice__*.json"],
        ),
        "Deception-InstructedDeception": (
            "Deception-InstructedDeception",
            ["instructed-deception__*.jsonl", "instructed_deception__*.jsonl"],
        ),
        "Deception-Mask": (
            "Deception-Mask",
            ["mask__*.jsonl", "mask__*.parquet", "mask__*.json"],
        ),
        "Deception-AILiar": (
            "Deception-AILiar",
            ["ai_liar/paired_examples.jsonl", "ai_liar/ai_liar__*.json", "ai_liar/*.json"],
        ),
        "Deception-Roleplaying": (
            "Deception-Roleplaying",
            ["roleplaying/dataset.yaml", "roleplaying/*.yaml", "dataset.yaml"],
        ),
    }
    if dataset not in typed_sources:
        return ["--dataset", dataset]

    base_dataset, patterns = typed_sources[dataset]
    matches: list[Path] = []
    for pattern in patterns:
        matches.extend(sorted(RAW_ROOT.rglob(pattern)))
    if not matches:
        raise FileNotFoundError(f"Missing source file for {dataset} under {RAW_ROOT}")
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

This cell uses pregenerated completions and labels when the source dataset provides them. In that mode Cerebras is
not used. `CEREBRAS_API_KEY` is only required for datasets that are prompts-only and need newly generated completions
to be judged. `HF_TOKEN` is still required for gated Llama weights.

After caching, sanity-check the 8B activation cache before running transfer evaluations:

```python
from pathlib import Path
import json
from collections import Counter

root = Path("/content/drive/MyDrive/Efficacy-of-ensemble-of-attention-probes/data/activations_fullprompt")
model_dir = "meta-llama_Meta-Llama-3-8B-Instruct"
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
%%bash
set -euo pipefail
DHA_ROOT=/content/Deception-Honesty-Axis
SRC_MODEL=meta-llama-llama-3-2-3b-instruct
DST_MODEL=meta-llama-meta-llama-3-8b-instruct

cd "$DHA_ROOT"
for AXIS in quantity-axis-v2 sycophancy-pilot-v1; do
  SRC="artifacts/corpora/$SRC_MODEL/assistant-axis/$AXIS"
  DST="artifacts/corpora/$DST_MODEL/assistant-axis/$AXIS"
  mkdir -p "$DST/indexes" "$DST/meta"
  cp -a "$SRC/rollouts" "$DST/"
  cp -a "$SRC/indexes/rollouts.jsonl" "$DST/indexes/"
  if [ -f "$SRC/meta/coverage.json" ]; then
    cp -a "$SRC/meta/coverage.json" "$DST/meta/source_3b_coverage.json"
  fi

  rm -rf "$DST/activations"
  rm -f "$DST/indexes/activations.jsonl"
  rm -f "$DST/meta/extract_activations_status.json"
  rm -f "$DST/checkpoints/extract_activations_progress.json"
done
```

Then run the 8B pipeline from activation extraction onward. Omit the `rollouts` stage unless you intentionally want
fresh 8B generations instead of fixed-text 8B activations.

The `activations` stage is the cache step for the 8B role-rollout activations. It reads the materialized rollout
index and writes layer-16 pooled response activations under:

```text
artifacts/corpora/meta-llama-meta-llama-3-8b-instruct/assistant-axis/quantity-axis-v2/activations/
artifacts/corpora/meta-llama-meta-llama-3-8b-instruct/assistant-axis/sycophancy-pilot-v1/activations/
```

The matching resume indexes and status files are:

```text
artifacts/corpora/meta-llama-meta-llama-3-8b-instruct/assistant-axis/<axis>/indexes/activations.jsonl
artifacts/corpora/meta-llama-meta-llama-3-8b-instruct/assistant-axis/<axis>/meta/extract_activations_status.json
artifacts/corpora/meta-llama-meta-llama-3-8b-instruct/assistant-axis/<axis>/meta/coverage.json
```

QuantityV2:

```bash
%%bash
set -euo pipefail
DHA_ROOT=/content/Deception-Honesty-Axis
cd "$DHA_ROOT"
python "$DHA_ROOT/scripts/run_variant_pipeline.py" \
  --experiment-config configs/experiments/quantity_axis_v2_llama3_8b_instruct.json \
  --probe-config configs/probes/role_axis_transfer_quantity_v2_cumulative_pc_sweep_llama3_8b_instruct.json \
  --run-id llama3-8b-quantity-v2 \
  --stages activations role_vectors pca axis_bundle transfer postprocess
```

Sycophancy pilot v1:

```bash
%%bash
set -euo pipefail
DHA_ROOT=/content/Deception-Honesty-Axis
cd "$DHA_ROOT"
python "$DHA_ROOT/scripts/run_variant_pipeline.py" \
  --experiment-config configs/experiments/sycophancy_pilot_v1_llama3_8b_instruct.json \
  --probe-config configs/probes/role_axis_transfer_sycophancy_pilot_v1_cumulative_pc_sweep_llama3_8b_instruct.json \
  --run-id llama3-8b-sycophancy-pilot-v1 \
  --stages activations role_vectors pca axis_bundle transfer postprocess
```

## Raw Transfer, Persona-PC Transfer, And Baselines

Run raw activation probes:

```bash
%%bash
set -euo pipefail
DHA_ROOT=/content/Deception-Honesty-Axis
cd "$DHA_ROOT"
python "$DHA_ROOT/scripts/evaluate_activation_row_transfer.py" \
  --config configs/probes/activation_row_transfer_deception_quantity_v2_llama3_8b_instruct.json \
  --run-id llama3-8b-deception-raw

python "$DHA_ROOT/scripts/evaluate_activation_row_transfer.py" \
  --config configs/probes/activation_row_transfer_sycophancy_pilot_v1_llama3_8b_instruct.json \
  --run-id llama3-8b-sycophancy-raw
```

The sycophancy role-axis pipeline builds the 8B role axis, but its `transfer` stage needs separate benchmark
activation rows for the matching model. Do not use the completion-split sycophancy config until those benchmark
activations exist. For the external sycophancy benchmarks, extract activation rows at layer 16, then generate an
activation-row transfer config from the completed extraction runs:

```bash
%%bash
set -euo pipefail
DHA_ROOT=/content/Deception-Honesty-Axis
cd "$DHA_ROOT"

python "$DHA_ROOT/scripts/extract_external_sycophancy_activations.py" \
  --source-repo-id arianaazarbal/sycophancy_dataset \
  --source-split train \
  --adapter ariana_sycophancy \
  --model-name meta-llama/Meta-Llama-3-8B-Instruct \
  --layer 16 \
  --pooling mean_response \
  --batch-size 4 \
  --target-name ariana-sycophancy-train \
  --run-id llama3-8b-ariana-sycophancy-train

python "$DHA_ROOT/scripts/extract_external_sycophancy_activations.py" \
  --source-repo-id henrypapadatos/Open-ended_sycophancy \
  --source-split train \
  --adapter open_ended_sycophancy \
  --model-name meta-llama/Meta-Llama-3-8B-Instruct \
  --layer 16 \
  --pooling mean_response \
  --batch-size 4 \
  --target-name open-ended-sycophancy-train \
  --run-id llama3-8b-open-ended-sycophancy-train
```

For OEQ, first prepare/pull the three prepared HF datasets, then run the same extraction script with
`--adapter oeq_prepared` and target names `oeq-validation-human`, `oeq-indirectness-human`, and
`oeq-framing-human`. After all available extraction runs complete:

```bash
%%bash
set -euo pipefail
DHA_ROOT=/content/Deception-Honesty-Axis
cd "$DHA_ROOT"

python "$DHA_ROOT/scripts/prepare_sycophancy_activation_row_transfer_config.py" \
  --model-name meta-llama/Meta-Llama-3-8B-Instruct \
  --expected-layer-number 16 \
  --output-config configs/probes/activation_row_transfer_sycophancy_pilot_v1_llama3_8b_external.json \
  --allow-missing
```

Use the generated `activation_row_transfer_sycophancy_pilot_v1_llama3_8b_external.json` config for 8B sycophancy
raw/projected/subspace transfer. If `--allow-missing` skipped OEQ targets, the run is a partial sycophancy run.

Run projected probes for `k=1,2,3`:

```bash
%%bash
set -euo pipefail
DHA_ROOT=/content/Deception-Honesty-Axis
cd "$DHA_ROOT"
python "$DHA_ROOT/scripts/evaluate_activation_row_transfer_pc_projection_sweep.py" \
  --config configs/probes/activation_row_transfer_deception_quantity_v2_llama3_8b_instruct.json \
  --axis-bundle-run-dir artifacts/runs/role-axis-bundles/meta-llama-meta-llama-3-8b-instruct/assistant-axis/quantity-axis-v2-cumulative-pc-sweep/mean_response/llama3-8b-quantity-v2 \
  --k-values 1 2 3 \
  --run-id llama3-8b-deception-qv2-pc-sweep

python "$DHA_ROOT/scripts/evaluate_activation_row_transfer_pc_projection_sweep.py" \
  --config configs/probes/activation_row_transfer_sycophancy_pilot_v1_llama3_8b_instruct.json \
  --axis-bundle-run-dir artifacts/runs/role-axis-bundles/meta-llama-meta-llama-3-8b-instruct/assistant-axis/sycophancy-pilot-v1-cumulative-pc-sweep/mean_response/llama3-8b-sycophancy-pilot-v1 \
  --k-values 1 2 3 \
  --run-id llama3-8b-sycophancy-pc-sweep
```

Run dataset-PCA and random orthonormal subspace baselines:

```bash
%%bash
set -euo pipefail
DHA_ROOT=/content/Deception-Honesty-Axis
cd "$DHA_ROOT"
python "$DHA_ROOT/scripts/evaluate_activation_row_transfer_subspace_baselines.py" \
  --config configs/probes/activation_row_transfer_deception_quantity_v2_llama3_8b_instruct.json \
  --k-values 1 2 3 \
  --random-seeds 11 23 37 41 53 \
  --run-id llama3-8b-deception-subspace-baselines

python "$DHA_ROOT/scripts/evaluate_activation_row_transfer_subspace_baselines.py" \
  --config configs/probes/activation_row_transfer_sycophancy_pilot_v1_llama3_8b_instruct.json \
  --k-values 1 2 3 \
  --random-seeds 11 23 37 41 53 \
  --run-id llama3-8b-sycophancy-subspace-baselines
```

Analyze deltas against raw runs after substituting the resolved raw/subspace run directories:

```bash
%%bash
set -euo pipefail
DHA_ROOT=/content/Deception-Honesty-Axis
cd "$DHA_ROOT"
python "$DHA_ROOT/scripts/analyze_activation_row_transfer_subspace_baselines.py" \
  --baseline-run-dir PATH_TO_RAW_TRANSFER_RUN \
  --subspace-run-dir PATH_TO_SUBSPACE_BASELINE_RUN \
  --run-id llama3-8b-deception-subspace-analysis
```

## Upload Artifacts

```bash
%%bash
set -euo pipefail
DHA_ROOT=/content/Deception-Honesty-Axis
cd "$DHA_ROOT"
python "$DHA_ROOT/scripts/sync_hf_artifacts.py" \
  --direction push \
  --repo-id Prasadmahadik/deception-honesty-axis-artifacts \
  --local-dir artifacts
```
