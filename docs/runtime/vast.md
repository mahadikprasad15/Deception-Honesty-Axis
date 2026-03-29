# Vast.ai Workflow

## Recommended Use

Use Vast.ai for the full rollout and activation jobs once the pipeline is smoke-tested in Colab.

## Instance Selection

Pick the lowest-cost offer that still meets these constraints:

- secure cloud
- Docker runtime
- Jupyter or SSH access
- at least 24 GB VRAM
- at least 150 GB local disk

## Initial Setup

Inside the instance:

```bash
cd /workspace
git clone https://github.com/mahadikprasad15/Deception-Honesty-Axis.git
cd Deception-Honesty-Axis
python3 -m pip install -r requirements.txt
```

If you plan to push artifacts to HF from the instance, export your token first:

```bash
export HF_TOKEN=...
huggingface-cli login --token "$HF_TOKEN"
```

## Long-Run Commands

```bash
PYTHONPATH=src python3 scripts/generate_rollouts.py --batch-size 4
PYTHONPATH=src python3 scripts/extract_activations.py
```

Then sync artifacts:

```bash
PYTHONPATH=src python3 scripts/sync_hf_artifacts.py --direction push --local-dir artifacts
```

## Resume

If the instance stops or you relaunch on another host:

- pull the latest artifacts from HF
- rerun `scripts/audit_corpus.py`
- rerun the interrupted stage

The stage scripts are append-only and skip indexed `item_id`s automatically.
