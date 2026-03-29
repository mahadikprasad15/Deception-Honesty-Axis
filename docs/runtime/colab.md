# Colab Workflow

## Recommended Use

Use Colab for:

- mounting Drive
- cloning or opening the repo
- syncing upstream source data
- running the smoke-test subset
- inspecting the corpus and analysis outputs

Do not treat Drive as the canonical artifact store for long runs with many shard files. Keep the growing corpus in `artifacts/` and sync it to HF after stage boundaries.

## Minimal Bootstrap

```python
from google.colab import drive
drive.mount("/content/drive")
```

```bash
cd /content
git clone https://github.com/mahadikprasad15/Deception-Honesty-Axis.git
cd Deception-Honesty-Axis
python3 -m pip install -r requirements.txt
```

## First Commands

```bash
PYTHONPATH=src python3 scripts/sync_source_data.py
PYTHONPATH=src python3 scripts/audit_corpus.py
```

For a smoke test, temporarily lower the subset counts in the config or add a throwaway config copy before running generation and activations.

## Validation

- run `scripts/audit_corpus.py` before and after each stage
- inspect `artifacts/corpora/.../meta/coverage.json`
- push the updated `artifacts/` subtree to HF when you finish a stage
