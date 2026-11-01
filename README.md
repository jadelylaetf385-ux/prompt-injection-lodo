# When Random Splits Mislead: LODO Evaluation of Prompt Injection Detectors

Code for the paper *"When Random Splits Mislead: Leave-One-Dataset-Out Evaluation of Lightweight Black-Box Prompt Injection Detectors"* (A. A. Dawah and A. D. Abbas, submitted to the International Journal of Intelligent Engineering and Systems).

The pipeline merges ten public prompt injection and instruction datasets into a deduplicated corpus. It then compares random cross-validation with leave-one-dataset-out (LODO) evaluation for lightweight black-box detectors, and studies why performance drops on unseen sources.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
hf auth login                   # needed for gated datasets (HackAPrompt)
```

Scripts 01–07 run on a CPU. Script 08 needs a GPU; `colab_finetune.ipynb` runs it on Google Colab.

## Pipeline

| Script | Purpose | Main outputs |
|---|---|---|
| `01_inventory_datasets.py` | Download the candidate datasets from the Hugging Face hub and describe them | `data/raw/`, `reports/inventory.md` |
| `02_build_corpus.py` | Select user-controlled text, deduplicate within and across sources, remove label conflicts, cap 20,000 texts per source and label | `data/corpus.parquet`, `reports/overlap_matrix.csv`, `reports/cleaning_audit.csv` |
| `03_lodo_experiments.py` | Surface, word TF-IDF, character TF-IDF and MiniLM detectors under random 5-fold CV and LODO; NotInject false positives; source identification | `results/main/`, `results/minilm/` |
| `04_offtheshelf_detectors.py` | Released guardrails (ProtectAI v2, deepset, PIGuard, Prompt Guard 2) on every source | `results/offtheshelf/` |
| `05_mitigations.py` | Source balancing, normalisation, source-adversarial pruning, label-free threshold | `results/mitigations/` |
| `06_fewshot_adaptation.py` | Add k labelled examples of the held-out source (k = 0, 10, 50, 200, 1000) | `results/fewshot/` |
| `07_bootstrap_ci.py` | Paired bootstrap confidence intervals for the random-versus-LODO gap | `results/bootstrap/` |
| `08_finetune_encoder.py` | Fine-tuned transformer encoder under the same two protocols (GPU) | `results/finetune/` |

Run them in order, for example `python scripts/03_lodo_experiments.py`. Each script documents its options in its header (`--help`).

To run the MiniLM detector separately: `python scripts/03_lodo_experiments.py --detectors minilm_lr --run-name minilm`.

## Data

All data come from public repositories and are subject to their own licences and access conditions; the corpus is rebuilt locally by script 02 and is not redistributed here.

| Source | Repository |
|---|---|
| deepset | `deepset/prompt-injections` |
| safeguard | `xTRam1/safe-guard-prompt-injection` |
| SPML | `reshabhs/SPML_Chatbot_Prompt_Injection` |
| jayavibhav | `jayavibhav/prompt-injection` |
| Gandalf | `Lakera/gandalf_ignore_instructions` |
| Mosscap | `Lakera/mosscap_prompt_injection` |
| HackAPrompt | `hackaprompt/hackaprompt-dataset` (gated) |
| LLMail-Inject | `microsoft/llmail-inject-challenge` |
| Alpaca | `tatsu-lab/alpaca` |
| Dolly | `databricks/databricks-dolly-15k` |
| NotInject (evaluation only) | `leolee99/NotInject` |

## Reproducibility

A seed of 42 is used throughout. The experiments in the paper were run on an Intel Core i7-1165G7 laptop with 24 GB of memory (CPU only), except script 08.

## Licence

MIT (code only). See `LICENSE`.
