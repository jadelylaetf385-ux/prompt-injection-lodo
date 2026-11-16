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
| `09_shortcut_and_error_analysis.py` | Strongly weighted n-grams and their source concentration, attacks missed by all detectors, pairwise LODO comparison | `results/analysis/` |
| `10_equal_data_and_latency.py` | Word TF-IDF retrained under the fine-tuning cap (equal-data control) and single-text CPU latency | `results/equal_data_latency/` |

### Revision scripts (response to the reviewers)

| Script | Purpose | Main outputs |
|---|---|---|
| `07_bootstrap_ci.py` (updated) | Also records every source-level draw, skipped draws and the share of non-negative gaps | `results/bootstrap/source_bootstrap_draws.csv` |
| `08b_finetune_robustness.py` | Encoder with other seeds / more epochs and a 10% training-source validation split (GPU, `colab_revision.ipynb`) | `results/finetune_runs/<run>/` |
| `11_guardrails_native_length.py` | Token-length statistics, truncation side and guardrail scores at 512 tokens (GPU) | `results/offtheshelf_native/` |
| `12_revision_reanalysis.py` | Within-source AUC, exact pair decomposition of pooled AUC, training-only calibration sensitivity, identical-sample comparison with paired bootstrap, machine-readable results table | `results/revision/` |
| `13_shortcut_interventions.py` | Target masking, length matching and mixed-only training under LODO | `results/interventions/` |
| `14_low_fpr_operating_points.py` | Thresholds for 1% / 5% FPR chosen on training sources only, Wilson intervals | `results/operating_points/` |
| `15_fewshot_disjoint.py` | Few-shot adaptation with template- and cluster-disjoint pool/test splits | `results/fewshot_disjoint/` |
| `16_reproducibility_manifest.py` | Environment, Hugging Face revisions, raw-file hashes, per-record manifest, fold/adaptation indices, per-record scores | `results/repro/` |
| `17_label_audit.py` | Stratified sample for the manual label audit (`--make`) and agreement / sensitivity analysis (`--score`) | `results/label_audit/` |

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

A seed of 42 is used throughout. Hardware by component:

* Scripts 01–07, 09, 10 and 12: Intel Core i7-1165G7 laptop with 24 GB of memory, CPU only.
* Scripts 08, 08b and 11: Google Colab, NVIDIA T4 GPU. Script 08 runs 16 trainings: 5 random folds, 10 LODO folds and 1 run on the whole capped pool used only for NotInject.
* Scripts 13–15 (revision): two-core cloud CPU instance, same code and seed.

`results/repro/` (script 16) lets a reader check that the same records are paired under both protocols without access to the texts:

| File | Content |
|---|---|
| `environment.json`, `pip_freeze.txt` | Python, platform and package versions |
| `hub_revisions.csv` | Commit hash of every dataset and model repository on the Hugging Face hub |
| `raw_file_hashes.csv` | SHA-256 of every downloaded raw file |
| `record_manifest.csv.gz` (release asset) | Every unique text (SHA-1 of the normalised, lower-cased text) with its fate: within- or cross-source label conflict, duplicate kept in another source, removed by the cap, or in the corpus |
| `corpus_rebuild_check.json` | Confirms that the rebuilt corpus equals `data/corpus.parquet` |
| `fold_indices.csv.gz`, `capped_training_keys.csv.gz`, `fewshot_indices.csv.gz` | Random-split folds, the capped training sets of the encoder, few-shot pools/test parts and draws |
| `per_record_scores.parquet` (release asset) | Key, source, label, protocol, detector and score for every trained detector |
| `settings.json` | Model, solver, regularisation and training settings of every detector |

Texts are never written to these files. `per_record_scores.parquet` and `record_manifest.csv.gz` are too large for the repository and are attached to the release `v1.1-revision` as downloadable assets. The weights of the fine-tuned encoder were not retained.

The version of the code used for the paper is the GitHub release `v1.1-revision`.

## Licence

MIT (code only). See `LICENSE`.
