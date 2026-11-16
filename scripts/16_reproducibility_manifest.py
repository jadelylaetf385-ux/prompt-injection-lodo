"""Reproducibility package for the revision (run on the machine that holds the
raw data).

Writes results/repro/ with:

environment.json             Python, platform, CPU, key package versions
pip_freeze.txt               full package list
hub_revisions.csv            dataset and model repositories with the revision
                             found in the local Hugging Face cache and the
                             current remote revision
raw_file_hashes.csv          SHA-256 of every raw Parquet file under data/raw
record_manifest.csv.gz       every unique (source, text-hash) record after
                             whitespace normalisation, with its fate:
                             within_source_label_conflict |
                             cross_source_label_conflict |
                             duplicate_kept_in_<source> | removed_by_cap |
                             removed_overlaps_training_pool | in_corpus
record_manifest_summary.csv  record counts per source and fate
corpus_rebuild_check.json    whether the rebuilt corpus equals
                             data/corpus.parquet
fold_indices.csv.gz          text hash -> random-split fold (1-5)
capped_training_keys.csv.gz  text hashes used to train each fold of the
                             fine-tuned encoder (cap 3,000 per source/label)
fewshot_indices.csv.gz       pool/test assignment and drawn examples of
                             script 06
per_record_scores.parquet    text hash, source, label, protocol, detector,
                             score for every trained detector
settings.json                model and training settings of every detector

Text is never written, only SHA-1 hashes of the normalised, lower-cased text,
so the package can be published without redistributing the datasets.

Usage
-----
    python scripts/16_reproducibility_manifest.py
    python scripts/16_reproducibility_manifest.py --skip-hub
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RES = ROOT / "results"
OUT = RES / "repro"
SEED = 42

DATASETS = {
    "deepset": "deepset/prompt-injections", "safeguard": "xTRam1/safe-guard-prompt-injection",
    "spml": "reshabhs/SPML_Chatbot_Prompt_Injection", "jayavibhav": "jayavibhav/prompt-injection",
    "gandalf": "Lakera/gandalf_ignore_instructions", "mosscap": "Lakera/mosscap_prompt_injection",
    "hackaprompt": "hackaprompt/hackaprompt-dataset", "llmail": "microsoft/llmail-inject-challenge",
    "alpaca": "tatsu-lab/alpaca", "dolly": "databricks/databricks-dolly-15k", "notinject": "leolee99/NotInject",
}
MODELS = ["protectai/deberta-v3-base-prompt-injection-v2", "deepset/deberta-v3-base-injection", "leolee99/PIGuard",
          "distilroberta-base", "sentence-transformers/all-MiniLM-L6-v2"]
SCORES = {"surface": "main/scores_surface.parquet", "word_lr": "main/scores_word_lr.parquet",
          "char_svm": "main/scores_char_svm.parquet", "minilm_lr": "minilm/scores_minilm_lr.parquet",
          "finetuned": "finetune/scores_finetuned.parquet"}
SETTINGS = {
    "surface": "StandardScaler + LogisticRegression(lbfgs, C=1.0, L2, max_iter=2000, class_weight='balanced'); "
               "six features: log1p chars, log1p words, upper/digit/punctuation shares, log1p newlines",
    "word_lr": "TfidfVectorizer(word 1-2, min_df=2, max_features=200000, sublinear_tf, lowercase) + "
               "LogisticRegression(liblinear, C=1.0, L2, max_iter=2000, class_weight='balanced'); threshold 0.5",
    "char_svm": "TfidfVectorizer(char_wb 3-5, min_df=3, max_features=300000, sublinear_tf, lowercase) + "
                "LinearSVC(C=0.5, class_weight='balanced', max_iter=5000); threshold 0 on the decision function",
    "minilm_lr": "all-MiniLM-L6-v2, max 256 tokens, L2-normalised 384-d embeddings + "
                 "LogisticRegression(lbfgs, C=1.0, max_iter=3000, class_weight='balanced'); threshold 0.5",
    "finetuned": "distilroberta-base, 1 epoch, AdamW lr 3e-5, weight decay 0.01, linear decay after 6% warm-up, "
                 "batch 32, max 256 tokens (right truncation), class-weighted cross-entropy, fp16, gradient clip 1.0, "
                 "seed 42; training cap 3,000 per source and label (sampling seed 42); no model selection",
    "transforms": "every vectoriser, IDF, scaler and pruning step is fitted on the training part of each fold only",
    "random_split": "StratifiedKFold(5, shuffle=True, random_state=42) on source x label strata",
}


def log(message: str) -> None:
    print(message, flush=True)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name.split("_")[0], ROOT / "scripts" / name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def environment() -> None:
    import sklearn, pandas, numpy, scipy  # noqa: E401
    env = {"python": sys.version, "platform": platform.platform(), "processor": platform.processor(),
           "cpu_count": os.cpu_count(),
           "packages": {m.__name__: m.__version__ for m in (sklearn, pandas, numpy, scipy)}}
    for name in ("torch", "transformers", "sentence_transformers", "datasets", "huggingface_hub"):
        try:
            env["packages"][name] = importlib.import_module(name).__version__
        except Exception:
            env["packages"][name] = "not installed"
    (OUT / "environment.json").write_text(json.dumps(env, indent=2), encoding="utf-8")
    freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True).stdout
    (OUT / "pip_freeze.txt").write_text(freeze, encoding="utf-8")
    log("environment written")


def hub_revisions() -> None:
    from huggingface_hub import HfApi, scan_cache_dir
    api, rows = HfApi(), []
    local = {}
    try:
        for repo in scan_cache_dir().repos:
            local[(repo.repo_type, repo.repo_id)] = ",".join(sorted(r.commit_hash for r in repo.revisions))
    except Exception:
        pass
    for kind, ids in (("dataset", DATASETS.values()), ("model", MODELS)):
        for rid in ids:
            try:
                info = api.dataset_info(rid) if kind == "dataset" else api.model_info(rid)
                remote = info.sha
            except Exception as error:
                remote = f"unavailable ({type(error).__name__})"
            rows.append({"type": kind, "repo": rid, "local_cache_revision": local.get((kind, rid), ""),
                         "remote_revision_now": remote})
    pd.DataFrame(rows).to_csv(OUT / "hub_revisions.csv", index=False)
    log("hub revisions written")


def raw_hashes() -> None:
    rows = [{"file": str(p.relative_to(DATA)), "bytes": p.stat().st_size, "sha256": sha256(p)}
            for p in sorted((DATA / "raw").rglob("*.parquet"))]
    pd.DataFrame(rows).to_csv(OUT / "raw_file_hashes.csv", index=False)
    log(f"raw file hashes: {len(rows)} files")


def record_manifest(corpus: pd.DataFrame, cap: int = 20000) -> None:
    """Repeat the cleaning steps of script 02 and record the fate of every
    unique (source, text) record. The rebuilt corpus is checked against
    data/corpus.parquet."""
    build = load_script("02_build_corpus.py")
    frames, fate = [], []
    for name, (loader, role) in build.SOURCES.items():
        df = loader()
        df["text"] = df["text"].map(build.normalise)
        df = df[df["text"].str.len() >= build.MIN_CHARS]
        df["key"] = df["text"].map(build.key)
        n_labels = df.groupby("key")["label"].nunique()
        conflicted = set(n_labels[n_labels > 1].index)
        fate.append(df[df["key"].isin(conflicted)].drop_duplicates("key")[["key"]].assign(
            source=name, label=-1, role=role, status="within_source_label_conflict"))
        df = df[~df["key"].isin(conflicted)].drop_duplicates("key")
        df["role"] = role
        frames.append(df)
        log(f"  manifest: {name} {len(df):,} unique records")
    data = pd.concat(frames, ignore_index=True)
    train = data[data["role"] == "train_pool"]
    label_sets = train.groupby("key")["label"].nunique()
    cross = set(label_sets[label_sets > 1].index)
    fate.append(train[train["key"].isin(cross)][["key", "source", "label", "role"]].assign(
        status="cross_source_label_conflict"))
    train = train[~train["key"].isin(cross)]
    size = train["source"].value_counts()
    train = train.assign(_size=train["source"].map(size)).sort_values(["key", "_size", "source"])
    resolved = train.drop_duplicates("key", keep="first").drop(columns="_size")
    owner = resolved.set_index("key")["source"]
    dup = train.drop(columns="_size").merge(resolved[["key", "source"]], on=["key", "source"], how="left",
                                            indicator=True)
    dup = dup[dup["_merge"] == "left_only"]
    fate.append(dup[["key", "source", "label", "role"]].assign(
        status="duplicate_kept_in_" + dup["key"].map(owner)))
    shuffled = resolved.sample(frac=1.0, random_state=SEED)
    rank = shuffled.groupby(["source", "label"]).cumcount()
    capped = shuffled[rank < cap]
    fate.append(shuffled[rank >= cap][["key", "source", "label", "role"]].assign(status="removed_by_cap"))
    fate.append(capped[["key", "source", "label", "role"]].assign(status="in_corpus"))
    ev = data[data["role"] == "eval_only"]
    in_train = ev["key"].isin(set(capped["key"]))
    fate.append(ev[in_train][["key", "source", "label", "role"]].assign(status="removed_overlaps_training_pool"))
    fate.append(ev[~in_train][["key", "source", "label", "role"]].assign(status="in_corpus"))
    m = pd.concat(fate, ignore_index=True)
    rebuilt = set(zip(m.loc[m["status"] == "in_corpus", "source"], m.loc[m["status"] == "in_corpus", "key"]))
    stored = set(zip(corpus["source"], corpus["key"]))
    check = {"rebuilt_records": len(rebuilt), "stored_records": len(stored), "identical": rebuilt == stored,
             "only_in_rebuilt": len(rebuilt - stored), "only_in_stored": len(stored - rebuilt)}
    (OUT / "corpus_rebuild_check.json").write_text(json.dumps(check, indent=2), encoding="utf-8")
    m.to_csv(OUT / "record_manifest.csv.gz", index=False)
    summary = m.groupby(["source", "status"]).size().rename("records").reset_index()
    summary.to_csv(OUT / "record_manifest_summary.csv", index=False)
    log(f"rebuild check: {check}")
    log("record fates:\n" + summary.to_string(index=False))


def folds_and_indices(corpus: pd.DataFrame) -> None:
    train = corpus[corpus["role"] == "train_pool"].reset_index(drop=True)
    strata = train["source"] + "_" + train["label"].astype(str)
    fold = np.zeros(len(train), dtype=int)
    folds = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(train, strata))
    for k, (_, te) in enumerate(folds, 1):
        fold[te] = k
    train[["key", "source", "label"]].assign(random_fold=fold).to_csv(OUT / "fold_indices.csv.gz", index=False)

    def cap(frame):
        s = frame.sample(frac=1.0, random_state=SEED)
        return s[s.groupby(["source", "label"]).cumcount() < 3000]
    parts = [cap(train.iloc[tr])[["key"]].assign(fold=f"random_{k}") for k, (tr, _) in enumerate(folds, 1)]
    parts += [cap(train[train["source"] != s])[["key"]].assign(fold=f"lodo_{s}") for s in sorted(train["source"].unique())]
    parts.append(cap(train)[["key"]].assign(fold="full_pool_notinject"))
    pd.concat(parts).to_csv(OUT / "capped_training_keys.csv.gz", index=False)

    six = load_script("06_fewshot_adaptation.py")  # the exact split and draw functions of script 06
    rows = []
    for source in sorted(train["source"].unique()):
        pool, test = six.split_source(train[train["source"] == source], SEED)
        draws = {i: [] for i in pool.index}
        for k in (10, 50, 200, 1000):
            for r in range(3):
                added = six.draw(pool, k, SEED + r)
                if added is None:
                    break
                for i in added.index:
                    draws[i].append(f"k{k}r{r}")
        rows += [{"source": source, "key": train.at[i, "key"], "part": "pool", "draws": ";".join(draws[i])}
                 for i in pool.index]
        rows += [{"source": source, "key": train.at[i, "key"], "part": "test", "draws": ""} for i in test.index]
    f = pd.DataFrame(rows)
    f.to_csv(OUT / "fewshot_indices.csv.gz", index=False)
    log("fold, capped-training and few-shot indices written")


def per_record_scores(corpus: pd.DataFrame) -> None:
    train = corpus[corpus["role"] == "train_pool"].reset_index(drop=True)
    lodo_order = pd.concat([train[train["source"] == s] for s in sorted(train["source"].unique())])
    frames = []
    for name, rel in SCORES.items():
        path = RES / rel
        if not path.exists():
            continue
        d = pd.read_parquet(path)
        rnd, lodo = d[d["protocol"] == "random"], d[d["protocol"] == "lodo"]
        if not ((rnd["label"].to_numpy() == train["label"].to_numpy()).all()
                and (lodo["source"].to_numpy() == lodo_order["source"].to_numpy()).all()
                and (lodo["label"].to_numpy() == lodo_order["label"].to_numpy()).all()):
            raise ValueError(f"{name}: score rows do not align with the corpus")
        frames.append(pd.DataFrame({"key": train["key"], "source": train["source"], "label": train["label"],
                                    "protocol": "random", "detector": name, "score": rnd["score"].to_numpy()}))
        frames.append(pd.DataFrame({"key": lodo_order["key"].to_numpy(), "source": lodo_order["source"].to_numpy(),
                                    "label": lodo_order["label"].to_numpy(), "protocol": "lodo", "detector": name,
                                    "score": lodo["score"].to_numpy()}))
    pd.concat(frames, ignore_index=True).to_parquet(OUT / "per_record_scores.parquet", index=False)
    log("per-record scores written")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--skip-hub", action="store_true", help="do not query the Hugging Face hub")
    parser.add_argument("--skip-raw", action="store_true", help="skip parts that need data/raw")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    corpus = pd.read_parquet(DATA / "corpus.parquet")
    environment()
    (OUT / "settings.json").write_text(json.dumps(SETTINGS, indent=2), encoding="utf-8")
    if not args.skip_hub:
        hub_revisions()
    if not args.skip_raw:
        raw_hashes()
        record_manifest(corpus)
    folds_and_indices(corpus)
    per_record_scores(corpus)
    log(f"done: {OUT}")


if __name__ == "__main__":
    main()
