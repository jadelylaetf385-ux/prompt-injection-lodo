"""Shortcut features, error examples and pairwise detector comparison.

Three analyses that use the corpus and the saved out-of-fold scores only (no
new training except one word TF-IDF model on the full pool):

features   the word n-grams with the largest attack and benign weights in a
           word TF-IDF + logistic-regression detector trained on the whole
           pool, with the share of their occurrences that comes from a single
           source. A strongly weighted feature that occurs almost only in one
           source is a dataset signature rather than an attack cue.
errors     held-out attacks that every trained content detector missed under
           LODO (per source), and NotInject sentences flagged by the word
           detector trained on the whole pool. Short examples are written to a
           CSV for manual inspection.
pairwise   paired, stratified record-level bootstrap of the difference in
           pooled LODO ROC-AUC between pairs of detectors.

Alignment: script 03/08 store LODO scores grouped by sorted source, keeping
corpus order within each source; the corpus is reordered the same way and the
alignment is checked on source and label.

Usage
-----
    python scripts/09_shortcut_and_error_analysis.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
OUT = RESULTS / "analysis"
SEED = 42
N_BOOT = 1000

DETECTORS = {  # name: (score file, threshold, operator)
    "word_lr": ("main/scores_word_lr.parquet", 0.5, ">="),
    "char_svm": ("main/scores_char_svm.parquet", 0.0, ">"),
    "minilm_lr": ("minilm/scores_minilm_lr.parquet", 0.5, ">="),
    "finetuned": ("finetune/scores_finetuned.parquet", 0.5, ">="),
}
PAIRS = [("char_svm", "word_lr"), ("word_lr", "minilm_lr"), ("char_svm", "finetuned"), ("word_lr", "finetuned")]


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def lodo_frame(train: pd.DataFrame) -> pd.DataFrame:
    """Corpus rows in the order used for LODO scores, with each detector's score and prediction."""
    frame = pd.concat([train[train["source"] == s] for s in sorted(train["source"].unique())]).reset_index(drop=True)
    for name, (rel, threshold, op) in DETECTORS.items():
        path = RESULTS / rel
        if not path.exists():
            log(f"{name}: {rel} not found, skipped")
            continue
        scores = pd.read_parquet(path)
        lodo = scores[scores["protocol"] == "lodo"].reset_index(drop=True)
        if not (lodo["source"].equals(frame["source"]) and lodo["label"].astype(int).equals(frame["label"].astype(int))):
            raise ValueError(f"{name}: LODO scores do not align with the corpus")
        frame[f"s_{name}"] = lodo["score"].to_numpy()
        s = frame[f"s_{name}"].to_numpy()
        frame[f"p_{name}"] = (s > threshold) if op == ">" else (s >= threshold)
    return frame


def shortcut_features(train: pd.DataFrame, top: int = 20) -> pd.DataFrame:
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=200000, sublinear_tf=True)
    X = vec.fit_transform(train["text"])
    y = train["label"].to_numpy()
    model = LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear").fit(X, y)
    coef = model.coef_[0]
    names = np.array(vec.get_feature_names_out())
    present = (X > 0).tocsc()
    sources = train["source"].to_numpy()
    rows = []
    for direction, idx in (("attack", np.argsort(coef)[::-1][:top]), ("benign", np.argsort(coef)[:top])):
        for j in idx:
            docs = present[:, j].nonzero()[0]
            counts = pd.Series(sources[docs]).value_counts()
            rows.append({"direction": direction, "feature": names[j], "weight": float(coef[j]),
                         "documents": int(len(docs)), "dominant_source": counts.index[0],
                         "dominant_share": float(counts.iloc[0] / len(docs)),
                         "attack_share": float(y[docs].mean())})
    return pd.DataFrame(rows), vec, model


def error_examples(frame: pd.DataFrame, notinject: pd.DataFrame, vec, model) -> tuple[pd.DataFrame, pd.DataFrame]:
    preds = [c for c in frame.columns if c.startswith("p_")]
    frame["missed_by_all"] = (frame["label"] == 1) & ~frame[preds].any(axis=1)
    summary = (frame[frame["label"] == 1].groupby("source")["missed_by_all"].agg(["sum", "mean"])
               .rename(columns={"sum": "missed_by_all", "mean": "share_missed_by_all"}).reset_index())
    examples = []
    rng = np.random.default_rng(SEED)
    for source in ("llmail", "mosscap", "hackaprompt"):
        missed = frame[(frame["source"] == source) & frame["missed_by_all"]]
        if len(missed):
            pick = missed.iloc[rng.choice(len(missed), size=min(15, len(missed)), replace=False)]
            examples += [{"kind": f"{source} attack missed by all", "text": t} for t in pick["text"]]
    ni_scores = model.predict_proba(vec.transform(notinject["text"]))[:, 1]
    flagged = notinject.assign(score=ni_scores)[ni_scores >= 0.5].sort_values("score", ascending=False)
    examples += [{"kind": "NotInject flagged by word detector", "text": t, "score": s}
                 for t, s in zip(flagged["text"].head(15), flagged["score"].head(15))]
    return summary, pd.DataFrame(examples)


def pairwise(frame: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    y = frame["label"].to_numpy()
    cells = [g.index.to_numpy() for _, g in frame.groupby(["source", "label"])]
    rows = []
    for a, b in PAIRS:
        if f"s_{a}" not in frame or f"s_{b}" not in frame:
            continue
        sa, sb = frame[f"s_{a}"].to_numpy(), frame[f"s_{b}"].to_numpy()
        diffs = np.empty(N_BOOT)
        for k in range(N_BOOT):
            idx = np.concatenate([rng.choice(c, size=len(c), replace=True) for c in cells])
            diffs[k] = roc_auc_score(y[idx], sa[idx]) - roc_auc_score(y[idx], sb[idx])
        point = roc_auc_score(y, sa) - roc_auc_score(y, sb)
        rows.append({"detector_a": a, "detector_b": b, "auc_difference": float(point),
                     "ci_low": float(np.percentile(diffs, 2.5)), "ci_high": float(np.percentile(diffs, 97.5)),
                     "share_boot_a_better": float((diffs > 0).mean())})
        log(f"  {a} - {b}: {point:+.4f} [{rows[-1]['ci_low']:+.4f}, {rows[-1]['ci_high']:+.4f}]")
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    corpus = pd.read_parquet(DATA / "corpus.parquet")
    train = corpus[corpus["role"] == "train_pool"].reset_index(drop=True)
    notinject = corpus[corpus["source"] == "notinject"].reset_index(drop=True)

    log("shortcut features (word TF-IDF on the full pool)")
    features, vec, model = shortcut_features(train)
    features.to_csv(OUT / "shortcut_features.csv", index=False)
    log("\n" + features.round(3).to_string(index=False))

    log("error analysis")
    frame = lodo_frame(train)
    summary, examples = error_examples(frame, notinject, vec, model)
    summary.to_csv(OUT / "missed_by_all_detectors.csv", index=False)
    examples.to_csv(OUT / "error_examples.csv", index=False)
    log("\n" + summary.round(3).to_string(index=False))

    log("pairwise LODO ROC-AUC differences")
    pairs = pairwise(frame)
    pairs.to_csv(OUT / "pairwise_lodo_auc.csv", index=False)
    (OUT / "run_config.json").write_text(json.dumps({"seed": SEED, "n_boot": N_BOOT,
                                                     "detectors": list(DETECTORS)}, indent=2), encoding="utf-8")
    log("done")


if __name__ == "__main__":
    main()
