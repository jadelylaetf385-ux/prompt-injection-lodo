"""Low-cost mitigations of dataset shortcuts, evaluated under leave-one-dataset-out.

All variants use the word 1-2 gram TF-IDF + logistic-regression detector of
script 03 and change only how it is trained or how its decision threshold is
chosen. Each variant is evaluated with exactly the same LODO folds.

Variants
--------
baseline     the detector of script 03.
balanced     sample weights give every training source the same total weight,
             split equally between its labels, so large sources and
             single-label sources stop dominating the decision boundary.
normalised   texts are canonicalised before vectorisation: lower case, URLs,
             e-mail addresses and numbers masked, whitespace collapsed, and
             truncated to the first 1,000 characters, removing format and
             length cues that differ by source.
pruned       inside each fold, n-gram features that identify the source far
             more strongly than they identify the label are removed
             (source-adversarial feature pruning).
domain_thr   the decision threshold is chosen without any label from the
             held-out source: an inner leave-one-source-out loop over the
             training sources selects the threshold that maximises the mean
             per-source utility on unseen sources.
combined     balanced + normalised + pruned + domain_thr.

Usage
-----
    python scripts/05_mitigations.py
    python scripts/05_mitigations.py --variants baseline,balanced,normalised
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "results" / "mitigations"
SEED = 42
ALL_VARIANTS = ["baseline", "balanced", "normalised", "pruned", "domain_thr", "combined"]
VARIANT_SPEC = {
    "baseline": dict(balanced=False, normalised=False, pruned=False, domain_thr=False),
    "balanced": dict(balanced=True, normalised=False, pruned=False, domain_thr=False),
    "normalised": dict(balanced=False, normalised=True, pruned=False, domain_thr=False),
    "pruned": dict(balanced=False, normalised=False, pruned=True, domain_thr=False),
    "domain_thr": dict(balanced=False, normalised=False, pruned=False, domain_thr=True),
    "combined": dict(balanced=True, normalised=True, pruned=True, domain_thr=True),
}
THRESHOLD_GRID = np.round(np.arange(0.05, 0.96, 0.05), 2)


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


# --------------------------------------------------------------------------- #
# Building blocks
# --------------------------------------------------------------------------- #

_URL = re.compile(r"https?://\S+|www\.\S+")
_MAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_NUM = re.compile(r"\d+")
_WS = re.compile(r"\s+")


def canonicalise(text: str, max_chars: int = 1000) -> str:
    t = str(text).lower()
    t = _URL.sub(" urltoken ", t)
    t = _MAIL.sub(" emailtoken ", t)
    t = _NUM.sub(" numtoken ", t)
    t = _WS.sub(" ", t).strip()
    return t[:max_chars]


def source_balanced_weights(sources: np.ndarray, labels: np.ndarray) -> np.ndarray:
    frame = pd.DataFrame({"s": sources, "y": labels})
    n_sources = frame["s"].nunique()
    cells = frame.groupby(["s", "y"]).size()
    labels_per_source = frame.groupby("s")["y"].nunique()
    weights = np.empty(len(frame))
    for i, (s, y) in enumerate(zip(frame["s"], frame["y"])):
        weights[i] = 1.0 / (n_sources * labels_per_source[s] * cells[(s, y)])
    return weights * len(frame)


def utility(y: np.ndarray, pred: np.ndarray) -> float:
    """Per-source utility used for label-free threshold selection: balanced
    accuracy for mixed sources, detection rate for attack-only sources and
    specificity for benign-only sources."""
    labels = set(np.unique(y))
    if labels == {0, 1}:
        return 0.5 * (pred[y == 1].mean() + (1 - pred[y == 0].mean()))
    return pred.mean() if labels == {1} else 1 - pred.mean()


class Detector:
    def __init__(self, balanced: bool, normalised: bool, pruned: bool, prune_fraction: float):
        self.balanced, self.normalised, self.pruned = balanced, normalised, pruned
        self.prune_fraction = prune_fraction
        self.keep = None

    def _texts(self, texts):
        return [canonicalise(t) for t in texts] if self.normalised else list(texts)

    def fit(self, texts, labels, sources):
        docs = self._texts(texts)
        self.vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=200000, sublinear_tf=True)
        X = self.vec.fit_transform(docs)
        y = np.asarray(labels)
        w = source_balanced_weights(np.asarray(sources), y) if self.balanced else None
        if self.pruned:
            X = self._prune(X, y, np.asarray(sources), w)
        self.model = LogisticRegression(max_iter=2000, solver="liblinear",
                                        class_weight=None if self.balanced else "balanced")
        self.model.fit(X, y, sample_weight=w)
        return self

    def _prune(self, X, y, sources, w):
        rng = np.random.default_rng(SEED)
        idx = rng.choice(X.shape[0], size=min(X.shape[0], 60000), replace=False)
        src_clf = LinearSVC(C=0.5, max_iter=5000).fit(X[idx], sources[idx])
        source_strength = np.abs(src_clf.coef_).max(axis=0)
        lab_clf = LogisticRegression(max_iter=2000, solver="liblinear", class_weight="balanced").fit(X[idx], y[idx])
        label_strength = np.abs(lab_clf.coef_[0])
        ratio = source_strength / (label_strength + 1e-6)
        n_drop = int(self.prune_fraction * X.shape[1])
        drop = np.argsort(ratio)[::-1][:n_drop]
        self.keep = np.setdiff1d(np.arange(X.shape[1]), drop)
        self.n_pruned = n_drop
        return X[:, self.keep]

    def score(self, texts):
        X = self.vec.transform(self._texts(texts))
        if self.keep is not None:
            X = X[:, self.keep]
        return self.model.predict_proba(X)[:, 1]


def choose_domain_threshold(train: pd.DataFrame, spec: dict, prune_fraction: float) -> float:
    """Inner leave-one-source-out over the training sources only."""
    collected = []
    for inner in sorted(train["source"].unique()):
        held = train["source"] == inner
        det = Detector(spec["balanced"], spec["normalised"], spec["pruned"], prune_fraction)
        det.fit(train.loc[~held, "text"], train.loc[~held, "label"], train.loc[~held, "source"])
        collected.append((train.loc[held, "label"].to_numpy(), det.score(train.loc[held, "text"])))
    best, best_u = 0.5, -1.0
    for t in THRESHOLD_GRID:
        u = np.mean([utility(y, (s >= t).astype(int)) for y, s in collected])
        if u > best_u:
            best, best_u = float(t), u
    return best


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #

def source_metrics(y, s, t):
    pred = (s >= t).astype(int)
    labels = set(np.unique(y))
    row = {"n": int(len(y)), "threshold": t, "utility": float(utility(y, pred))}
    if labels == {0, 1}:
        row.update(kind="mixed", roc_auc=float(roc_auc_score(y, s)),
                   average_precision=float(average_precision_score(y, s)),
                   f1=float(f1_score(y, pred, zero_division=0)),
                   detection_rate=float(pred[y == 1].mean()), false_positive_rate=float(pred[y == 0].mean()))
    elif labels == {1}:
        row.update(kind="attack_only", detection_rate=float(pred.mean()))
    else:
        row.update(kind="benign_only", false_positive_rate=float(pred.mean()))
    return row


def run_variant(name, train, notinject, prune_fraction):
    spec = VARIANT_SPEC[name]
    rows, scores = [], []
    for source in sorted(train["source"].unique()):
        held = train["source"] == source
        tr = train[~held]
        threshold = choose_domain_threshold(tr, spec, prune_fraction) if spec["domain_thr"] else 0.5
        det = Detector(spec["balanced"], spec["normalised"], spec["pruned"], prune_fraction)
        det.fit(tr["text"], tr["label"], tr["source"])
        s = det.score(train.loc[held, "text"])
        y = train.loc[held, "label"].to_numpy()
        rows.append({"variant": name, "source": source, **source_metrics(y, s, threshold)})
        scores.append(pd.DataFrame({"source": source, "label": y, "score": s, "threshold": threshold}))
        log(f"  {name}: held out {source} (threshold {threshold:.2f})")

    pooled = pd.concat(scores)
    y, s = pooled["label"].to_numpy(), pooled["score"].to_numpy()
    pred = (s >= pooled["threshold"].to_numpy()).astype(int)
    per = pd.DataFrame(rows)

    # NotInject: detector trained on all sources, threshold chosen the same way.
    threshold = choose_domain_threshold(train, spec, prune_fraction) if spec["domain_thr"] else 0.5
    det = Detector(spec["balanced"], spec["normalised"], spec["pruned"], prune_fraction)
    det.fit(train["text"], train["label"], train["source"])
    ni_fpr = float((det.score(notinject["text"]) >= threshold).mean())

    summary = {
        "variant": name,
        "pooled_roc_auc": float(roc_auc_score(y, s)),
        "pooled_detection_rate": float(pred[y == 1].mean()),
        "pooled_false_positive_rate": float(pred[y == 0].mean()),
        "mean_source_utility": float(per["utility"].mean()),
        "mean_mixed_auc": float(per.loc[per["kind"] == "mixed", "roc_auc"].mean()),
        "llmail_detection_rate": float(per.loc[per["source"] == "llmail", "detection_rate"].mean())
        if "llmail" in set(per["source"]) else np.nan,
        "notinject_false_positive_rate": ni_fpr,
    }
    return per, summary, pooled.assign(variant=name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variants", default=",".join(ALL_VARIANTS))
    parser.add_argument("--prune-fraction", type=float, default=0.05)
    args = parser.parse_args()
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    unknown = set(variants) - set(ALL_VARIANTS)
    if unknown:
        raise ValueError(f"unknown variants: {sorted(unknown)}")
    OUT.mkdir(parents=True, exist_ok=True)

    corpus = pd.read_parquet(DATA / "corpus.parquet")
    train = corpus[corpus["role"] == "train_pool"].reset_index(drop=True)
    notinject = corpus[corpus["source"] == "notinject"].reset_index(drop=True)
    log(f"training pool {len(train):,}; variants: {variants}")

    per_all, summaries, timings = [], [], {}
    for name in variants:
        started = time.perf_counter()
        log(f"=== {name} ===")
        per, summary, pooled = run_variant(name, train, notinject, args.prune_fraction)
        per_all.append(per)
        summaries.append(summary)
        pooled.to_parquet(OUT / f"scores_{name}.parquet", index=False)
        timings[name] = round(time.perf_counter() - started, 1)
        pd.concat(per_all).to_csv(OUT / "per_source_metrics.csv", index=False)
        pd.DataFrame(summaries).to_csv(OUT / "summary.csv", index=False)
        log(f"{name} done in {timings[name]:.0f}s")

    (OUT / "run_config.json").write_text(json.dumps({
        "variants": variants, "prune_fraction": args.prune_fraction, "seed": SEED,
        "threshold_grid": THRESHOLD_GRID.tolist(), "timings_seconds": timings}, indent=2), encoding="utf-8")
    log("summary:\n" + pd.DataFrame(summaries).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
