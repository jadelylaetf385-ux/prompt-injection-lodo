"""Random-split versus leave-one-dataset-out (LODO) evaluation of black-box detectors.

Detectors (all trained from scratch on the corpus, CPU only)
--------------------------------------------------------------
surface   logistic regression on six surface statistics of the text (length,
          word count, character-class ratios). It knows nothing about content
          and therefore measures how far format alone separates the labels.
word_lr   word 1-2 gram TF-IDF with logistic regression.
char_svm  character 3-5 gram TF-IDF with a linear SVM.
minilm_lr all-MiniLM-L6-v2 sentence embeddings with logistic regression.

Protocols
---------
random    stratified 5-fold cross-validation over the pooled training sources;
          every record receives an out-of-fold score.
lodo      each source is held out in turn; the detector is trained on all other
          sources and scored on the held-out one.
For both protocols the same per-source metrics are computed, so the gap between
them is measured source by source: ROC-AUC, average precision and F1 for
sources containing both labels, detection rate for attack-only sources and
false-positive rate for benign-only sources.

Additional outputs
------------------
notinject  false-positive rate on NotInject (benign texts containing trigger
           words) for each detector trained on all training sources.
source_id  how accurately the same features identify which dataset a text
           came from (a direct measure of dataset-specific signatures).

Usage
-----
    python scripts/03_lodo_experiments.py
    python scripts/03_lodo_experiments.py --detectors surface,word_lr,char_svm
    python scripts/03_lodo_experiments.py --exclude mosscap --run-name no_mosscap
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
SEED = 42
ALL_DETECTORS = ["surface", "word_lr", "char_svm", "minilm_lr"]
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


# --------------------------------------------------------------------------- #
# Features and detectors
# --------------------------------------------------------------------------- #

def surface_features(texts: pd.Series) -> np.ndarray:
    s = texts.astype(str)
    length = s.str.len().clip(lower=1)
    words = s.str.split().str.len().fillna(0)
    upper = s.str.count(r"[A-Z]") / length
    digit = s.str.count(r"[0-9]") / length
    punct = s.str.count(r"[^\w\s]") / length
    newline = s.str.count("\n")
    return np.column_stack([np.log1p(length), np.log1p(words), upper, digit, punct, np.log1p(newline)])


class Detector:
    """Common fit/score interface; score() returns a continuous attack score and
    threshold is the decision point on that score."""

    threshold = 0.5

    def fit(self, X, y):
        raise NotImplementedError

    def score(self, X):
        raise NotImplementedError


class SurfaceLR(Detector):
    def fit(self, texts, y):
        self.scaler = StandardScaler()
        Z = self.scaler.fit_transform(surface_features(texts))
        self.model = LogisticRegression(max_iter=2000, class_weight="balanced")
        self.model.fit(Z, y)
        return self

    def score(self, texts):
        return self.model.predict_proba(self.scaler.transform(surface_features(texts)))[:, 1]


class WordLR(Detector):
    def fit(self, texts, y):
        self.vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=200000,
                                   sublinear_tf=True, lowercase=True)
        X = self.vec.fit_transform(texts)
        self.model = LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear")
        self.model.fit(X, y)
        return self

    def score(self, texts):
        return self.model.predict_proba(self.vec.transform(texts))[:, 1]


class CharSVM(Detector):
    threshold = 0.0

    def fit(self, texts, y):
        self.vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3,
                                   max_features=300000, sublinear_tf=True, lowercase=True)
        X = self.vec.fit_transform(texts)
        self.model = LinearSVC(C=0.5, class_weight="balanced", max_iter=5000)
        self.model.fit(X, y)
        return self

    def score(self, texts):
        return self.model.decision_function(self.vec.transform(texts))


class EmbeddingLR(Detector):
    """Logistic regression on pre-computed sentence embeddings (rows of E)."""

    def __init__(self, embeddings: np.ndarray):
        self.E = embeddings

    def fit(self, index, y):
        self.model = LogisticRegression(max_iter=3000, class_weight="balanced", C=1.0)
        self.model.fit(self.E[index], y)
        return self

    def score(self, index):
        return self.model.predict_proba(self.E[index])[:, 1]


def make_detector(name: str, embeddings: np.ndarray | None) -> Detector:
    if name == "surface":
        return SurfaceLR()
    if name == "word_lr":
        return WordLR()
    if name == "char_svm":
        return CharSVM()
    if name == "minilm_lr":
        return EmbeddingLR(embeddings)
    raise ValueError(name)


def inputs_for(name: str, frame: pd.DataFrame):
    """Text detectors take the texts; the embedding detector takes row indices."""
    return frame["row"].to_numpy() if name == "minilm_lr" else frame["text"]


# --------------------------------------------------------------------------- #
# Embeddings (computed once, cached on disk)
# --------------------------------------------------------------------------- #

def load_or_compute_embeddings(corpus: pd.DataFrame) -> np.ndarray:
    cache = DATA / "embeddings_minilm.npy"
    keys_file = DATA / "embeddings_minilm_keys.json"
    keys = corpus["key"].tolist()
    if cache.exists() and keys_file.exists() and json.loads(keys_file.read_text()) == keys:
        log("embeddings: loaded from cache")
        return np.load(cache)
    from sentence_transformers import SentenceTransformer

    log(f"embeddings: encoding {len(corpus):,} texts with {EMBEDDING_MODEL} on CPU (cached afterwards)")
    model = SentenceTransformer(EMBEDDING_MODEL, device="cpu")
    model.max_seq_length = 256
    E = model.encode(corpus["text"].tolist(), batch_size=64, show_progress_bar=True,
                     normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
    np.save(cache, E)
    keys_file.write_text(json.dumps(keys))
    return E


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #

def source_kind(labels: pd.Series) -> str:
    values = set(labels.unique())
    if values == {0, 1}:
        return "mixed"
    return "attack_only" if values == {1} else "benign_only"


def per_source_metrics(y: np.ndarray, s: np.ndarray, threshold: float, kind: str) -> dict:
    pred = (s >= threshold).astype(int) if threshold == 0.5 else (s > threshold).astype(int)
    row = {"n": int(len(y)), "kind": kind}
    if kind == "mixed":
        row.update({
            "roc_auc": float(roc_auc_score(y, s)),
            "average_precision": float(average_precision_score(y, s)),
            "f1": float(f1_score(y, pred, zero_division=0)),
            "accuracy": float(accuracy_score(y, pred)),
            "detection_rate": float(pred[y == 1].mean()),
            "false_positive_rate": float(pred[y == 0].mean()),
        })
    elif kind == "attack_only":
        row["detection_rate"] = float(pred.mean())
    else:
        row["false_positive_rate"] = float(pred.mean())
    return row


def summarise(frame: pd.DataFrame, detector: str, protocol: str, threshold: float, kinds: dict) -> list[dict]:
    rows = []
    for source, group in frame.groupby("source"):
        metrics = per_source_metrics(group["label"].to_numpy(), group["score"].to_numpy(), threshold, kinds[source])
        rows.append({"detector": detector, "protocol": protocol, "source": source, **metrics})
    return rows


def pooled(frame: pd.DataFrame, detector: str, protocol: str, threshold: float) -> dict:
    y, s = frame["label"].to_numpy(), frame["score"].to_numpy()
    pred = (s >= threshold).astype(int) if threshold == 0.5 else (s > threshold).astype(int)
    return {
        "detector": detector, "protocol": protocol,
        "pooled_roc_auc": float(roc_auc_score(y, s)),
        "pooled_average_precision": float(average_precision_score(y, s)),
        "pooled_f1": float(f1_score(y, pred)),
        "pooled_accuracy": float(accuracy_score(y, pred)),
        "pooled_detection_rate": float(pred[y == 1].mean()),
        "pooled_false_positive_rate": float(pred[y == 0].mean()),
    }


# --------------------------------------------------------------------------- #
# Protocols
# --------------------------------------------------------------------------- #

def run_random(name, train, embeddings, n_splits=5):
    strata = train["source"] + "_" + train["label"].astype(str)
    folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    scores = np.zeros(len(train))
    for k, (tr, te) in enumerate(folds.split(train, strata), start=1):
        det = make_detector(name, embeddings)
        det.fit(inputs_for(name, train.iloc[tr]), train["label"].iloc[tr].to_numpy())
        scores[te] = det.score(inputs_for(name, train.iloc[te]))
        log(f"  {name} random fold {k}/{n_splits}")
    out = train[["source", "label"]].copy()
    out["score"] = scores
    return out, det.threshold


def run_lodo(name, train, embeddings):
    parts = []
    threshold = None
    for source in sorted(train["source"].unique()):
        held = train["source"] == source
        det = make_detector(name, embeddings)
        det.fit(inputs_for(name, train[~held]), train.loc[~held, "label"].to_numpy())
        part = train.loc[held, ["source", "label"]].copy()
        part["score"] = det.score(inputs_for(name, train[held]))
        parts.append(part)
        threshold = det.threshold
        log(f"  {name} LODO held out {source}")
    return pd.concat(parts), threshold


def run_source_id(name, train, embeddings, sample=60000):
    """Predict the source dataset from the detector's own features (3-fold CV)."""
    sub = _balanced_sample(train, sample)
    y = sub["source"].to_numpy()
    folds = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    preds = np.empty(len(sub), dtype=object)
    for tr, te in folds.split(sub, y):
        if name == "surface":
            scaler = StandardScaler()
            Xtr = scaler.fit_transform(surface_features(sub["text"].iloc[tr]))
            Xte = scaler.transform(surface_features(sub["text"].iloc[te]))
        elif name in ("word_lr", "char_svm"):
            vec = (TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=200000, sublinear_tf=True)
                   if name == "word_lr" else
                   TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, max_features=300000, sublinear_tf=True))
            Xtr = vec.fit_transform(sub["text"].iloc[tr])
            Xte = vec.transform(sub["text"].iloc[te])
        else:
            Xtr = embeddings[sub["row"].to_numpy()[tr]]
            Xte = embeddings[sub["row"].to_numpy()[te]]
        clf = (LinearSVC(C=0.5, max_iter=5000) if sparse.issparse(Xtr)
               else LogisticRegression(max_iter=3000))
        clf.fit(Xtr, y[tr])
        preds[te] = clf.predict(Xte)
    n_sources = len(np.unique(y))
    return {
        "detector": name,
        "n_records": int(len(sub)),
        "n_sources": int(n_sources),
        "chance_accuracy": round(1.0 / n_sources, 4),
        "accuracy": float(accuracy_score(y, preds)),
        "macro_f1": float(f1_score(y, preds, average="macro")),
    }


def _balanced_sample(train: pd.DataFrame, total: int) -> pd.DataFrame:
    per = max(1, total // train["source"].nunique())
    shuffled = train.sample(frac=1.0, random_state=SEED)
    return shuffled[shuffled.groupby("source").cumcount() < per]


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--detectors", default=",".join(ALL_DETECTORS))
    parser.add_argument("--exclude", default="", help="comma-separated training sources to drop entirely")
    parser.add_argument("--run-name", default="main")
    parser.add_argument("--skip-source-id", action="store_true")
    args = parser.parse_args()

    detectors = [d.strip() for d in args.detectors.split(",") if d.strip()]
    unknown = set(detectors) - set(ALL_DETECTORS)
    if unknown:
        raise ValueError(f"unknown detectors: {sorted(unknown)}")
    out_dir = ROOT / "results" / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    corpus = pd.read_parquet(DATA / "corpus.parquet").reset_index(drop=True)
    corpus["row"] = np.arange(len(corpus))
    excluded = [s.strip() for s in args.exclude.split(",") if s.strip()]
    train = corpus[(corpus["role"] == "train_pool") & (~corpus["source"].isin(excluded))].reset_index(drop=True)
    notinject = corpus[corpus["source"] == "notinject"].reset_index(drop=True)
    kinds = {s: source_kind(g["label"]) for s, g in train.groupby("source")}
    log(f"training pool: {len(train):,} texts from {train['source'].nunique()} sources; excluded: {excluded or 'none'}")
    log("source kinds: " + ", ".join(f"{k}={v}" for k, v in sorted(kinds.items())))

    embeddings = load_or_compute_embeddings(corpus) if "minilm_lr" in detectors else None

    per_source, pooled_rows, ni_rows, sid_rows, timings = [], [], [], [], {}
    for name in detectors:
        started = time.perf_counter()
        log(f"=== {name}: random split ===")
        rnd, thr = run_random(name, train, embeddings)
        per_source += summarise(rnd, name, "random", thr, kinds)
        pooled_rows.append(pooled(rnd, name, "random", thr))

        log(f"=== {name}: leave-one-dataset-out ===")
        lodo, thr = run_lodo(name, train, embeddings)
        per_source += summarise(lodo, name, "lodo", thr, kinds)
        pooled_rows.append(pooled(lodo, name, "lodo", thr))

        log(f"=== {name}: NotInject ===")
        det = make_detector(name, embeddings)
        det.fit(inputs_for(name, train), train["label"].to_numpy())
        s = det.score(inputs_for(name, notinject))
        pred = (s >= det.threshold) if det.threshold == 0.5 else (s > det.threshold)
        ni_rows.append({"detector": name, "n": int(len(notinject)), "false_positive_rate": float(np.mean(pred))})

        if not args.skip_source_id:
            log(f"=== {name}: source identification ===")
            sid_rows.append(run_source_id(name, train, embeddings))

        both = pd.concat([rnd.assign(protocol="random"), lodo.assign(protocol="lodo")])
        both.assign(detector=name).to_parquet(out_dir / f"scores_{name}.parquet", index=False)
        timings[name] = round(time.perf_counter() - started, 1)
        log(f"{name} done in {timings[name]:.0f}s")

        # Write after every detector so an interruption keeps finished results.
        pd.DataFrame(per_source).to_csv(out_dir / "per_source_metrics.csv", index=False)
        pd.DataFrame(pooled_rows).to_csv(out_dir / "pooled_metrics.csv", index=False)
        pd.DataFrame(ni_rows).to_csv(out_dir / "notinject.csv", index=False)
        if sid_rows:
            pd.DataFrame(sid_rows).to_csv(out_dir / "source_identification.csv", index=False)

    (out_dir / "run_config.json").write_text(json.dumps({
        "detectors": detectors, "excluded_sources": excluded, "seed": SEED,
        "training_records": int(len(train)), "source_kinds": kinds,
        "embedding_model": EMBEDDING_MODEL if "minilm_lr" in detectors else None,
        "timings_seconds": timings,
    }, indent=2), encoding="utf-8")

    pool = pd.DataFrame(pooled_rows)
    log("pooled results:\n" + pool[["detector", "protocol", "pooled_roc_auc", "pooled_f1",
                                    "pooled_detection_rate", "pooled_false_positive_rate"]].round(4).to_string(index=False))
    log(f"all results in results/{args.run_name}/")


if __name__ == "__main__":
    main()
