"""Equal-data control for the fine-tuned encoder, and CPU inference latency.

equal_data  The fine-tuned encoder of script 08 was trained on at most 3,000
            texts per source and label. To check that its larger LODO drop is
            not caused by the smaller training set, the word TF-IDF detector is
            retrained under exactly the same cap and the same random and LODO
            folds, and its pooled ROC-AUC, LLMail and Mosscap detection rates
            are reported.
latency     Single-text CPU inference time of the lightweight detectors
            (surface, word TF-IDF, character TF-IDF and, if
            sentence-transformers is installed, MiniLM), measured on 1,000
            texts drawn from the corpus after a warm-up. Reported as median
            and 95th percentile in milliseconds per text.

Usage
-----
    python scripts/10_equal_data_and_latency.py
    python scripts/10_equal_data_and_latency.py --skip-equal-data
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "results" / "equal_data_latency"
SEED = 42
CAP = 3000


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def cap(frame: pd.DataFrame, per_cell: int = CAP) -> pd.DataFrame:
    shuffled = frame.sample(frac=1.0, random_state=SEED)
    return shuffled[shuffled.groupby(["source", "label"]).cumcount() < per_cell]


def word_lr(train_texts, train_labels):
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=200000, sublinear_tf=True)
    model = LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear")
    model.fit(vec.fit_transform(train_texts), train_labels)
    return vec, model


def equal_data(train: pd.DataFrame) -> dict:
    strata = train["source"] + "_" + train["label"].astype(str)
    rnd = np.zeros(len(train))
    for k, (tr, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=SEED).split(train, strata), 1):
        sub = cap(train.iloc[tr])
        vec, model = word_lr(sub["text"], sub["label"])
        rnd[te] = model.predict_proba(vec.transform(train["text"].iloc[te]))[:, 1]
        log(f"  equal-data random fold {k}/5")
    parts = []
    for source in sorted(train["source"].unique()):
        held = train["source"] == source
        sub = cap(train[~held])
        vec, model = word_lr(sub["text"], sub["label"])
        parts.append(pd.DataFrame({"source": source, "label": train.loc[held, "label"].to_numpy(),
                                   "score": model.predict_proba(vec.transform(train.loc[held, "text"]))[:, 1]}))
        log(f"  equal-data LODO held out {source}")
    lodo = pd.concat(parts)
    auc_r = roc_auc_score(train["label"], rnd)
    auc_l = roc_auc_score(lodo["label"], lodo["score"])
    det = lambda s: float((lodo.loc[lodo["source"] == s, "score"] >= 0.5).mean())
    return {"detector": "word_lr_capped", "cap_per_source_and_label": CAP, "auc_random": auc_r, "auc_lodo": auc_l,
            "gap_points": 100 * (auc_l - auc_r), "llmail_detection_rate": det("llmail"),
            "mosscap_detection_rate": det("mosscap")}


def surface_features(texts: pd.Series) -> np.ndarray:  # identical to script 03
    s = texts.astype(str)
    length = s.str.len().clip(lower=1)
    words = s.str.split().str.len().fillna(0)
    upper = s.str.count(r"[A-Z]") / length
    digit = s.str.count(r"[0-9]") / length
    punct = s.str.count(r"[^\w\s]") / length
    newline = s.str.count("\n")
    return np.column_stack([np.log1p(length), np.log1p(words), upper, digit, punct, np.log1p(newline)])


def time_per_text(predict, texts: list[str], warmup: int = 50) -> tuple[float, float]:
    for t in texts[:warmup]:
        predict([t])
    times = []
    for t in texts:
        start = time.perf_counter()
        predict([t])
        times.append(1000 * (time.perf_counter() - start))
    return float(np.median(times)), float(np.percentile(times, 95))


def latency(train: pd.DataFrame) -> list[dict]:
    fit = cap(train)
    texts = train.sample(n=1000, random_state=SEED)["text"].tolist()
    y = fit["label"].to_numpy()
    rows = []

    scaler = StandardScaler().fit(surface_features(fit["text"]))
    lr = LogisticRegression(max_iter=2000, class_weight="balanced").fit(scaler.transform(surface_features(fit["text"])), y)
    rows.append(("surface", lambda b: lr.predict_proba(scaler.transform(surface_features(pd.Series(b))))[:, 1]))

    vec, model = word_lr(fit["text"], y)
    rows.append(("word_lr", lambda b: model.predict_proba(vec.transform(b))[:, 1]))

    cvec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, max_features=300000,
                           sublinear_tf=True, lowercase=True)
    svm = LinearSVC(C=0.5, class_weight="balanced", max_iter=5000).fit(cvec.fit_transform(fit["text"]), y)
    rows.append(("char_svm", lambda b: svm.decision_function(cvec.transform(b))))

    try:
        from sentence_transformers import SentenceTransformer
        enc = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
        enc.max_seq_length = 256
        emb = enc.encode(fit["text"].tolist()[:20000], batch_size=64, normalize_embeddings=True)
        mlr = LogisticRegression(max_iter=3000, class_weight="balanced").fit(emb, y[:20000])
        rows.append(("minilm_lr", lambda b: mlr.predict_proba(enc.encode(b, normalize_embeddings=True))[:, 1]))
    except Exception as error:  # optional dependency or no network
        log(f"  MiniLM latency skipped ({type(error).__name__})")

    out = []
    for name, predict in rows:
        median, p95 = time_per_text(predict, texts)
        out.append({"detector": name, "median_ms_per_text": median, "p95_ms_per_text": p95})
        log(f"  {name}: median {median:.2f} ms, p95 {p95:.2f} ms")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--skip-equal-data", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    corpus = pd.read_parquet(DATA / "corpus.parquet")
    train = corpus[corpus["role"] == "train_pool"].reset_index(drop=True)

    if not args.skip_equal_data:
        log("equal-data control (word TF-IDF, cap 3,000 per source and label)")
        result = equal_data(train)
        pd.DataFrame([result]).to_csv(OUT / "equal_data_word_lr.csv", index=False)
        log(f"  random {result['auc_random']:.4f}, LODO {result['auc_lodo']:.4f}, gap {result['gap_points']:.1f} "
            f"points, LLMail DR {result['llmail_detection_rate']:.3f}, Mosscap DR {result['mosscap_detection_rate']:.3f}")

    log("single-text CPU latency")
    lat = latency(train)
    pd.DataFrame(lat).to_csv(OUT / "latency_cpu.csv", index=False)
    (OUT / "run_config.json").write_text(json.dumps({"seed": SEED, "cap": CAP, "processor": platform.processor(),
                                                     "platform": platform.platform()}, indent=2), encoding="utf-8")
    log("done")


if __name__ == "__main__":
    main()
