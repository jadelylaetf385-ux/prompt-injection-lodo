"""How many labelled examples of an unseen source restore detection?

For every source, the word TF-IDF + logistic-regression detector is trained on
all other sources plus k examples drawn from the held-out source, and evaluated
on a fixed test portion of that source that is never used for adaptation.
k = 0 reproduces the leave-one-dataset-out setting. Each k > 0 is repeated with
several random draws.

The held-out source is split once, per label, into an adaptation pool (at most
1,000 records and at most half of the source) and a test part; draws of k are
taken from the pool only, so every k is evaluated on the same test records.

Usage
-----
    python scripts/06_fewshot_adaptation.py
    python scripts/06_fewshot_adaptation.py --k 0,10,50,200,1000 --repeats 3
"""

from __future__ import annotations

import argparse
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
OUT = ROOT / "results" / "fewshot"
SEED = 42
POOL_MAX = 1000


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def fit_score(train_texts, train_labels, test_texts) -> np.ndarray:
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=200000, sublinear_tf=True)
    X = vec.fit_transform(train_texts)
    model = LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear")
    model.fit(X, train_labels)
    return model.predict_proba(vec.transform(test_texts))[:, 1]


def evaluate(y: np.ndarray, s: np.ndarray, threshold: float = 0.5) -> dict:
    pred = (s >= threshold).astype(int)
    labels = set(np.unique(y))
    if labels == {0, 1}:
        return {"kind": "mixed", "roc_auc": float(roc_auc_score(y, s)),
                "detection_rate": float(pred[y == 1].mean()), "false_positive_rate": float(pred[y == 0].mean()),
                "utility": float(0.5 * (pred[y == 1].mean() + 1 - pred[y == 0].mean()))}
    if labels == {1}:
        return {"kind": "attack_only", "detection_rate": float(pred.mean()), "utility": float(pred.mean())}
    return {"kind": "benign_only", "false_positive_rate": float(pred.mean()), "utility": float(1 - pred.mean())}


def split_source(frame: pd.DataFrame, rng_seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    pools, tests = [], []
    for _, group in frame.groupby("label"):
        group = group.sample(frac=1.0, random_state=rng_seed)
        n_pool = min(POOL_MAX, len(group) // 2)
        pools.append(group.iloc[:n_pool])
        tests.append(group.iloc[n_pool:])
    return pd.concat(pools), pd.concat(tests)


def draw(pool: pd.DataFrame, k: int, seed: int) -> pd.DataFrame | None:
    """k examples, split as evenly as possible across the labels present."""
    labels = sorted(pool["label"].unique())
    per = [k // len(labels) + (1 if i < k % len(labels) else 0) for i in range(len(labels))]
    parts = []
    for label, n in zip(labels, per):
        group = pool[pool["label"] == label]
        if n > len(group):
            return None
        parts.append(group.sample(n=n, random_state=seed))
    return pd.concat(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--k", default="0,10,50,200,1000")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--sources", default="", help="comma-separated subset of held-out sources")
    args = parser.parse_args()
    ks = [int(v) for v in args.k.split(",")]
    OUT.mkdir(parents=True, exist_ok=True)

    corpus = pd.read_parquet(DATA / "corpus.parquet")
    train = corpus[corpus["role"] == "train_pool"].reset_index(drop=True)
    sources = [s.strip() for s in args.sources.split(",") if s.strip()] or sorted(train["source"].unique())

    rows = []
    out_csv = OUT / "fewshot.csv"
    for source in sources:
        held = train[train["source"] == source]
        rest = train[train["source"] != source]
        pool, test = split_source(held, SEED)
        y_test = test["label"].to_numpy()
        log(f"{source}: pool {len(pool):,}, test {len(test):,}")
        for k in ks:
            repeats = 1 if k == 0 else args.repeats
            for r in range(repeats):
                added = pool.iloc[0:0] if k == 0 else draw(pool, k, SEED + r)
                if added is None:
                    log(f"  {source}: k={k} exceeds the adaptation pool; skipped")
                    break
                texts = pd.concat([rest["text"], added["text"]])
                labels = np.concatenate([rest["label"].to_numpy(), added["label"].to_numpy()])
                s = fit_score(texts, labels, test["text"])
                rows.append({"source": source, "k": k, "repeat": r, "n_test": int(len(test)),
                             **evaluate(y_test, s)})
                log(f"  {source}: k={k} repeat {r + 1}/{repeats} utility {rows[-1]['utility']:.3f}")
            pd.DataFrame(rows).to_csv(out_csv, index=False)

    table = pd.DataFrame(rows)
    summary = table.groupby(["source", "kind", "k"]).agg(
        utility_mean=("utility", "mean"), utility_std=("utility", "std"),
        detection_rate=("detection_rate", "mean"), false_positive_rate=("false_positive_rate", "mean"),
        roc_auc=("roc_auc", "mean")).reset_index()
    summary.to_csv(OUT / "fewshot_summary.csv", index=False)
    (OUT / "run_config.json").write_text(json.dumps({"k": ks, "repeats": args.repeats, "pool_max": POOL_MAX,
                                                     "seed": SEED}, indent=2), encoding="utf-8")
    log("utility by source and k:\n" + summary.pivot_table(index="source", columns="k",
                                                            values="utility_mean").round(3).to_string())


if __name__ == "__main__":
    main()
