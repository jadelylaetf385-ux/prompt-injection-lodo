"""Detection at low false-positive targets chosen on training sources only
(revision experiment).

For every held-out source s, an inner leave-one-source-out loop over the nine
training sources scores each training source with a model that did not see it.
The threshold for a target false-positive rate alpha is the (1 - alpha)
quantile of the pooled inner benign scores, i.e. the threshold that would have
produced a false-positive rate of alpha on training sources the model had not
seen. The outer model, trained on all nine training sources, is then applied to
s with that threshold. No label from s is used. The default threshold (0.5 for
logistic regression, 0 for the SVM) is reported for comparison.

For NotInject the same procedure is run over all ten sources.

Rates are reported with Wilson 95% intervals.

Usage
-----
    python scripts/14_low_fpr_operating_points.py
    python scripts/14_low_fpr_operating_points.py --detector char_svm --targets 0.01,0.05
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "results" / "operating_points"
SEED = 42


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def fit(detector: str, texts, y):
    if detector == "word_lr":
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=200000, sublinear_tf=True, lowercase=True)
        m = LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear").fit(vec.fit_transform(texts), y)
        return lambda t: m.predict_proba(vec.transform(t))[:, 1]
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, max_features=300000,
                          sublinear_tf=True, lowercase=True)
    m = LinearSVC(C=0.5, class_weight="balanced", max_iter=5000).fit(vec.fit_transform(texts), y)
    return lambda t: m.decision_function(vec.transform(t))


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return np.nan, np.nan
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def thresholds(detector: str, frame: pd.DataFrame, targets: list[float]) -> dict:
    benign = []
    for u in sorted(frame["source"].unique()):
        held = frame["source"] == u
        if (frame.loc[held, "label"] == 0).sum() == 0:
            continue  # an attack-only source contributes no benign scores
        f = fit(detector, frame.loc[~held, "text"], frame.loc[~held, "label"].to_numpy())
        benign.append(f(frame.loc[held & (frame["label"] == 0), "text"]))
    benign = np.concatenate(benign)
    return {a: float(np.quantile(benign, 1 - a)) for a in targets}


def rates(y: np.ndarray, s: np.ndarray, t: float, op: str) -> dict:
    pred = (s > t) if op == ">" else (s >= t)
    out = {}
    if y.sum():
        k, n = int(pred[y == 1].sum()), int(y.sum())
        out.update(detection_rate=k / n, dr_low=wilson(k, n)[0], dr_high=wilson(k, n)[1])
    if (y == 0).any():
        k, n = int(pred[y == 0].sum()), int((y == 0).sum())
        out.update(false_positive_rate=k / n, fpr_low=wilson(k, n)[0], fpr_high=wilson(k, n)[1])
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--detector", default="word_lr", choices=["word_lr", "char_svm"])
    parser.add_argument("--targets", default="0.01,0.05")
    args = parser.parse_args()
    targets = [float(t) for t in args.targets.split(",")]
    default_t, op = (0.5, ">=") if args.detector == "word_lr" else (0.0, ">")
    OUT.mkdir(parents=True, exist_ok=True)
    corpus = pd.read_parquet(DATA / "corpus.parquet")
    train = corpus[corpus["role"] == "train_pool"].reset_index(drop=True)
    notinject = corpus[corpus["source"] == "notinject"].reset_index(drop=True)

    rows, pooled = [], []
    for source in sorted(train["source"].unique()):
        held = train["source"] == source
        tr = train[~held]
        th = thresholds(args.detector, tr, targets)
        s = fit(args.detector, tr["text"], tr["label"].to_numpy())(train.loc[held, "text"])
        y = train.loc[held, "label"].to_numpy()
        pooled.append(pd.DataFrame({"source": source, "label": y, "score": s} | {f"t_{a}": th[a] for a in targets}))
        for name, t in [("default", default_t)] + [(f"fpr_{a}", th[a]) for a in targets]:
            rows.append({"detector": args.detector, "held_out": source, "operating_point": name, "threshold": t,
                         "n": len(y), **rates(y, s, t, op)})
        log(f"{source}: thresholds " + ", ".join(f"{a}: {th[a]:.4f}" for a in targets))
        pd.DataFrame(rows).to_csv(OUT / f"per_source_{args.detector}.csv", index=False, float_format="%.6f")

    p = pd.concat(pooled)
    summary = []
    for name in ["default"] + [f"fpr_{a}" for a in targets]:
        t = default_t if name == "default" else p[f"t_{name[4:]}"].to_numpy()
        pred = (p["score"].to_numpy() > t) if op == ">" else (p["score"].to_numpy() >= t)
        y = p["label"].to_numpy()
        k1, n1, k0, n0 = int(pred[y == 1].sum()), int(y.sum()), int(pred[y == 0].sum()), int((y == 0).sum())
        summary.append({"detector": args.detector, "operating_point": name,
                        "pooled_detection_rate": k1 / n1, "dr_low": wilson(k1, n1)[0], "dr_high": wilson(k1, n1)[1],
                        "pooled_fpr": k0 / n0, "fpr_low": wilson(k0, n0)[0], "fpr_high": wilson(k0, n0)[1]})
    th_all = thresholds(args.detector, train, targets)
    s_ni = fit(args.detector, train["text"], train["label"].to_numpy())(notinject["text"])
    for row in summary:
        t = default_t if row["operating_point"] == "default" else th_all[float(row["operating_point"][4:])]
        k = int(((s_ni > t) if op == ">" else (s_ni >= t)).sum())
        row.update(notinject_fpr=k / len(s_ni), ni_low=wilson(k, len(s_ni))[0], ni_high=wilson(k, len(s_ni))[1])
    pd.DataFrame(summary).to_csv(OUT / f"summary_{args.detector}.csv", index=False, float_format="%.6f")
    log("summary:\n" + pd.DataFrame(summary).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
