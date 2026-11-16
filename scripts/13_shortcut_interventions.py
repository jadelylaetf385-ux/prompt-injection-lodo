"""Controlled interventions on suspected shortcuts (revision experiment).

The word TF-IDF + logistic-regression detector of script 03 is retrained under
LODO with one change at a time, and the change in source identification,
within-source discrimination and held-out detection is measured. Each variant
removes one suspected cue while leaving the attack content in place as far as
possible; examples of transformed attacks are written out for inspection.

baseline        texts unchanged (retrained here, same settings as script 03)
mask_targets    competition target strings removed from every text, in
                training and test alike: "pwned" (HackAPrompt target phrase),
                "contact@contact.com", "contact" and "confirmation"
                (LLMail-Inject goal), "password(s)" (Gandalf/Mosscap goal)
length_matched  training texts subsampled so that attack and benign texts have
                the same length distribution (equal counts per label within
                each of 20 log-length quantile bins); test texts unchanged
mixed_only      training restricted to the four sources that contain both
                labels, so that no training source identifies its label

Outputs (results/interventions/): summary.csv (pooled AUC, macro within-source
AUC, detection and false-positive rates per held-out source, NotInject FPR,
source-identification accuracy), per_source.csv, masked_examples.csv.

Usage
-----
    python scripts/13_shortcut_interventions.py
    python scripts/13_shortcut_interventions.py --variants mask_targets,mixed_only
"""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RES = ROOT / "results"
OUT = RES / "interventions"
SEED = 42
MIXED = ["deepset", "jayavibhav", "safeguard", "spml"]
TARGETS = re.compile(r"contact@contact\.com|\bpwn+e*d\b|\bcontact\b|\bconfirmation\b|\bpasswords?\b", re.I)


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def mask(texts: pd.Series) -> pd.Series:
    return texts.str.replace(TARGETS, " ", regex=True).str.replace(r"\s+", " ", regex=True).str.strip()


def fit_word(texts, y):
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=200000, sublinear_tf=True, lowercase=True)
    m = LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear").fit(vec.fit_transform(texts), y)
    return lambda t: m.predict_proba(vec.transform(t))[:, 1]


def length_matched(frame: pd.DataFrame, bins: int = 20) -> pd.DataFrame:
    loglen = np.log1p(frame["text"].str.len())
    edges = np.unique(np.quantile(loglen, np.linspace(0, 1, bins + 1)))
    b = np.clip(np.digitize(loglen, edges[1:-1]), 0, len(edges) - 2)
    keep = []
    for _, g in frame.assign(bin=b).groupby("bin"):
        n = g["label"].value_counts()
        if len(n) < 2:
            continue
        m = n.min()
        keep += [g[g["label"] == lab].sample(n=m, random_state=SEED) for lab in (0, 1)]
    return pd.concat(keep)


def balanced_sample(train: pd.DataFrame, total: int = 60000) -> pd.DataFrame:  # as in script 03
    per = total // train["source"].nunique()
    s = train.sample(frac=1.0, random_state=SEED)
    return s[s.groupby("source").cumcount() < per]


def source_id_accuracy(texts: pd.Series, sources: np.ndarray) -> float:
    preds = np.empty(len(texts), dtype=object)
    for tr, te in StratifiedKFold(3, shuffle=True, random_state=SEED).split(texts, sources):
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=200000, sublinear_tf=True)
        clf = LinearSVC(C=0.5, max_iter=5000).fit(vec.fit_transform(texts.iloc[tr]), sources[tr])
        preds[te] = clf.predict(vec.transform(texts.iloc[te]))
    return float(accuracy_score(sources, preds))


def evaluate(name: str, lodo: pd.DataFrame, ni_fpr: float, sid: float, n_train: list[int]) -> tuple[dict, list]:
    rows = []
    for source, g in lodo.groupby("source"):
        y, s = g["label"].to_numpy(), g["score"].to_numpy()
        pred = s >= 0.5
        rows.append({"variant": name, "source": source,
                     "roc_auc": roc_auc_score(y, s) if len(set(y)) == 2 else np.nan,
                     "detection_rate": pred[y == 1].mean() if y.sum() else np.nan,
                     "false_positive_rate": pred[y == 0].mean() if (y == 0).any() else np.nan})
    per = pd.DataFrame(rows)
    y, s = lodo["label"].to_numpy(), lodo["score"].to_numpy()
    summary = {"variant": name, "pooled_auc": roc_auc_score(y, s),
               "macro_within_auc_mixed": per.loc[per["source"].isin(MIXED), "roc_auc"].mean(),
               "pooled_detection_rate": (s[y == 1] >= 0.5).mean(), "pooled_fpr": (s[y == 0] >= 0.5).mean(),
               "notinject_fpr": ni_fpr, "source_id_accuracy": sid,
               "median_training_size": float(np.median(n_train))}
    for _, r in per.iterrows():
        val = r["roc_auc"] if r["source"] in MIXED else (
            r["detection_rate"] if not np.isnan(r["detection_rate"]) else r["false_positive_rate"])
        summary[f"{r['source']}"] = val
    return summary, rows


def run_variant(name: str, train: pd.DataFrame, notinject: pd.DataFrame, sid_sample: pd.DataFrame):
    texts = mask(train["text"]) if name == "mask_targets" else train["text"]
    ni_texts = mask(notinject["text"]) if name == "mask_targets" else notinject["text"]
    data = train.assign(text=texts)
    parts, sizes = [], []
    for source in sorted(train["source"].unique()):
        held = data["source"] == source
        tr = data[~held]
        if name == "mixed_only":
            tr = tr[tr["source"].isin(MIXED)]
        elif name == "length_matched":
            tr = length_matched(tr)
        sizes.append(len(tr))
        f = fit_word(tr["text"], tr["label"].to_numpy())
        parts.append(data.loc[held, ["source", "label"]].assign(score=f(data.loc[held, "text"])))
        log(f"  {name}: held out {source} (trained on {len(tr):,})")
    full = data if name != "mixed_only" else data[data["source"].isin(MIXED)]
    if name == "length_matched":
        full = length_matched(data)
    ni_fpr = float((fit_word(full["text"], full["label"].to_numpy())(ni_texts) >= 0.5).mean())
    sid = source_id_accuracy(mask(sid_sample["text"]) if name == "mask_targets" else sid_sample["text"],
                             sid_sample["source"].to_numpy()) if name in ("baseline", "mask_targets") else np.nan
    return evaluate(name, pd.concat(parts), ni_fpr, sid, sizes)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variants", default="baseline,mask_targets,length_matched,mixed_only")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    corpus = pd.read_parquet(DATA / "corpus.parquet")
    train = corpus[corpus["role"] == "train_pool"].reset_index(drop=True)
    notinject = corpus[corpus["source"] == "notinject"].reset_index(drop=True)
    sid_sample = balanced_sample(train)

    hits = train["text"].str.contains(TARGETS)
    ex = train[hits & (train["label"] == 1)].groupby("source").head(5)
    pd.DataFrame({"source": ex["source"], "original": ex["text"].str[:300],
                  "masked": mask(ex["text"]).str[:300]}).to_csv(OUT / "masked_examples.csv", index=False)
    log("share of texts containing a target string, by source and label:\n"
        + train.assign(hit=hits).groupby(["source", "label"])["hit"].mean().round(3).to_string())

    summaries, rows = [], []
    for name in [v.strip() for v in args.variants.split(",") if v.strip()]:
        log(f"=== {name} ===")
        s, r = run_variant(name, train, notinject, sid_sample)
        summaries.append(s)
        rows += r
        pd.DataFrame(summaries).to_csv(OUT / "summary.csv", index=False, float_format="%.6f")
        pd.DataFrame(rows).to_csv(OUT / "per_source.csv", index=False, float_format="%.6f")
    log("summary:\n" + pd.DataFrame(summaries).round(3).T.to_string())


if __name__ == "__main__":
    main()
