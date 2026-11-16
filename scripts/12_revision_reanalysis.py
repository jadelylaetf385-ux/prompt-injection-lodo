"""Re-analyses requested in the first review round (no GPU needed).

Uses the corpus and the saved per-record scores of scripts 03, 04 and 08.

within      ROC-AUC inside every mixed-label source, its macro average over the
            four mixed sources, under both protocols.
pairs       Exact decomposition of pooled ROC-AUC into (attack source,
            benign source) pairs: pooled AUC = sum_ij w_ij * AUC_ij with
            w_ij = n+_i * n-_j / (N+ * N-). Pairs with i == j measure
            discrimination within a source; i != j pairs measure how scores of
            different sources line up. The random-to-LODO change is split the
            same way.
calibration Training-only calibration sensitivity for the word and character
            detectors: every fold model is refitted, and its held-out scores are
            mapped (a) to z-scores and (b) to the empirical CDF of benign
            training scores, both computed on that fold's training data only.
            Both maps are monotone within a fold, so within-source AUC is
            unchanged and any change in pooled AUC is due to cross-fold score
            alignment.
sample      Trained detectors (LODO) and released guardrails on the identical
            evaluation sample of script 04, with paired bootstrap intervals for
            per-source differences.
capped      Word TF-IDF retrained under the 3,000-per-cell cap of script 08,
            per source and protocol.
table       Machine-readable table of every detector, source, protocol and
            training condition with n, DR, FPR, ROC-AUC and AP at full precision.

Usage
-----
    python scripts/12_revision_reanalysis.py                 # all parts
    python scripts/12_revision_reanalysis.py --parts within,pairs,sample,table
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
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RES = ROOT / "results"
OUT = RES / "revision"
SEED = 42
N_BOOT = 1000
CAP = 3000

TRAINED = {  # name: (score file, threshold, operator, training condition)
    "surface": ("main/scores_surface.parquet", 0.5, ">=", "full"),
    "word_lr": ("main/scores_word_lr.parquet", 0.5, ">=", "full"),
    "char_svm": ("main/scores_char_svm.parquet", 0.0, ">", "full"),
    "minilm_lr": ("minilm/scores_minilm_lr.parquet", 0.5, ">=", "full"),
    "finetuned": ("finetune/scores_finetuned.parquet", 0.5, ">=", "capped_3000"),
}
GUARDRAILS = ["protectai_v2", "deepset_deberta", "piguard"]


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def predict(s: np.ndarray, threshold: float, op: str) -> np.ndarray:
    return (s > threshold) if op == ">" else (s >= threshold)


# --------------------------------------------------------------------------- #
# Loading: one long frame with key, source, label, protocol, detector, score
# --------------------------------------------------------------------------- #

def load_scores(train: pd.DataFrame) -> pd.DataFrame:
    lodo_order = pd.concat([train[train["source"] == s] for s in sorted(train["source"].unique())])
    frames = []
    for name, (rel, *_rest) in TRAINED.items():
        path = RES / rel
        if not path.exists():
            log(f"{name}: {rel} missing, skipped")
            continue
        d = pd.read_parquet(path)
        rnd = d[d["protocol"] == "random"].reset_index(drop=True)
        lodo = d[d["protocol"] == "lodo"].reset_index(drop=True)
        if not (rnd["source"].equals(train["source"]) and lodo["source"].equals(lodo_order["source"].reset_index(drop=True))):
            raise ValueError(f"{name}: score order does not match the corpus")
        frames.append(pd.DataFrame({"key": train["key"], "source": train["source"], "label": train["label"],
                                    "protocol": "random", "detector": name, "score": rnd["score"].to_numpy()}))
        frames.append(pd.DataFrame({"key": lodo_order["key"].to_numpy(), "source": lodo_order["source"].to_numpy(),
                                    "label": lodo_order["label"].to_numpy(), "protocol": "lodo", "detector": name,
                                    "score": lodo["score"].to_numpy()}))
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------- #
# Within-source AUC and pair decomposition
# --------------------------------------------------------------------------- #

def pair_auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """P(pos > neg) + 0.5 P(pos == neg), exact via sorting."""
    neg = np.sort(neg)
    below = np.searchsorted(neg, pos, side="left")
    ties = np.searchsorted(neg, pos, side="right") - below
    return float((below + 0.5 * ties).sum() / (len(pos) * len(neg)))


def within_and_pairs(scores: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    within, pairs, summary = [], [], []
    for (det, prot), g in scores.groupby(["detector", "protocol"]):
        mixed = [s for s, h in g.groupby("source") if h["label"].nunique() == 2]
        for s in mixed:
            h = g[g["source"] == s]
            within.append({"detector": det, "protocol": prot, "source": s,
                           "roc_auc": roc_auc_score(h["label"], h["score"])})
        pos = {s: h["score"].to_numpy() for s, h in g[g["label"] == 1].groupby("source")}
        neg = {s: h["score"].to_numpy() for s, h in g[g["label"] == 0].groupby("source")}
        n_pos, n_neg = sum(map(len, pos.values())), sum(map(len, neg.values()))
        for i, p in pos.items():
            for j, n in neg.items():
                pairs.append({"detector": det, "protocol": prot, "attack_source": i, "benign_source": j,
                              "weight": len(p) * len(n) / (n_pos * n_neg), "pair_auc": pair_auc(p, n)})
    within, pairs = pd.DataFrame(within), pd.DataFrame(pairs)
    pairs["contribution"] = pairs["weight"] * pairs["pair_auc"]
    pairs["same_source"] = pairs["attack_source"] == pairs["benign_source"]
    for (det, prot), g in pairs.groupby(["detector", "protocol"]):
        w_in = g.loc[g["same_source"], "weight"].sum()
        summary.append({"detector": det, "protocol": prot, "pooled_auc": g["contribution"].sum(),
                        "macro_mixed_auc": within[(within["detector"] == det) & (within["protocol"] == prot)]["roc_auc"].mean(),
                        "weight_same_source": w_in,
                        "auc_same_source_pairs": g.loc[g["same_source"], "contribution"].sum() / w_in,
                        "auc_cross_source_pairs": g.loc[~g["same_source"], "contribution"].sum() / (1 - w_in)})
    summary = pd.DataFrame(summary)
    # split of the random-to-LODO change into same-source and cross-source pairs (weights are identical)
    piv = pairs.pivot_table(index=["detector", "attack_source", "benign_source", "same_source"],
                            columns="protocol", values="contribution").reset_index()
    piv["delta"] = piv["lodo"] - piv["random"]
    split = piv.groupby(["detector", "same_source"])["delta"].sum().unstack()
    split.columns = ["delta_cross_source_pairs", "delta_same_source_pairs"]
    summary = summary.merge(split.reset_index(), on="detector", how="left")
    return within, pairs, summary


# --------------------------------------------------------------------------- #
# Training-only calibration sensitivity (word and character detectors)
# --------------------------------------------------------------------------- #

def fit_word(texts, y):
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=200000, sublinear_tf=True, lowercase=True)
    m = LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear").fit(vec.fit_transform(texts), y)
    return lambda t: m.predict_proba(vec.transform(t))[:, 1]


def fit_char(texts, y):
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, max_features=300000,
                          sublinear_tf=True, lowercase=True)
    m = LinearSVC(C=0.5, class_weight="balanced", max_iter=5000).fit(vec.fit_transform(texts), y)
    return lambda t: m.decision_function(vec.transform(t))


def calibration(train: pd.DataFrame) -> pd.DataFrame:
    strata = train["source"] + "_" + train["label"].astype(str)
    folds = [("random", tr, te) for tr, te in StratifiedKFold(5, shuffle=True, random_state=SEED).split(train, strata)]
    folds += [("lodo", np.where(train["source"] != s)[0], np.where(train["source"] == s)[0])
              for s in sorted(train["source"].unique())]
    rows = []
    for name, fit in (("word_lr", fit_word), ("char_svm", fit_char)):
        out = {p: {"y": [], "raw": [], "z": [], "cdf": []} for p in ("random", "lodo")}
        for prot, tr, te in folds:
            f = fit(train["text"].iloc[tr], train["label"].iloc[tr].to_numpy())
            s_tr, s_te = f(train["text"].iloc[tr]), f(train["text"].iloc[te])
            ben = np.sort(s_tr[train["label"].iloc[tr].to_numpy() == 0])
            o = out[prot]
            o["y"].append(train["label"].iloc[te].to_numpy())
            o["raw"].append(s_te)
            o["z"].append((s_te - s_tr.mean()) / s_tr.std())
            o["cdf"].append(np.searchsorted(ben, s_te, side="right") / len(ben))
            log(f"  calibration {name} {prot}: fold done")
        for prot, o in out.items():
            y = np.concatenate(o["y"])
            rows.append({"detector": name, "protocol": prot,
                         **{f"pooled_auc_{k}": roc_auc_score(y, np.concatenate(o[k])) for k in ("raw", "z", "cdf")}})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Identical-sample comparison with released guardrails
# --------------------------------------------------------------------------- #

def metric_value(y, s, thr, op):
    pred = predict(s, thr, op)
    lab = set(np.unique(y))
    if lab == {0, 1}:
        return {"roc_auc": roc_auc_score(y, s), "detection_rate": pred[y == 1].mean(),
                "false_positive_rate": pred[y == 0].mean()}
    if lab == {1}:
        return {"detection_rate": pred.mean()}
    return {"false_positive_rate": pred.mean()}


def identical_sample(scores: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    sample = pd.read_csv(RES / "offtheshelf" / "evaluation_sample.csv")
    sample = sample[sample["source"] != "notinject"]
    models = {}
    for name, (_, thr, op, _) in TRAINED.items():
        d = scores[(scores["detector"] == name) & (scores["protocol"] == "lodo")]
        if len(d):
            models[name] = (sample.merge(d[["key", "score"]], on="key", how="left"), thr, op)
    for g in GUARDRAILS:
        d = pd.read_parquet(RES / "offtheshelf" / f"scores_{g}.parquet")
        models[g] = (sample.merge(d[["key", "score"]].drop_duplicates("key"), on="key", how="left"), 0.5, ">=")
    for name, (m, *_r) in models.items():
        if m["score"].isna().any():
            raise ValueError(f"{name}: missing scores on the evaluation sample")
    rows = []
    for name, (m, thr, op) in models.items():
        for source, g in m.groupby("source"):
            rows.append({"model": name, "source": source, "n": len(g),
                         **metric_value(g["label"].to_numpy(), g["score"].to_numpy(), thr, op)})
    per = pd.DataFrame(rows)
    # paired bootstrap: character detector (LODO) minus each guardrail, same records
    rng = np.random.default_rng(SEED)
    ref, rthr, rop = models["char_svm"]
    diffs = []
    for g in GUARDRAILS:
        other, othr, oop = models[g]
        for source, idx in ref.groupby("source").groups.items():
            idx = np.asarray(list(idx))
            y = ref.loc[idx, "label"].to_numpy()
            a, b = ref.loc[idx, "score"].to_numpy(), other.loc[idx, "score"].to_numpy()
            mixed = len(set(y)) == 2
            def stat(i):
                if mixed:
                    return roc_auc_score(y[i], a[i]) - roc_auc_score(y[i], b[i])
                return predict(a[i], rthr, rop).mean() - predict(b[i], othr, oop).mean()
            pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
            boots = []
            for _ in range(N_BOOT):
                i = np.concatenate([rng.choice(pos, len(pos)) if len(pos) else pos,
                                    rng.choice(neg, len(neg)) if len(neg) else neg])
                boots.append(stat(i))
            full = np.arange(len(y))
            diffs.append({"guardrail": g, "source": source,
                          "metric": "roc_auc" if mixed else ("detection_rate" if y.min() == 1 else "false_positive_rate"),
                          "char_minus_guardrail": stat(full),
                          "ci_low": float(np.percentile(boots, 2.5)), "ci_high": float(np.percentile(boots, 97.5))})
    return per, pd.DataFrame(diffs)


# --------------------------------------------------------------------------- #
# Capped word detector per source, and the machine-readable table
# --------------------------------------------------------------------------- #

def cap(frame: pd.DataFrame) -> pd.DataFrame:
    s = frame.sample(frac=1.0, random_state=SEED)
    return s[s.groupby(["source", "label"]).cumcount() < CAP]


def capped_word(train: pd.DataFrame) -> pd.DataFrame:
    strata = train["source"] + "_" + train["label"].astype(str)
    rnd = np.zeros(len(train))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=SEED).split(train, strata):
        sub = cap(train.iloc[tr])
        rnd[te] = fit_word(sub["text"], sub["label"].to_numpy())(train["text"].iloc[te])
    parts = []
    for s in sorted(train["source"].unique()):
        held = train["source"] == s
        sub = cap(train[~held])
        parts.append(pd.DataFrame({"key": train.loc[held, "key"], "source": s, "label": train.loc[held, "label"],
                                   "score": fit_word(sub["text"], sub["label"].to_numpy())(train.loc[held, "text"])}))
    lodo = pd.concat(parts)
    return pd.concat([train[["key", "source", "label"]].assign(score=rnd, protocol="random"),
                      lodo.assign(protocol="lodo")], ignore_index=True).assign(detector="word_lr")


def results_table(scores: pd.DataFrame, capped: pd.DataFrame | None, sample_per: pd.DataFrame | None) -> pd.DataFrame:
    rows = []
    sets = [(scores, None)] + ([(capped, "capped_3000")] if capped is not None else [])
    for frame, cond_override in sets:
        for (det, prot), g in frame.groupby(["detector", "protocol"]):
            _, thr, op, cond = TRAINED[det]
            cond = cond_override or cond
            for source, h in list(g.groupby("source")) + [("ALL", g)]:
                y, s = h["label"].to_numpy(), h["score"].to_numpy()
                pred = predict(s, thr, op)
                mixed = len(set(y)) == 2
                rows.append({"detector": det, "training_condition": cond, "protocol": prot, "source": source,
                             "n": len(y), "n_attack": int(y.sum()), "n_benign": int((y == 0).sum()),
                             "detection_rate": pred[y == 1].mean() if y.sum() else np.nan,
                             "false_positive_rate": pred[y == 0].mean() if (y == 0).any() else np.nan,
                             "roc_auc": roc_auc_score(y, s) if mixed else np.nan,
                             "average_precision": average_precision_score(y, s) if mixed else np.nan,
                             "threshold": thr})
    if sample_per is not None:
        for _, r in sample_per[sample_per["model"].isin(GUARDRAILS)].iterrows():
            rows.append({"detector": r["model"], "training_condition": "released", "protocol": "evaluation_sample_256",
                         "source": r["source"], "n": r["n"], "detection_rate": r.get("detection_rate"),
                         "false_positive_rate": r.get("false_positive_rate"), "roc_auc": r.get("roc_auc"),
                         "threshold": 0.5})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--parts", default="within,pairs,calibration,sample,capped,table")
    args = parser.parse_args()
    parts = set(args.parts.split(","))
    OUT.mkdir(parents=True, exist_ok=True)
    corpus = pd.read_parquet(DATA / "corpus.parquet")
    train = corpus[corpus["role"] == "train_pool"].reset_index(drop=True)
    scores = load_scores(train)
    log(f"loaded {scores['detector'].nunique()} detectors, {len(scores):,} scored rows")

    if parts & {"within", "pairs"}:
        within, pairs, summary = within_and_pairs(scores)
        within.to_csv(OUT / "within_source_auc.csv", index=False)
        pairs.to_csv(OUT / "pooled_auc_pair_decomposition.csv", index=False)
        summary.to_csv(OUT / "pooled_auc_summary.csv", index=False)
        log("pooled-AUC decomposition:\n" + summary.round(4).to_string(index=False))
    sample_per = None
    if "sample" in parts:
        sample_per, diffs = identical_sample(scores)
        sample_per.to_csv(OUT / "identical_sample_per_source.csv", index=False)
        diffs.to_csv(OUT / "identical_sample_paired_differences.csv", index=False)
        log("identical-sample comparison written")
    capped = None
    if "capped" in parts:
        capped = capped_word(train)
        capped.to_parquet(OUT / "scores_word_lr_capped.parquet", index=False)
        log("capped word detector done")
    if "table" in parts:
        table = results_table(scores, capped, sample_per)
        table.to_csv(OUT / "results_all_detectors_sources.csv", index=False, float_format="%.6f")
        log(f"results table: {len(table)} rows")
    if "calibration" in parts:
        cal = calibration(train)
        cal.to_csv(OUT / "calibration_sensitivity.csv", index=False)
        log("calibration sensitivity:\n" + cal.round(4).to_string(index=False))
    (OUT / "run_config.json").write_text(json.dumps({"seed": SEED, "n_boot": N_BOOT, "cap": CAP,
                                                     "parts": sorted(parts)}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
