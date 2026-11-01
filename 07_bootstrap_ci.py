"""Bootstrap confidence intervals for the random-versus-LODO comparison.

Reads the out-of-fold scores written by script 03 and reports 95% percentile
intervals for

* pooled ROC-AUC under each protocol and for the gap (LODO minus random),
  using a paired, stratified record-level bootstrap: records are resampled
  within every source x label cell, and the same resampled records are scored
  under both protocols;
* the same gap under a source-level (cluster) bootstrap, in which whole
  sources are resampled with replacement. This answers a different question:
  how much the gap could change with another draw of public sources;
* per-source metrics under LODO (ROC-AUC for mixed sources, detection rate for
  attack-only sources, false-positive rate for benign-only sources).

Pairing: script 03 stores random-split scores in corpus order and LODO scores
grouped by source (sorted), keeping corpus order within each source. A stable
sort of the random-split rows by source therefore aligns the two protocols
record by record; the alignment is checked on the labels.

Usage
-----
    python scripts/07_bootstrap_ci.py
    python scripts/07_bootstrap_ci.py --n-boot 2000
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = RESULTS / "bootstrap"
SEED = 42

SCORE_FILES = {
    "surface": ("main/scores_surface.parquet", 0.5, ">="),
    "word_lr": ("main/scores_word_lr.parquet", 0.5, ">="),
    "char_svm": ("main/scores_char_svm.parquet", 0.0, ">"),
    "minilm_lr": ("minilm/scores_minilm_lr.parquet", 0.5, ">="),
    "finetuned": ("finetune/scores_finetuned.parquet", 0.5, ">="),
}


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def ci(values: np.ndarray) -> tuple[float, float]:
    values = values[~np.isnan(values)]
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def load_pair(path: Path) -> pd.DataFrame:
    """Return one row per record with the random and LODO scores side by side."""
    frame = pd.read_parquet(path)
    rnd = frame[frame["protocol"] == "random"].reset_index(drop=True)
    lodo = frame[frame["protocol"] == "lodo"].reset_index(drop=True)
    rnd = rnd.iloc[np.argsort(rnd["source"].to_numpy(), kind="stable")].reset_index(drop=True)
    if not (rnd["source"].equals(lodo["source"]) and rnd["label"].equals(lodo["label"])):
        raise ValueError(f"{path.name}: random and LODO rows could not be aligned")
    return pd.DataFrame({"source": lodo["source"], "label": lodo["label"].astype(int),
                         "s_random": rnd["score"].to_numpy(), "s_lodo": lodo["score"].to_numpy()})


def record_bootstrap(pair: pd.DataFrame, n_boot: int, rng: np.random.Generator) -> dict:
    cells = [g.index.to_numpy() for _, g in pair.groupby(["source", "label"])]
    y = pair["label"].to_numpy()
    sr, sl = pair["s_random"].to_numpy(), pair["s_lodo"].to_numpy()
    auc_r, auc_l = np.empty(n_boot), np.empty(n_boot)
    for b in range(n_boot):
        idx = np.concatenate([rng.choice(c, size=len(c), replace=True) for c in cells])
        auc_r[b] = roc_auc_score(y[idx], sr[idx])
        auc_l[b] = roc_auc_score(y[idx], sl[idx])
    gap = auc_l - auc_r
    return {"auc_random": float(roc_auc_score(y, sr)), "auc_random_ci": ci(auc_r),
            "auc_lodo": float(roc_auc_score(y, sl)), "auc_lodo_ci": ci(auc_l),
            "gap": float(roc_auc_score(y, sl) - roc_auc_score(y, sr)), "gap_ci_record": ci(gap),
            "gap_boot_max": float(gap.max())}


def source_bootstrap(pair: pd.DataFrame, n_boot: int, rng: np.random.Generator) -> tuple[float, float]:
    groups = {s: g for s, g in pair.groupby("source")}
    names = sorted(groups)
    gaps = np.full(n_boot, np.nan)
    for b in range(n_boot):
        draw = pd.concat([groups[s] for s in rng.choice(names, size=len(names), replace=True)])
        y = draw["label"].to_numpy()
        if y.min() == y.max():
            continue
        gaps[b] = roc_auc_score(y, draw["s_lodo"]) - roc_auc_score(y, draw["s_random"])
    return ci(gaps)


def per_source(pair: pd.DataFrame, threshold: float, op: str, n_boot: int, rng: np.random.Generator) -> list[dict]:
    rows = []
    for source, g in pair.groupby("source"):
        y, s = g["label"].to_numpy(), g["s_lodo"].to_numpy()
        pred = (s > threshold) if op == ">" else (s >= threshold)
        labels = set(np.unique(y))
        if labels == {0, 1}:
            pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
            boots = np.empty(n_boot)
            for b in range(n_boot):
                idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
                boots[b] = roc_auc_score(y[idx], s[idx])
            rows.append({"source": source, "metric": "roc_auc", "value": float(roc_auc_score(y, s)),
                         "ci_low": ci(boots)[0], "ci_high": ci(boots)[1], "n": len(y)})
        else:
            metric = "detection_rate" if labels == {1} else "false_positive_rate"
            boots = np.array([pred[rng.integers(0, len(pred), len(pred))].mean() for _ in range(n_boot)])
            rows.append({"source": source, "metric": metric, "value": float(pred.mean()),
                         "ci_low": ci(boots)[0], "ci_high": ci(boots)[1], "n": len(y)})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-boot", type=int, default=1000)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    pooled_rows, source_rows = [], []
    for name, (rel, threshold, op) in SCORE_FILES.items():
        path = RESULTS / rel
        if not path.exists():
            log(f"{name}: {rel} not found, skipped")
            continue
        rng = np.random.default_rng(SEED)
        pair = load_pair(path)
        log(f"{name}: {len(pair):,} paired records")
        row = {"detector": name, **record_bootstrap(pair, args.n_boot, rng)}
        row["gap_ci_source"] = source_bootstrap(pair, args.n_boot, rng)
        pooled_rows.append(row)
        source_rows += [{"detector": name, **r} for r in per_source(pair, threshold, op, args.n_boot, rng)]
        log(f"  AUC random {row['auc_random']:.4f} {row['auc_random_ci']}, LODO {row['auc_lodo']:.4f} "
            f"{row['auc_lodo_ci']}, gap {row['gap']:.4f} record CI {row['gap_ci_record']} "
            f"source CI {row['gap_ci_source']}")

    pooled = pd.DataFrame(pooled_rows)
    for col in ("auc_random_ci", "auc_lodo_ci", "gap_ci_record", "gap_ci_source"):
        pooled[col + "_low"] = pooled[col].map(lambda t: t[0])
        pooled[col + "_high"] = pooled[col].map(lambda t: t[1])
    pooled = pooled.drop(columns=["auc_random_ci", "auc_lodo_ci", "gap_ci_record", "gap_ci_source"])
    pooled.to_csv(OUT / "pooled_ci.csv", index=False)
    pd.DataFrame(source_rows).to_csv(OUT / "per_source_ci.csv", index=False)
    (OUT / "run_config.json").write_text(json.dumps({"n_boot": args.n_boot, "seed": SEED}, indent=2), encoding="utf-8")
    log("pooled:\n" + pooled.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
