"""Evaluate publicly released prompt-injection classifiers on every source.

These detectors cannot be retrained, so they are scored as released on a
stratified sample of each source and on NotInject. Their training data is not
fully documented and may include some of the public sources used here; a high
score on a source can therefore reflect exposure rather than generalisation,
and results are reported with that caveat.

Scores are cached per model, so an interrupted run resumes where it stopped.

Usage
-----
    python scripts/04_offtheshelf_detectors.py
    python scripts/04_offtheshelf_detectors.py --models protectai_v2,promptguard2_22m --per-source 2000
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "results" / "offtheshelf"
SEED = 42

MODELS = {
    "protectai_v2": "protectai/deberta-v3-base-prompt-injection-v2",
    "deepset_deberta": "deepset/deberta-v3-base-injection",
    "promptguard2_22m": "meta-llama/Llama-Prompt-Guard-2-22M",
    "promptguard2_86m": "meta-llama/Llama-Prompt-Guard-2-86M",
    "piguard": "leolee99/PIGuard",
}
ATTACK_LABEL_HINTS = ("injection", "malicious", "jailbreak", "unsafe", "label_1")


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def attack_index(id2label: dict) -> int:
    labels = {int(k): str(v).lower() for k, v in id2label.items()}
    for hint in ATTACK_LABEL_HINTS:
        for index, name in labels.items():
            if hint in name:
                return index
    raise ValueError(f"cannot identify the attack label in {id2label}")


class HFScorer:
    """Batch inference on CPU; returns the probability of the attack class."""

    def __init__(self, repo: str, max_length: int, batch_size: int):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(repo, trust_remote_code=True)
        self.model.eval()
        self.index = attack_index(self.model.config.id2label)
        self.labels = dict(self.model.config.id2label)
        self.max_length = max_length
        self.batch_size = batch_size
        torch.set_num_threads(max(1, torch.get_num_threads()))

    def score(self, texts: list[str]) -> np.ndarray:
        out = []
        with self.torch.no_grad():
            for start in range(0, len(texts), self.batch_size):
                batch = texts[start:start + self.batch_size]
                enc = self.tok(batch, truncation=True, max_length=self.max_length, padding=True, return_tensors="pt")
                logits = self.model(**enc).logits
                out.append(self.torch.softmax(logits, dim=-1)[:, self.index].cpu().numpy())
        return np.concatenate(out) if out else np.array([])


def build_sample(corpus: pd.DataFrame, per_source: int) -> pd.DataFrame:
    train = corpus[corpus["role"] == "train_pool"].sample(frac=1.0, random_state=SEED)
    sample = train[train.groupby(["source", "label"]).cumcount() < per_source]
    return pd.concat([sample, corpus[corpus["source"] == "notinject"]], ignore_index=True)


def metrics(group: pd.DataFrame, threshold: float = 0.5) -> dict:
    y, s = group["label"].to_numpy(), group["score"].to_numpy()
    pred = (s >= threshold).astype(int)
    labels = set(np.unique(y))
    row = {"n": int(len(y))}
    if labels == {0, 1}:
        row.update(kind="mixed", roc_auc=float(roc_auc_score(y, s)),
                   average_precision=float(average_precision_score(y, s)),
                   f1=float(f1_score(y, pred, zero_division=0)), accuracy=float(accuracy_score(y, pred)),
                   detection_rate=float(pred[y == 1].mean()), false_positive_rate=float(pred[y == 0].mean()))
    elif labels == {1}:
        row.update(kind="attack_only", detection_rate=float(pred.mean()))
    else:
        row.update(kind="benign_only", false_positive_rate=float(pred.mean()))
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", default=",".join(MODELS))
    parser.add_argument("--per-source", type=int, default=2000, help="records per source and label")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    corpus = pd.read_parquet(DATA / "corpus.parquet")
    sample = build_sample(corpus, args.per_source)
    sample[["key", "source", "label"]].to_csv(OUT / "evaluation_sample.csv", index=False)
    log(f"evaluation sample: {len(sample):,} texts ({args.per_source} per source and label, plus NotInject)")

    rows, info = [], {}
    for name in [m.strip() for m in args.models.split(",") if m.strip()]:
        repo = MODELS[name]
        cache = OUT / f"scores_{name}.parquet"
        if cache.exists():
            scored = pd.read_parquet(cache)
            log(f"{name}: scores loaded from cache")
        else:
            log(f"{name}: loading {repo}")
            try:
                scorer = HFScorer(repo, args.max_length, args.batch_size)
            except Exception as error:
                log(f"{name}: could not be loaded ({type(error).__name__}: {str(error)[:200]})")
                info[name] = {"repo": repo, "status": "not loaded", "error": str(error)[:300]}
                continue
            info[name] = {"repo": repo, "status": "loaded", "id2label": scorer.labels,
                          "attack_index": scorer.index}
            started = time.perf_counter()
            scored = sample[["key", "source", "label"]].copy()
            parts = []
            for source, group in sample.groupby("source", sort=True):
                parts.append(pd.Series(scorer.score(group["text"].tolist()), index=group.index))
                log(f"  {name}: scored {source} ({len(group):,})")
            scored["score"] = pd.concat(parts).sort_index().to_numpy()
            info[name]["seconds"] = round(time.perf_counter() - started, 1)
            scored.to_parquet(cache, index=False)

        for source, group in scored.groupby("source"):
            rows.append({"detector": name, "source": source, **metrics(group)})
        pd.DataFrame(rows).to_csv(OUT / "per_source_metrics.csv", index=False)
        (OUT / "models.json").write_text(json.dumps(info, indent=2, default=str), encoding="utf-8")

    if rows:
        table = pd.DataFrame(rows)
        table["value"] = table["roc_auc"].where(table["kind"] == "mixed",
                         table["detection_rate"].where(table["kind"] == "attack_only", table["false_positive_rate"]))
        log("AUC (mixed) / detection rate (attack-only) / false-positive rate (benign-only):\n"
            + table.pivot_table(index="source", columns="detector", values="value").round(3).to_string())


if __name__ == "__main__":
    main()
