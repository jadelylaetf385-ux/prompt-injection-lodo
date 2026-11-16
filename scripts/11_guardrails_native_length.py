"""Released guardrails at their native input length, and truncation statistics
(revision experiment).

Script 04 scored every guardrail with a common limit of 256 tokens. This script
rescores the same evaluation sample (results/offtheshelf/evaluation_sample.csv)
with the maximum length supported by each model's published configuration and
reports, per source and model, how many texts exceed 256 and 512 tokens and on
which side the tokenizer truncates.

Outputs (results/offtheshelf_native/):
    truncation_stats.csv          per model and source: n, median tokens,
                                  share > 256, share > 512, truncation side
    scores_<model>_len<L>.parquet key, source, label, score
    per_source_metrics.csv        AUC / DR / FPR per model, source and length

A GPU makes this fast (minutes); on a CPU expect about 1.5 h per model.

Usage
-----
    python scripts/11_guardrails_native_length.py
    python scripts/11_guardrails_native_length.py --models piguard --max-length 512
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
SAMPLE = ROOT / "results" / "offtheshelf" / "evaluation_sample.csv"
OUT = ROOT / "results" / "offtheshelf_native"

MODELS = {
    "protectai_v2": "protectai/deberta-v3-base-prompt-injection-v2",
    "deepset_deberta": "deepset/deberta-v3-base-injection",
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


def token_stats(tok, frame: pd.DataFrame, name: str) -> pd.DataFrame:
    lengths = [len(tok(t, truncation=False)["input_ids"]) for t in frame["text"].tolist()]
    f = frame.assign(tokens=lengths)
    rows = []
    for source, g in f.groupby("source"):
        rows.append({"model": name, "source": source, "n": len(g), "median_tokens": float(g["tokens"].median()),
                     "share_over_256": float((g["tokens"] > 256).mean()),
                     "share_over_512": float((g["tokens"] > 512).mean()),
                     "truncation_side": getattr(tok, "truncation_side", "right"),
                     "model_max_length": int(min(getattr(tok, "model_max_length", 10**6), 10**6))})
    return pd.DataFrame(rows)


@torch.no_grad()
def score(model, tok, texts: list[str], max_length: int, batch_size: int, index: int, device: str) -> np.ndarray:
    out = np.zeros(len(texts), dtype=np.float32)
    order = np.argsort([len(t) for t in texts])
    for start in range(0, len(order), batch_size):
        idx = order[start:start + batch_size]
        enc = tok([texts[i] for i in idx], truncation=True, max_length=max_length, padding=True,
                  return_tensors="pt").to(device)
        out[idx] = torch.softmax(model(**enc).logits.float(), dim=-1)[:, index].cpu().numpy()
    return out


def metrics(group: pd.DataFrame) -> dict:
    y, s = group["label"].to_numpy(), group["score"].to_numpy()
    pred = s >= 0.5
    labels = set(np.unique(y))
    if labels == {0, 1}:
        return {"kind": "mixed", "roc_auc": roc_auc_score(y, s), "detection_rate": pred[y == 1].mean(),
                "false_positive_rate": pred[y == 0].mean()}
    if labels == {1}:
        return {"kind": "attack_only", "detection_rate": pred.mean()}
    return {"kind": "benign_only", "false_positive_rate": pred.mean()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", default=",".join(MODELS))
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    corpus = pd.read_parquet(DATA / "corpus.parquet")[["key", "text", "source", "label"]]
    sample = pd.read_csv(SAMPLE)[["key", "source"]].merge(corpus, on=["key", "source"], how="left")
    if sample["text"].isna().any():
        raise ValueError("some sampled keys were not found in the corpus")
    log(f"evaluation sample: {len(sample):,} texts; device {device}")

    stats, per_rows = [], []
    for name in [m.strip() for m in args.models.split(",") if m.strip()]:
        repo = MODELS[name]
        log(f"{name}: loading {repo}")
        tok = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
        model = AutoModelForSequenceClassification.from_pretrained(repo, trust_remote_code=True).to(device).eval()
        stats.append(token_stats(tok, sample, name))
        pd.concat(stats).to_csv(OUT / "truncation_stats.csv", index=False)
        cache = OUT / f"scores_{name}_len{args.max_length}.parquet"
        if cache.exists():
            scored = pd.read_parquet(cache)
            log(f"{name}: scores loaded from cache")
        else:
            started = time.perf_counter()
            s = score(model, tok, sample["text"].tolist(), args.max_length, args.batch_size,
                      attack_index(model.config.id2label), device)
            scored = sample[["key", "source", "label"]].assign(score=s)
            scored.to_parquet(cache, index=False)
            log(f"{name}: scored at {args.max_length} tokens in {time.perf_counter() - started:.0f}s")
        for source, g in scored.groupby("source"):
            per_rows.append({"model": name, "max_length": args.max_length, "source": source, "n": len(g),
                             **metrics(g)})
        pd.DataFrame(per_rows).to_csv(OUT / f"per_source_metrics_len{args.max_length}.csv", index=False)
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
    log("truncation statistics:\n" + pd.concat(stats).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
