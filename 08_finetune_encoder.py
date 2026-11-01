"""Fine-tuned transformer encoder under random and leave-one-dataset-out splits.

Answers the question whether the generalisation gap of script 03 is a property
of linear models on fixed features or persists for a fine-tuned encoder. The
protocol is the same as in script 03:

random   stratified 5-fold cross-validation over source x label strata;
lodo     each source held out in turn.

For speed, the training part of every fold is subsampled to at most
--train-cap texts per source and label (the same cap for both protocols, so
the comparison stays fair); every held-out text is scored. A model trained on
the capped full pool is also scored on NotInject.

Scores are written in the format of script 03 (random rows in corpus order,
LODO rows grouped by sorted source), so script 07 can bootstrap them. Each
fold is cached, so an interrupted run (e.g. a Colab disconnect) resumes.

Needs a GPU. On Google Colab (T4) the default settings take roughly 1.5-3 h.

Usage
-----
    python scripts/08_finetune_encoder.py
    python scripts/08_finetune_encoder.py --model distilroberta-base --train-cap 3000 --max-length 256
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "results" / "finetune"
SEED = 42


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def cap(frame: pd.DataFrame, per_cell: int) -> pd.DataFrame:
    shuffled = frame.sample(frac=1.0, random_state=SEED)
    return shuffled[shuffled.groupby(["source", "label"]).cumcount() < per_cell]


class Encoder:
    def __init__(self, args, device):
        self.args, self.device = args, device
        self.tok = AutoTokenizer.from_pretrained(args.model)

    def _batches(self, texts, labels=None, shuffle=False, batch_size=32):
        order = np.arange(len(texts))
        if shuffle:
            np.random.shuffle(order)
        else:  # sort by length for faster inference; scores are put back in place
            order = np.argsort([len(t) for t in texts])
        for start in range(0, len(order), batch_size):
            idx = order[start:start + batch_size]
            enc = self.tok([texts[i] for i in idx], truncation=True, max_length=self.args.max_length,
                           padding=True, return_tensors="pt")
            yield idx, enc, (None if labels is None else torch.tensor(labels[idx]))

    def fit(self, texts: list[str], labels: np.ndarray) -> None:
        set_seed(SEED)
        a = self.args
        self.model = AutoModelForSequenceClassification.from_pretrained(a.model, num_labels=2).to(self.device)
        counts = np.bincount(labels, minlength=2)
        weight = torch.tensor(len(labels) / (2.0 * np.maximum(counts, 1)), dtype=torch.float, device=self.device)
        loss_fn = torch.nn.CrossEntropyLoss(weight=weight)
        steps = a.epochs * math.ceil(len(texts) / a.batch_size)
        opt = torch.optim.AdamW(self.model.parameters(), lr=a.lr, weight_decay=0.01)
        sched = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
        scaler = torch.amp.GradScaler("cuda", enabled=a.fp16)
        self.model.train()
        done = 0
        for _ in range(a.epochs):
            for _, enc, y in self._batches(texts, labels, shuffle=True, batch_size=a.batch_size):
                enc = {k: v.to(self.device) for k, v in enc.items()}
                with torch.autocast("cuda", dtype=torch.float16, enabled=a.fp16):
                    loss = loss_fn(self.model(**enc).logits.float(), y.to(self.device))
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                sched.step()
                done += 1
                if done % 200 == 0:
                    log(f"    step {done}/{steps} loss {loss.item():.4f}")

    @torch.no_grad()
    def score(self, texts: list[str]) -> np.ndarray:
        self.model.eval()
        out = np.zeros(len(texts), dtype=np.float32)
        for idx, enc, _ in self._batches(texts, batch_size=self.args.eval_batch_size):
            enc = {k: v.to(self.device) for k, v in enc.items()}
            with torch.autocast("cuda", dtype=torch.float16, enabled=self.args.fp16):
                logits = self.model(**enc).logits.float()
            out[idx] = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
        return out


def cached_fold(name: str, train: pd.DataFrame, test_texts: list[str], args, device) -> np.ndarray:
    path = OUT / "folds" / f"{name}.npy"
    if path.exists():
        log(f"  {name}: loaded from cache")
        return np.load(path)
    started = time.perf_counter()
    enc = Encoder(args, device)
    enc.fit(train["text"].tolist(), train["label"].to_numpy())
    scores = enc.score(test_texts)
    np.save(path, scores)
    log(f"  {name}: trained on {len(train):,}, scored {len(test_texts):,} in {time.perf_counter() - started:.0f}s")
    del enc
    torch.cuda.empty_cache()
    return scores


def metrics(y: np.ndarray, s: np.ndarray, t: float = 0.5) -> dict:
    pred = (s >= t).astype(int)
    labels = set(np.unique(y))
    if labels == {0, 1}:
        return {"kind": "mixed", "roc_auc": roc_auc_score(y, s), "average_precision": average_precision_score(y, s),
                "detection_rate": pred[y == 1].mean(), "false_positive_rate": pred[y == 0].mean()}
    if labels == {1}:
        return {"kind": "attack_only", "detection_rate": pred.mean()}
    return {"kind": "benign_only", "false_positive_rate": pred.mean()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="distilroberta-base")
    parser.add_argument("--train-cap", type=int, default=3000, help="max training texts per source and label")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--no-fp16", action="store_true")
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.fp16 = device == "cuda" and not args.no_fp16
    if device == "cpu":
        log("WARNING: no GPU found; this will be very slow")
    (OUT / "folds").mkdir(parents=True, exist_ok=True)

    corpus = pd.read_parquet(DATA / "corpus.parquet")
    train = corpus[corpus["role"] == "train_pool"].reset_index(drop=True)
    notinject = corpus[corpus["source"] == "notinject"].reset_index(drop=True)
    log(f"model {args.model}; pool {len(train):,}; cap {args.train_cap} per source and label; device {device}")

    # random protocol: same folds as script 03
    strata = train["source"] + "_" + train["label"].astype(str)
    rnd = np.zeros(len(train), dtype=np.float32)
    for k, (tr, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=SEED).split(train, strata), 1):
        log(f"random fold {k}/5")
        rnd[te] = cached_fold(f"random_{k}", cap(train.iloc[tr], args.train_cap),
                              train["text"].iloc[te].tolist(), args, device)

    # LODO protocol
    lodo_parts = []
    for source in sorted(train["source"].unique()):
        held = train["source"] == source
        log(f"LODO held out {source}")
        s = cached_fold(f"lodo_{source}", cap(train[~held], args.train_cap),
                        train.loc[held, "text"].tolist(), args, device)
        part = train.loc[held, ["source", "label"]].copy()
        part["score"] = s
        lodo_parts.append(part)

    log("NotInject (model trained on the capped full pool)")
    ni = cached_fold("full_pool_notinject", cap(train, args.train_cap), notinject["text"].tolist(), args, device)

    random_df = train[["source", "label"]].copy()
    random_df["score"] = rnd
    random_df["protocol"] = "random"
    lodo_df = pd.concat(lodo_parts)
    lodo_df["protocol"] = "lodo"
    scores = pd.concat([random_df, lodo_df], ignore_index=True)
    scores["detector"] = "finetuned"
    scores.to_parquet(OUT / "scores_finetuned.parquet", index=False)

    per_rows, pooled_rows = [], []
    for protocol, frame in (("random", random_df), ("lodo", lodo_df)):
        for source, g in frame.groupby("source"):
            per_rows.append({"detector": "finetuned", "protocol": protocol, "source": source, "n": len(g),
                             **metrics(g["label"].to_numpy(), g["score"].to_numpy())})
        y, s = frame["label"].to_numpy(), frame["score"].to_numpy()
        pooled_rows.append({"detector": "finetuned", "protocol": protocol, "pooled_roc_auc": roc_auc_score(y, s),
                            "pooled_average_precision": average_precision_score(y, s),
                            "pooled_detection_rate": (s[y == 1] >= 0.5).mean(),
                            "pooled_false_positive_rate": (s[y == 0] >= 0.5).mean()})
    pd.DataFrame(per_rows).to_csv(OUT / "per_source_metrics.csv", index=False)
    pd.DataFrame(pooled_rows).to_csv(OUT / "pooled_metrics.csv", index=False)
    pd.DataFrame([{"detector": "finetuned", "n": len(ni), "false_positive_rate": float((ni >= 0.5).mean())}]).to_csv(
        OUT / "notinject.csv", index=False)
    (OUT / "run_config.json").write_text(json.dumps({k: v for k, v in vars(args).items()} | {"seed": SEED},
                                                    indent=2), encoding="utf-8")
    log("pooled:\n" + pd.DataFrame(pooled_rows).round(4).to_string(index=False))
    log(f"NotInject FPR {float((ni >= 0.5).mean()):.3f}")


if __name__ == "__main__":
    main()
