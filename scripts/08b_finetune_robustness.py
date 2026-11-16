"""Robustness of the fine-tuned encoder: random seeds, training duration and
training-only validation curves (revision experiment).

Same protocol, folds, training cap and hyperparameters as script 08, with three
additions requested in review:

* --seed sets model initialisation and data order (the capped training subset
  is drawn with the fixed data seed 42, as in script 08, so seeds isolate
  optimisation randomness).
* --epochs E trains for E epochs with a linear schedule over all E epochs and
  scores the held-out texts after every epoch, so one run yields results for
  epochs 1..E.
* 10% of each fold's capped training texts (stratified by source and label)
  are held out as a validation set drawn only from training sources; loss and
  ROC-AUC on it are logged every --eval-every steps and at each epoch end.

Outputs go to results/finetune_runs/<run-name>/: per-epoch scores in the format
of script 03 (scores_ep<e>.parquet), per-source and pooled metrics per epoch,
NotInject FPR per epoch, and training/validation curves (curves.csv). Every
fold and epoch is cached, so an interrupted Colab session resumes.

Usage (Colab, GPU)
------------------
    python scripts/08b_finetune_robustness.py --seed 43 --epochs 1 --run-name seed43_ep1
    python scripts/08b_finetune_robustness.py --seed 44 --epochs 1 --run-name seed44_ep1
    python scripts/08b_finetune_robustness.py --seed 42 --epochs 3 --run-name seed42_ep3
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
from sklearn.model_selection import StratifiedKFold, train_test_split
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
DATA_SEED = 42  # folds and capped subsets, identical to scripts 03 and 08


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def cap(frame: pd.DataFrame, per_cell: int) -> pd.DataFrame:
    shuffled = frame.sample(frac=1.0, random_state=DATA_SEED)
    return shuffled[shuffled.groupby(["source", "label"]).cumcount() < per_cell]


class Run:
    """Trains one fold and scores the held-out texts after every epoch."""

    def __init__(self, args, device):
        self.args, self.device = args, device
        self.tok = AutoTokenizer.from_pretrained(args.model)

    def _batches(self, texts, labels=None, shuffle=False, batch_size=32):
        order = np.random.permutation(len(texts)) if shuffle else np.argsort([len(t) for t in texts])
        for start in range(0, len(order), batch_size):
            idx = order[start:start + batch_size]
            enc = self.tok([texts[i] for i in idx], truncation=True, max_length=self.args.max_length,
                           padding=True, return_tensors="pt")
            yield idx, enc, (None if labels is None else torch.tensor(labels[idx]))

    @torch.no_grad()
    def score(self, texts: list[str]) -> np.ndarray:
        self.model.eval()
        out = np.zeros(len(texts), dtype=np.float32)
        for idx, enc, _ in self._batches(texts, batch_size=self.args.eval_batch_size):
            enc = {k: v.to(self.device) for k, v in enc.items()}
            with torch.autocast("cuda", dtype=torch.float16, enabled=self.args.fp16):
                logits = self.model(**enc).logits.float()
            out[idx] = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
        self.model.train()
        return out

    def validate(self, texts, labels, loss_fn) -> tuple[float, float]:
        s = self.score(texts)
        p = np.clip(s, 1e-6, 1 - 1e-6)
        w = loss_fn.weight.detach().cpu().numpy()
        loss = -np.mean(w[labels] * np.where(labels == 1, np.log(p), np.log(1 - p)))
        auc = roc_auc_score(labels, s) if len(set(labels)) == 2 else float("nan")
        return float(loss), float(auc)

    def fit_and_score(self, fold: str, train: pd.DataFrame, test_sets: dict, cache_dir: Path, curves: list):
        a = self.args
        tr, va = train_test_split(train, test_size=0.10, random_state=DATA_SEED,
                                  stratify=train["source"] + "_" + train["label"].astype(str))
        texts, labels = tr["text"].tolist(), tr["label"].to_numpy()
        vtexts, vlabels = va["text"].tolist(), va["label"].to_numpy()
        set_seed(a.seed)
        self.model = AutoModelForSequenceClassification.from_pretrained(a.model, num_labels=2).to(self.device)
        counts = np.bincount(labels, minlength=2)
        weight = torch.tensor(len(labels) / (2.0 * np.maximum(counts, 1)), dtype=torch.float, device=self.device)
        loss_fn = torch.nn.CrossEntropyLoss(weight=weight)
        steps = a.epochs * math.ceil(len(texts) / a.batch_size)
        opt = torch.optim.AdamW(self.model.parameters(), lr=a.lr, weight_decay=0.01)
        sched = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
        scaler = torch.amp.GradScaler("cuda", enabled=a.fp16)
        self.model.train()
        done, running = 0, []
        for epoch in range(1, a.epochs + 1):
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
                running.append(loss.item())
                if done % a.eval_every == 0:
                    vl, vauc = self.validate(vtexts, vlabels, loss_fn)
                    curves.append({"fold": fold, "step": done, "epoch": done / math.ceil(len(texts) / a.batch_size),
                                   "train_loss": float(np.mean(running)), "val_loss": vl, "val_auc": vauc})
                    running = []
                    log(f"    {fold} step {done}/{steps} val loss {vl:.4f} val AUC {vauc:.4f}")
            vl, vauc = self.validate(vtexts, vlabels, loss_fn)
            curves.append({"fold": fold, "step": done, "epoch": float(epoch),
                           "train_loss": float(np.mean(running)) if running else float("nan"),
                           "val_loss": vl, "val_auc": vauc})
            for name, test_texts in test_sets.items():
                np.save(cache_dir / f"{name}_ep{epoch}.npy", self.score(test_texts))
            log(f"  {fold}: epoch {epoch} scored")
        del self.model
        torch.cuda.empty_cache()


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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--train-cap", type=int, default=3000)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--eval-every", type=int, default=300)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--no-fp16", action="store_true")
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.fp16 = device == "cuda" and not args.no_fp16
    out = ROOT / "results" / "finetune_runs" / args.run_name
    cache = out / "folds"
    cache.mkdir(parents=True, exist_ok=True)
    curves_path = out / "curves.csv"
    curves = pd.read_csv(curves_path).to_dict("records") if curves_path.exists() else []

    corpus = pd.read_parquet(DATA / "corpus.parquet")
    train = corpus[corpus["role"] == "train_pool"].reset_index(drop=True)
    notinject = corpus[corpus["source"] == "notinject"].reset_index(drop=True)
    log(f"run {args.run_name}: seed {args.seed}, epochs {args.epochs}, device {device}")

    def done(name):  # every epoch of this fold already cached
        return all((cache / f"{name}_ep{e}.npy").exists() for e in range(1, args.epochs + 1))

    runner = Run(args, device)
    strata = train["source"] + "_" + train["label"].astype(str)
    folds = list(StratifiedKFold(5, shuffle=True, random_state=DATA_SEED).split(train, strata))
    for k, (tr, te) in enumerate(folds, 1):
        name = f"random_{k}"
        if not done(name):
            log(f"random fold {k}/5")
            runner.fit_and_score(name, cap(train.iloc[tr], args.train_cap),
                                 {name: train["text"].iloc[te].tolist()}, cache, curves)
            pd.DataFrame(curves).to_csv(curves_path, index=False)
    sources = sorted(train["source"].unique())
    for source in sources:
        name = f"lodo_{source}"
        if not done(name):
            log(f"LODO held out {source}")
            held = train["source"] == source
            runner.fit_and_score(name, cap(train[~held], args.train_cap),
                                 {name: train.loc[held, "text"].tolist()}, cache, curves)
            pd.DataFrame(curves).to_csv(curves_path, index=False)
    if not done("full_pool_notinject"):
        log("NotInject (model trained on the capped full pool)")
        runner.fit_and_score("full_pool", cap(train, args.train_cap),
                             {"full_pool_notinject": notinject["text"].tolist()}, cache, curves)
        pd.DataFrame(curves).to_csv(curves_path, index=False)

    pooled_rows, per_rows, ni_rows = [], [], []
    for e in range(1, args.epochs + 1):
        rnd = np.zeros(len(train), dtype=np.float32)
        for k, (_, te) in enumerate(folds, 1):
            rnd[te] = np.load(cache / f"random_{k}_ep{e}.npy")
        random_df = train[["source", "label"]].assign(score=rnd, protocol="random")
        lodo_df = pd.concat([train.loc[train["source"] == s, ["source", "label"]]
                             .assign(score=np.load(cache / f"lodo_{s}_ep{e}.npy")) for s in sources])
        lodo_df["protocol"] = "lodo"
        scores = pd.concat([random_df, lodo_df], ignore_index=True).assign(detector="finetuned")
        scores.to_parquet(out / f"scores_ep{e}.parquet", index=False)
        for protocol, frame in (("random", random_df), ("lodo", lodo_df)):
            y, s = frame["label"].to_numpy(), frame["score"].to_numpy()
            pooled_rows.append({"run": args.run_name, "seed": args.seed, "epoch": e, "protocol": protocol,
                                "pooled_roc_auc": roc_auc_score(y, s),
                                "pooled_average_precision": average_precision_score(y, s),
                                "pooled_detection_rate": (s[y == 1] >= 0.5).mean(),
                                "pooled_false_positive_rate": (s[y == 0] >= 0.5).mean()})
            for source, g in frame.groupby("source"):
                per_rows.append({"run": args.run_name, "seed": args.seed, "epoch": e, "protocol": protocol,
                                 "source": source, "n": len(g),
                                 **metrics(g["label"].to_numpy(), g["score"].to_numpy())})
        ni = np.load(cache / f"full_pool_notinject_ep{e}.npy")
        ni_rows.append({"run": args.run_name, "seed": args.seed, "epoch": e, "n": len(ni),
                        "false_positive_rate": float((ni >= 0.5).mean())})
    pd.DataFrame(pooled_rows).to_csv(out / "pooled_metrics.csv", index=False)
    pd.DataFrame(per_rows).to_csv(out / "per_source_metrics.csv", index=False)
    pd.DataFrame(ni_rows).to_csv(out / "notinject.csv", index=False)
    (out / "run_config.json").write_text(json.dumps(vars(args) | {"data_seed": DATA_SEED,
                                                                  "validation_fraction": 0.10}, indent=2),
                                         encoding="utf-8")
    log("pooled:\n" + pd.DataFrame(pooled_rows).round(4).to_string(index=False))
    log("NotInject:\n" + pd.DataFrame(ni_rows).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
