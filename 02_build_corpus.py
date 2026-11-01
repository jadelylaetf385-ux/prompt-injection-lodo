"""Build the unified prompt-injection corpus from the raw downloads.

Every source is mapped to the same three fields (text, label, source), using
the column that carries the user-controlled text rather than any surrounding
template. The script then

1. removes empty texts and duplicates within each source,
2. measures exact overlap between sources on a normalised text key and writes
   the overlap matrix (a result in its own right: overlap makes
   leave-one-dataset-out evaluation leak),
3. drops texts that carry contradictory labels across sources,
4. keeps every remaining shared text in exactly one source (the smallest), so
   that holding a source out of training really removes its texts,
5. caps each source and label at a fixed size with a fixed seed.

Sources marked "eval_only" are kept apart and never used for training.

Usage
-----
    python scripts/02_build_corpus.py
    python scripts/02_build_corpus.py --cap 20000
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from itertools import combinations
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data"
REPORTS = ROOT / "reports"
SEED = 42
MIN_CHARS = 3


# --------------------------------------------------------------------------- #
# Source mapping
# --------------------------------------------------------------------------- #

def _read_all(name: str) -> pd.DataFrame:
    folder = RAW / name
    files = sorted(folder.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files under {folder}")
    return pd.concat([pd.read_parquet(path) for path in files], ignore_index=True)


def _frame(texts, labels, source: str) -> pd.DataFrame:
    return pd.DataFrame({"text": list(texts), "label": list(labels), "source": source})


def load_deepset():
    df = _read_all("deepset")
    return _frame(df["text"], df["label"].astype(int), "deepset")


def load_safeguard():
    df = _read_all("safeguard")
    return _frame(df["text"], df["label"].astype(int), "safeguard")


def load_spml():
    # Only the user turn is attacker-controlled; the system prompt is context.
    df = _read_all("spml")
    return _frame(df["User Prompt"], df["Prompt injection"].astype(int), "spml")


def load_jayavibhav():
    df = _read_all("jayavibhav")
    return _frame(df["text"], df["label"].astype(int), "jayavibhav")


def load_gandalf():
    df = _read_all("gandalf")
    return _frame(df["text"], [1] * len(df), "gandalf")


def load_mosscap():
    df = _read_all("mosscap")
    return _frame(df["prompt"], [1] * len(df), "mosscap")


def load_hackaprompt():
    # The "prompt" column wraps every input in the level template, a trivial
    # shortcut; only the participant input is used, and only inputs that
    # succeeded are kept so that every positive is a working injection.
    df = _read_all("hackaprompt")
    correct = df["correct"].astype(str).str.lower().isin({"true", "1"})
    df = df[correct]
    return _frame(df["user_input"], [1] * len(df), "hackaprompt")


def load_llmail():
    # Indirect injections: attacker-written e-mail bodies.
    df = _read_all("llmail")
    return _frame(df["body"], [1] * len(df), "llmail")


def load_alpaca():
    # The "text" column embeds a fixed instruction template; the instruction
    # (plus its optional input) is the user-facing text.
    df = _read_all("alpaca")
    extra = df["input"].fillna("").astype(str)
    text = df["instruction"].astype(str) + extra.map(lambda value: ("\n" + value) if value.strip() else "")
    return _frame(text, [0] * len(df), "alpaca")


def load_dolly():
    df = _read_all("dolly")
    return _frame(df["instruction"], [0] * len(df), "dolly")


def load_notinject():
    df = _read_all("notinject")
    return _frame(df["prompt"], [0] * len(df), "notinject")


SOURCES = {
    "deepset": (load_deepset, "train_pool"),
    "safeguard": (load_safeguard, "train_pool"),
    "spml": (load_spml, "train_pool"),
    "jayavibhav": (load_jayavibhav, "train_pool"),
    "gandalf": (load_gandalf, "train_pool"),
    "mosscap": (load_mosscap, "train_pool"),
    "hackaprompt": (load_hackaprompt, "train_pool"),
    "llmail": (load_llmail, "train_pool"),
    "alpaca": (load_alpaca, "train_pool"),
    "dolly": (load_dolly, "train_pool"),
    "notinject": (load_notinject, "eval_only"),
}


# --------------------------------------------------------------------------- #
# Cleaning helpers
# --------------------------------------------------------------------------- #

_WS = re.compile(r"\s+")


def normalise(text: str) -> str:
    return _WS.sub(" ", str(text)).strip()


def key(text: str) -> str:
    return hashlib.sha1(normalise(text).lower().encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cap", type=int, default=20000, help="maximum records per source and label")
    args = parser.parse_args()
    REPORTS.mkdir(parents=True, exist_ok=True)

    audit = []
    frames = []
    for name, (loader, role) in SOURCES.items():
        try:
            df = loader()
        except FileNotFoundError as error:
            print(f"skip {name}: {error}")
            continue
        raw_rows = len(df)
        df["text"] = df["text"].map(normalise)
        df = df[df["text"].str.len() >= MIN_CHARS]
        df["key"] = df["text"].map(key)
        before = len(df)
        # A text repeated inside one source with both labels is ambiguous.
        conflicted = df.groupby("key")["label"].nunique()
        conflicted = set(conflicted[conflicted > 1].index)
        df = df[~df["key"].isin(conflicted)]
        df = df.drop_duplicates("key")
        df["role"] = role
        audit.append({
            "source": name, "role": role, "raw_rows": raw_rows,
            "after_empty_filter": before, "within_source_label_conflicts": len(conflicted),
            "after_within_dedup": len(df),
            "positives": int((df["label"] == 1).sum()), "negatives": int((df["label"] == 0).sum()),
        })
        frames.append(df)
        print(f"{name:12s} raw {raw_rows:>9,} -> unique {len(df):>9,}")

    data = pd.concat(frames, ignore_index=True)
    train = data[data["role"] == "train_pool"]

    # Overlap matrix between training sources (counts of shared texts).
    names = sorted(train["source"].unique())
    keysets = {name: set(train.loc[train["source"] == name, "key"]) for name in names}
    matrix = pd.DataFrame(0, index=names, columns=names)
    for name in names:
        matrix.loc[name, name] = len(keysets[name])
    for a, b in combinations(names, 2):
        shared = len(keysets[a] & keysets[b])
        matrix.loc[a, b] = matrix.loc[b, a] = shared
    matrix.to_csv(REPORTS / "overlap_matrix.csv")
    share = matrix.copy().astype(float)
    for name in names:
        share.loc[name] = matrix.loc[name] / max(matrix.loc[name, name], 1)
    share.round(4).to_csv(REPORTS / "overlap_share_of_row_source.csv")

    # Texts that appear in several sources with different labels.
    label_sets = train.groupby("key")["label"].nunique()
    cross_conflicts = set(label_sets[label_sets > 1].index)
    conflict_rows = train[train["key"].isin(cross_conflicts)].sort_values("key")
    conflict_rows[["source", "label", "text"]].to_csv(REPORTS / "cross_source_label_conflicts.csv", index=False)
    train = train[~train["key"].isin(cross_conflicts)]

    # Keep each shared text in the smallest source only.
    size = train["source"].value_counts()
    train = train.assign(_size=train["source"].map(size)).sort_values(["key", "_size", "source"])
    resolved = train.drop_duplicates("key", keep="first").drop(columns="_size")
    removed = train.drop(columns="_size").merge(resolved[["key", "source"]], on=["key", "source"], how="left", indicator=True)
    removed = removed[removed["_merge"] == "left_only"]
    removed_by_source = removed["source"].value_counts().to_dict()

    # Cap per source and label.
    shuffled = resolved.sample(frac=1.0, random_state=SEED)
    rank = shuffled.groupby(["source", "label"]).cumcount()
    capped = shuffled[rank < args.cap].reset_index(drop=True)

    eval_only = data[data["role"] == "eval_only"]
    eval_only = eval_only[~eval_only["key"].isin(set(capped["key"]))]

    corpus = pd.concat([capped, eval_only], ignore_index=True)
    corpus["length"] = corpus["text"].str.len()
    corpus = corpus[["text", "label", "source", "role", "key", "length"]]
    corpus.to_parquet(OUT / "corpus.parquet", index=False)

    summary = corpus.groupby(["source", "role", "label"]).agg(records=("text", "size"),
                                                               median_length=("length", "median")).reset_index()
    summary.to_csv(REPORTS / "corpus_summary.csv", index=False)

    audit_df = pd.DataFrame(audit)
    audit_df["removed_as_cross_source_duplicate"] = audit_df["source"].map(removed_by_source).fillna(0).astype(int)
    audit_df["cross_source_label_conflict_texts"] = len(cross_conflicts)
    audit_df.to_csv(REPORTS / "cleaning_audit.csv", index=False)

    record = {
        "cap_per_source_and_label": args.cap,
        "seed": SEED,
        "training_pool_records": int((corpus["role"] == "train_pool").sum()),
        "eval_only_records": int((corpus["role"] == "eval_only").sum()),
        "cross_source_label_conflict_texts": len(cross_conflicts),
        "cross_source_duplicates_removed": removed_by_source,
    }
    (REPORTS / "corpus_build.json").write_text(json.dumps(record, indent=2), encoding="utf-8")

    print("\noverlap matrix (shared texts):")
    print(matrix.to_string())
    print("\ncorpus:")
    print(summary.to_string(index=False))
    print(f"\ncross-source label conflicts dropped: {len(cross_conflicts):,}")
    print(f"cross-source duplicates removed: {removed_by_source}")
    print("written: data/corpus.parquet and reports/{cleaning_audit,overlap_matrix,corpus_summary}.csv")


if __name__ == "__main__":
    main()
