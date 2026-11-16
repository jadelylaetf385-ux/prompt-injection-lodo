"""Manual label audit (revision, Reviewer 2 comment 2).

Step 1 (--make): draws a stratified sample of training-pool texts for two
independent annotators and writes
    results/label_audit/annotation_A.xlsx, annotation_B.xlsx  (no labels, no scores)
    results/label_audit/sample_key.csv                        (id, key, source, label, LODO scores)
The sample has, for every training source, up to --per-cell texts from each of
four cells: LODO false negatives and false positives of the character
detector, and correctly classified attacks and benign texts. Rows are shuffled
so annotators cannot infer the cell.

Step 2 (--score): after both annotators have filled the column "annotation"
(INJECTION, BENIGN or UNCLEAR) independently, writes
    agreement.csv     Cohen's kappa and raw agreement between annotators
    label_validity.csv
                      per source and cell: share of the dataset labels that
                      the adjudicated annotation confirms
    sensitivity.csv   LODO detection and false-positive rates of every
                      detector on the audited records, with dataset labels and
                      with adjudicated labels (UNCLEAR excluded)
Disagreements are adjudicated with the file adjudication.xlsx if present
(column "final"); otherwise records on which the annotators disagree are
excluded and their number is reported.

Usage
-----
    python scripts/17_label_audit.py --make --per-cell 8
    python scripts/17_label_audit.py --score
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RES = ROOT / "results"
OUT = RES / "label_audit"
SEED = 42
SCORES = {"surface": ("main/scores_surface.parquet", 0.5, ">="),
          "word_lr": ("main/scores_word_lr.parquet", 0.5, ">="),
          "char_svm": ("main/scores_char_svm.parquet", 0.0, ">"),
          "minilm_lr": ("minilm/scores_minilm_lr.parquet", 0.5, ">="),
          "finetuned": ("finetune/scores_finetuned.parquet", 0.5, ">=")}
GUIDE = [
    ("Task", "Decide for each text whether it is a prompt-injection attempt, judging the text alone."),
    ("INJECTION", "The text tries to make an LLM-based system ignore, override or leak its instructions, "
                  "reveal protected data (a password, system prompt, secret), perform an action the operator "
                  "did not intend, or produce a target output set by the attacker (for example 'I have been "
                  "PWNED'). Role-play, encoding or politeness used for that purpose still count."),
    ("BENIGN", "An ordinary request or message with no attempt to subvert instructions, even if it mentions "
               "security, passwords or the word 'ignore' in a harmless way."),
    ("UNCLEAR", "The text is too short, truncated, not in a language you can judge, or a reasonable reader "
                "could decide either way. Use sparingly and add a note."),
    ("Rules", "Work independently; do not discuss items until both files are complete. Do not look up the "
              "source dataset. Fill 'annotation' with exactly INJECTION, BENIGN or UNCLEAR; 'note' is optional."),
]


def load_scores(train: pd.DataFrame) -> pd.DataFrame:
    """LODO scores of every detector in corpus order (script 03 stores LODO
    scores grouped by sorted source, corpus order within a source)."""
    order = np.concatenate([np.where(train["source"] == s)[0] for s in sorted(train["source"].unique())])
    out = pd.DataFrame(index=train.index)
    for name, (rel, _, _) in SCORES.items():
        path = RES / rel
        if not path.exists():
            continue
        lodo = pd.read_parquet(path).query("protocol == 'lodo'")
        if not (lodo["label"].to_numpy() == train["label"].to_numpy()[order]).all():
            raise ValueError(f"{name}: LODO rows do not align with the corpus")
        col = np.empty(len(train))
        col[order] = lodo["score"].to_numpy()
        out[name] = col
    return out


def predict(name: str, s: np.ndarray) -> np.ndarray:
    _, t, op = SCORES[name]
    return (s > t) if op == ">" else (s >= t)


def make(per_cell: int) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    corpus = pd.read_parquet(DATA / "corpus.parquet")
    train = corpus[corpus["role"] == "train_pool"].reset_index(drop=True)
    scores = load_scores(train)
    pred = predict("char_svm", scores["char_svm"].to_numpy())
    y = train["label"].to_numpy()
    cell = np.select([(y == 1) & ~pred, (y == 0) & pred, (y == 1) & pred], ["false_negative", "false_positive",
                     "correct_attack"], "correct_benign")
    frame = pd.concat([train[["key", "source", "label", "text"]], scores], axis=1).assign(cell=cell)
    picked = (frame.sample(frac=1.0, random_state=SEED)
              .loc[lambda d: d.groupby(["source", "cell"]).cumcount() < per_cell]
              .sample(frac=1.0, random_state=SEED + 1).reset_index(drop=True))
    picked.insert(0, "id", [f"T{i + 1:04d}" for i in range(len(picked))])
    picked.drop(columns="text").to_csv(OUT / "sample_key.csv", index=False)
    sheet = picked[["id", "text"]].assign(annotation="", note="")
    for who in ("A", "B"):
        with pd.ExcelWriter(OUT / f"annotation_{who}.xlsx", engine="openpyxl") as xw:
            sheet.to_excel(xw, sheet_name="items", index=False)
            pd.DataFrame(GUIDE, columns=["item", "guideline"]).to_excel(xw, sheet_name="guidelines", index=False)
            ws = xw.sheets["items"]
            ws.column_dimensions["B"].width = 110
            ws.column_dimensions["C"].width = 14
            ws.column_dimensions["D"].width = 30
            from openpyxl.styles import Alignment
            from openpyxl.worksheet.datavalidation import DataValidation
            for row in ws.iter_rows(min_row=2, min_col=2, max_col=2):
                row[0].alignment = Alignment(wrap_text=True, vertical="top")
            dv = DataValidation(type="list", formula1='"INJECTION,BENIGN,UNCLEAR"', allow_blank=True)
            ws.add_data_validation(dv)
            dv.add(f"C2:C{len(sheet) + 1}")
            xw.sheets["guidelines"].column_dimensions["B"].width = 120
    print(f"{len(picked)} texts written to {OUT}")
    print(picked.groupby(["source", "cell"]).size().unstack(fill_value=0).to_string())


def kappa(a: pd.Series, b: pd.Series) -> float:
    cats = sorted(set(a) | set(b))
    po = float((a == b).mean())
    pe = sum(float((a == c).mean()) * float((b == c).mean()) for c in cats)
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def score() -> None:
    key = pd.read_csv(OUT / "sample_key.csv")
    ann = {}
    for who in ("A", "B"):
        d = pd.read_excel(OUT / f"annotation_{who}.xlsx", sheet_name="items")
        ann[who] = d.set_index("id")["annotation"].astype(str).str.strip().str.upper()
    both = pd.DataFrame(ann).dropna()
    both = both[both["A"].isin(["INJECTION", "BENIGN", "UNCLEAR"]) & both["B"].isin(["INJECTION", "BENIGN", "UNCLEAR"])]
    pd.DataFrame([{"items_annotated_by_both": len(both), "raw_agreement": float((both["A"] == both["B"]).mean()),
                   "cohen_kappa": kappa(both["A"], both["B"]),
                   "disagreements": int((both["A"] != both["B"]).sum())}]).to_csv(OUT / "agreement.csv", index=False)
    final = both["A"].where(both["A"] == both["B"])
    adj = OUT / "adjudication.xlsx"
    if adj.exists():
        a = pd.read_excel(adj).set_index("id")["final"].astype(str).str.strip().str.upper()
        final = final.fillna(a)
    k = key.set_index("id").join(final.rename("final"), how="inner").dropna(subset=["final"])
    k = k[k["final"] != "UNCLEAR"]
    k["human"] = (k["final"] == "INJECTION").astype(int)
    k["confirmed"] = (k["human"] == k["label"]).astype(int)
    k.groupby(["source", "cell"]).agg(records=("confirmed", "size"), label_confirmed=("confirmed", "mean")) \
        .reset_index().to_csv(OUT / "label_validity.csv", index=False)
    rows = []
    for name in SCORES:
        if name not in k:
            continue
        p = predict(name, k[name].to_numpy())
        for basis, y in (("dataset_label", k["label"].to_numpy()), ("human_label", k["human"].to_numpy())):
            rows.append({"detector": name, "labels": basis, "records": len(y),
                         "detection_rate": float(p[y == 1].mean()) if (y == 1).any() else np.nan,
                         "false_positive_rate": float(p[y == 0].mean()) if (y == 0).any() else np.nan})
    pd.DataFrame(rows).to_csv(OUT / "sensitivity.csv", index=False)
    print(pd.read_csv(OUT / "agreement.csv").to_string(index=False))
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    print("Note: the sample over-represents errors by design; rates on it are not corpus-level rates.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--make", action="store_true")
    parser.add_argument("--score", action="store_true")
    parser.add_argument("--per-cell", type=int, default=8)
    args = parser.parse_args()
    if args.make:
        make(args.per_cell)
    if args.score:
        score()


if __name__ == "__main__":
    main()
