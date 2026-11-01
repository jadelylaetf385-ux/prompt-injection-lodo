"""Download candidate prompt-injection datasets and write an inventory report.

For every candidate the script records whether it could be loaded, its splits,
row counts, columns, licence, the distribution of any label column, text-length
statistics and two short examples. Each loaded split is stored unchanged as
Parquet under data/raw/ so that later stages never depend on the network.

Usage
-----
    python scripts/01_inventory_datasets.py
    python scripts/01_inventory_datasets.py --only deepset,notinject
"""

from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
REPORT_DIR = ROOT / "reports"

# Candidate sources. "role" records how a source is expected to be used and is
# confirmed or revised after the inventory has been reviewed.
CANDIDATES = {
    "deepset": {"repo": "deepset/prompt-injections", "role": "mixed"},
    "safeguard": {"repo": "xTRam1/safe-guard-prompt-injection", "role": "mixed"},
    "spml": {"repo": "reshabhs/SPML_Chatbot_Prompt_Injection", "role": "mixed"},
    "jayavibhav": {"repo": "jayavibhav/prompt-injection", "role": "mixed"},
    "qualifire": {"repo": "qualifire/prompt-injections-benchmark", "role": "mixed"},
    "gandalf": {"repo": "Lakera/gandalf_ignore_instructions", "role": "malicious"},
    "mosscap": {"repo": "Lakera/mosscap_prompt_injection", "role": "malicious"},
    "hackaprompt": {"repo": "hackaprompt/hackaprompt-dataset", "role": "malicious"},
    "llmail": {"repo": "microsoft/llmail-inject-challenge", "role": "malicious_indirect"},
    "notinject": {"repo": "leolee99/NotInject", "role": "benign_trigger_words"},
    "alpaca": {"repo": "tatsu-lab/alpaca", "role": "benign"},
    "dolly": {"repo": "databricks/databricks-dolly-15k", "role": "benign"},
}

LABEL_HINTS = ("label", "labels", "is_injection", "injection", "prompt injection", "type", "class")
TEXT_HINTS = ("text", "prompt", "user_input", "instruction", "user prompt", "body", "input")


def _find_column(columns: list[str], hints: tuple[str, ...]) -> str | None:
    lowered = {column.lower(): column for column in columns}
    for hint in hints:
        if hint in lowered:
            return lowered[hint]
    for hint in hints:
        for low, original in lowered.items():
            if hint in low:
                return original
    return None


def _licence(repo: str) -> str | None:
    try:
        from huggingface_hub import DatasetCard

        card = DatasetCard.load(repo)
        return card.data.to_dict().get("license")
    except Exception:
        return None


def _summarise_split(frame: pd.DataFrame) -> dict:
    columns = list(frame.columns)
    text_column = _find_column(columns, TEXT_HINTS)
    label_column = _find_column(columns, LABEL_HINTS)

    summary: dict = {
        "rows": int(len(frame)),
        "columns": columns,
        "text_column_guess": text_column,
        "label_column_guess": label_column,
    }
    if label_column is not None:
        counts = frame[label_column].astype(str).value_counts().head(10)
        summary["label_distribution"] = {key: int(value) for key, value in counts.items()}
    if text_column is not None:
        lengths = frame[text_column].astype(str).str.len()
        summary["text_length"] = {
            "min": int(lengths.min()),
            "median": float(lengths.median()),
            "p95": float(lengths.quantile(0.95)),
            "max": int(lengths.max()),
        }
        summary["exact_duplicate_texts"] = int(frame[text_column].astype(str).duplicated().sum())
        summary["examples"] = [str(value)[:300] for value in frame[text_column].head(2).tolist()]
    return summary


def inventory_one(name: str, spec: dict) -> dict:
    from datasets import load_dataset

    record: dict = {"name": name, "repo": spec["repo"], "role": spec["role"]}
    started = time.perf_counter()
    try:
        dataset = load_dataset(spec["repo"])
    except Exception as error:
        record["status"] = "failed"
        record["error"] = f"{type(error).__name__}: {str(error)[:400]}"
        return record

    record["status"] = "loaded"
    record["licence"] = _licence(spec["repo"])
    record["splits"] = {}
    target = RAW_DIR / name
    target.mkdir(parents=True, exist_ok=True)
    for split_name, split in dataset.items():
        frame = split.to_pandas()
        frame.to_parquet(target / f"{split_name}.parquet", index=False)
        record["splits"][split_name] = _summarise_split(frame)
    record["seconds"] = round(time.perf_counter() - started, 1)
    return record


def write_markdown(records: list[dict], path: Path) -> None:
    lines = ["# Dataset inventory", ""]
    lines.append("| Source | Repository | Status | Licence | Split: rows |")
    lines.append("|---|---|---|---|---|")
    for record in records:
        splits = ", ".join(f"{k}: {v['rows']:,}" for k, v in record.get("splits", {}).items()) or "-"
        lines.append(
            f"| {record['name']} | {record['repo']} | {record['status']} | {record.get('licence') or '-'} | {splits} |"
        )
    lines.append("")
    for record in records:
        lines.append(f"## {record['name']} ({record['repo']})")
        if record["status"] != "loaded":
            lines.append(f"- Error: {record.get('error')}")
            lines.append("")
            continue
        for split_name, summary in record["splits"].items():
            lines.append(f"- **{split_name}**: {summary['rows']:,} rows; columns {summary['columns']}")
            lines.append(f"  - text column guess: {summary['text_column_guess']}; label column guess: {summary['label_column_guess']}")
            if "label_distribution" in summary:
                lines.append(f"  - labels: {summary['label_distribution']}")
            if "text_length" in summary:
                lines.append(f"  - text length: {summary['text_length']}; exact duplicates: {summary['exact_duplicate_texts']:,}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", default="", help="comma-separated subset of candidate names")
    args = parser.parse_args()

    selected = [item.strip() for item in args.only.split(",") if item.strip()] or list(CANDIDATES)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    records = []
    for name in selected:
        print(f"[{time.strftime('%H:%M:%S')}] {name}: loading {CANDIDATES[name]['repo']}", flush=True)
        try:
            record = inventory_one(name, CANDIDATES[name])
        except Exception:
            record = {"name": name, "repo": CANDIDATES[name]["repo"], "status": "failed",
                      "error": traceback.format_exc(limit=2)[-400:]}
        print(f"    -> {record['status']}" + (f" ({record.get('error', '')[:120]})" if record["status"] != "loaded" else ""), flush=True)
        records.append(record)

    (REPORT_DIR / "inventory.json").write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(records, REPORT_DIR / "inventory.md")
    loaded = sum(record["status"] == "loaded" for record in records)
    print(f"done: {loaded}/{len(records)} sources loaded; report at reports/inventory.md")


if __name__ == "__main__":
    main()
