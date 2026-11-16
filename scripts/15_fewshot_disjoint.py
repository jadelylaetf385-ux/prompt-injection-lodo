"""Few-shot adaptation with template- and cluster-disjoint splits
(revision experiment).

Script 06 drew adaptation examples and test texts at random from the same
source, so near-identical variants of one template could fall on both sides.
Here each held-out source is first grouped, and whole groups are assigned to
either the adaptation pool or the test part:

random     no grouping (as in script 06, but with repeated splits)
template   near-duplicate families: connected components of the graph that
           links texts whose character 3-5-gram TF-IDF cosine similarity is at
           least --sim (default 0.8)
semantic   k-means clusters (k = --clusters) of 128-dimensional LSA vectors of
           word TF-IDF

Groups are split separately within each label, so mixed sources keep both
labels on both sides. For every scheme, --splits different pool/test splits
are drawn, and for every split and k, --draws random draws of k adaptation
examples are taken. The word TF-IDF detector of script 06 is used; its fit is
deterministic, so variation comes from the data draws. The k = 0 model is the
same for all schemes and is fitted once.

Output (results/fewshot_disjoint/): runs.csv (every run), summary.csv (mean
utility and 2.5/97.5 percentiles across runs), overlap.csv (share of test texts
whose template family also occurs in the adaptation pool, per scheme).

Usage
-----
    python scripts/15_fewshot_disjoint.py
    python scripts/15_fewshot_disjoint.py --sources llmail,mosscap --k 0,10,50,200 --splits 5 --draws 2
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse.csgraph import connected_components
from scipy import sparse
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "results" / "fewshot_disjoint"
SEED = 42
POOL_MAX = 1000


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def fit_score(train_texts, train_labels, test_texts) -> np.ndarray:  # identical to script 06
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=200000, sublinear_tf=True)
    model = LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear")
    model.fit(vec.fit_transform(train_texts), train_labels)
    return model.predict_proba(vec.transform(test_texts))[:, 1]


def utility(y: np.ndarray, s: np.ndarray) -> float:
    pred = s >= 0.5
    if len(set(y)) == 2:
        return float(0.5 * (pred[y == 1].mean() + 1 - pred[y == 0].mean()))
    return float(pred.mean()) if y[0] == 1 else float(1 - pred.mean())


def template_groups(texts: pd.Series, sim: float, chunk: int = 500) -> np.ndarray:
    X = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True).fit_transform(texts)
    rows, cols = [], []
    for start in range(0, X.shape[0], chunk):
        S = (X[start:start + chunk] @ X.T).tocoo()
        keep = S.data >= sim
        rows.append(S.row[keep] + start)
        cols.append(S.col[keep])
    A = sparse.coo_matrix((np.ones(sum(map(len, rows))), (np.concatenate(rows), np.concatenate(cols))),
                          shape=(X.shape[0], X.shape[0]))
    return connected_components(A, directed=False)[1]


def semantic_groups(texts: pd.Series, k: int) -> np.ndarray:
    X = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=100000, sublinear_tf=True).fit_transform(texts)
    Z = TruncatedSVD(128, random_state=SEED).fit_transform(X)
    Z /= np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9
    return KMeans(k, n_init=4, random_state=SEED).fit_predict(Z)


def split(held: pd.DataFrame, groups: np.ndarray | None, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    pools, tests = [], []
    for label, g in held.groupby("label"):
        target = min(POOL_MAX, len(g) // 2)
        if groups is None:
            idx = rng.permutation(len(g))
            pools.append(g.iloc[idx[:target]])
            tests.append(g.iloc[idx[target:]])
            continue
        gid = groups[held.index.get_indexer(g.index)]
        order = rng.permutation(np.unique(gid))
        pool_ids, n = [], 0
        for u in order:  # whole groups go to the pool until it holds about half the target
            if n >= target:
                break
            pool_ids.append(u)
            n += (gid == u).sum()
        in_pool = np.isin(gid, pool_ids)
        pool = g[in_pool]
        pools.append(pool.sample(n=min(target, len(pool)), random_state=seed))
        tests.append(g[~in_pool])
    return pd.concat(pools), pd.concat(tests)


def draw(pool: pd.DataFrame, k: int, seed: int) -> pd.DataFrame | None:
    labels = sorted(pool["label"].unique())
    per = [k // len(labels) + (1 if i < k % len(labels) else 0) for i in range(len(labels))]
    parts = []
    for label, n in zip(labels, per):
        g = pool[pool["label"] == label]
        if n > len(g):
            return None
        parts.append(g.sample(n=n, random_state=seed))
    return pd.concat(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sources", default="llmail,mosscap,hackaprompt,jayavibhav")
    parser.add_argument("--k", default="0,10,50,200")
    parser.add_argument("--splits", type=int, default=5)
    parser.add_argument("--draws", type=int, default=2)
    parser.add_argument("--sim", type=float, default=0.8)
    parser.add_argument("--clusters", type=int, default=50)
    parser.add_argument("--out", default=str(OUT), help="output directory (to run sources in parallel)")
    args = parser.parse_args()
    out = Path(args.out)
    ks = [int(v) for v in args.k.split(",")]
    out.mkdir(parents=True, exist_ok=True)
    corpus = pd.read_parquet(DATA / "corpus.parquet")
    train = corpus[corpus["role"] == "train_pool"].reset_index(drop=True)
    runs_path = out / "runs.csv"
    runs = pd.read_csv(runs_path).to_dict("records") if runs_path.exists() else []
    done = {(r["source"], r["scheme"], r["split"], r["k"], r["draw"]) for r in runs}
    overlap = []

    for source in [s.strip() for s in args.sources.split(",") if s.strip()]:
        held = train[train["source"] == source]
        rest = train[train["source"] != source]
        log(f"{source}: grouping {len(held):,} texts")
        groups = {"random": None, "template": template_groups(held["text"], args.sim),
                  "semantic": semantic_groups(held["text"], args.clusters)}
        log(f"  template families: {len(np.unique(groups['template'])):,}; semantic clusters: {args.clusters}")
        s0 = pd.Series(fit_score(rest["text"], rest["label"].to_numpy(), held["text"]), index=held.index)
        for scheme, gid in groups.items():
            for sp in range(args.splits):
                pool, test = split(held, gid, SEED + sp)
                tfam = groups["template"][held.index.get_indexer(test.index)]
                pfam = set(groups["template"][held.index.get_indexer(pool.index)])
                overlap.append({"source": source, "scheme": scheme, "split": sp,
                                "test_share_with_family_in_pool": float(np.isin(tfam, list(pfam)).mean()),
                                "n_pool": len(pool), "n_test": len(test)})
                y = test["label"].to_numpy()
                for k in ks:
                    for d in range(1 if k == 0 else args.draws):
                        if (source, scheme, sp, k, d) in done:
                            continue
                        if k == 0:
                            s = s0.loc[test.index].to_numpy()
                        else:
                            add = draw(pool, k, SEED + 100 * sp + d)
                            if add is None:
                                continue
                            s = fit_score(pd.concat([rest["text"], add["text"]]),
                                          np.concatenate([rest["label"].to_numpy(), add["label"].to_numpy()]),
                                          test["text"])
                        runs.append({"source": source, "scheme": scheme, "split": sp, "k": k, "draw": d,
                                     "n_test": len(test), "utility": utility(y, s)})
                        pd.DataFrame(runs).to_csv(runs_path, index=False)
                log(f"  {source} {scheme} split {sp + 1}/{args.splits} done")
        pd.DataFrame(overlap).to_csv(out / "overlap.csv", index=False)

    r = pd.DataFrame(runs)
    summary = r.groupby(["source", "scheme", "k"])["utility"].agg(
        mean="mean", low=lambda x: np.percentile(x, 2.5), high=lambda x: np.percentile(x, 97.5), runs="count").reset_index()
    summary.to_csv(out / "summary.csv", index=False, float_format="%.6f")
    log("mean utility:\n" + summary.pivot_table(index=["source", "scheme"], columns="k", values="mean").round(3).to_string())
    log("test texts sharing a template family with the pool:\n"
        + pd.DataFrame(overlap).groupby(["source", "scheme"])["test_share_with_family_in_pool"].mean().round(3).to_string())


if __name__ == "__main__":
    main()
