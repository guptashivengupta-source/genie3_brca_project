"""GENIE3 tree-ensemble network inference and sample-size-matched ensembles."""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor

TREE_METHODS = {"RF": RandomForestRegressor, "ET": ExtraTreesRegressor}


def _target_importances(X: np.ndarray, t: int, reg_cols: np.ndarray, tree_method: str,
                        n_trees: int, max_features, random_state: int) -> np.ndarray:
    y = X[:, t]
    var = float(y.var())
    mask = reg_cols != t
    out = np.zeros(reg_cols.size)
    if var <= 0 or not mask.any():
        return out
    model = TREE_METHODS[tree_method](n_estimators=n_trees, max_features=max_features,
                                      n_jobs=1, random_state=random_state)
    model.fit(X[:, reg_cols[mask]], y)
    # sklearn importances sum to 1; scaling by var(y) restores the variance reduction
    out[mask] = model.feature_importances_ * var
    return out


def genie3(expr: pd.DataFrame, regulators: Sequence[str] | None = None,
           tree_method: str = "RF", n_trees: int = 200,
           max_features: str | int | float = "sqrt", n_jobs: int = 1,
           random_state: int = 0, verbose: bool = False) -> pd.DataFrame:
    """Importance of every regulator for every target; rows regulators, cols targets."""
    if tree_method not in TREE_METHODS:
        raise ValueError(f"tree_method must be one of {sorted(TREE_METHODS)}")
    genes = list(expr.index)
    pos = {g: i for i, g in enumerate(genes)}
    regs = [g for g in (regulators if regulators is not None else genes) if g in pos]
    if not regs:
        raise ValueError("no regulator is present in the expression matrix")

    X = np.ascontiguousarray(expr.to_numpy(dtype=np.float64).T)
    reg_cols = np.array([pos[g] for g in regs])
    jobs = (delayed(_target_importances)(X, t, reg_cols, tree_method, n_trees,
                                         max_features, random_state + t)
            for t in range(len(genes)))
    columns = Parallel(n_jobs=n_jobs, verbose=5 if verbose else 0, batch_size=8)(jobs)
    return pd.DataFrame(np.column_stack(columns), index=regs, columns=genes)


def rank_normalize(vim: pd.DataFrame) -> pd.DataFrame:
    """Per-target ranks in [0, 1] so importances are comparable between networks."""
    ranked = vim.rank(axis=0, method="average", na_option="bottom")
    denom = max(len(vim.index) - 1, 1)
    return (ranked - 1.0) / denom


def proportion_normalize(vim: pd.DataFrame) -> pd.DataFrame:
    """Share of a target's explained variance per regulator; columns sum to one."""
    totals = vim.sum(axis=0).replace(0.0, np.nan)
    return vim.div(totals, axis=1).fillna(0.0)


def link_list(vim: pd.DataFrame, top_n: int | None = None,
              min_importance: float = 0.0) -> pd.DataFrame:
    stacked = vim.stack()
    stacked.index.names = ["regulator", "target"]
    edges = stacked.rename("importance").reset_index()
    edges = edges[(edges["regulator"] != edges["target"]) &
                  (edges["importance"] > min_importance)]
    edges = edges.sort_values("importance", ascending=False, kind="mergesort")
    if top_n is not None:
        edges = edges.head(top_n)
    return edges.reset_index(drop=True)


def annotate_edges(edges: pd.DataFrame, tfs: Sequence[str], group: str) -> pd.DataFrame:
    tfset = set(tfs)
    out = edges.copy()
    out["regulator_is_tf"] = out["regulator"].isin(tfset)
    out["group"] = group
    return out


def symmetric_adjacency(vim: pd.DataFrame) -> pd.DataFrame:
    """Undirected weights for module detection: the stronger of the two directions."""
    genes = sorted(set(vim.index) | set(vim.columns))
    full = vim.reindex(index=genes, columns=genes).fillna(0.0)
    adj = np.maximum(full.to_numpy(), full.to_numpy().T)
    np.fill_diagonal(adj, 0.0)
    return pd.DataFrame(adj, index=genes, columns=genes)


def subsample(expr: pd.DataFrame, n: int, rng: np.random.Generator,
              replace: bool = False) -> pd.DataFrame:
    cols = rng.choice(np.asarray(expr.columns), size=n, replace=replace)
    return expr.loc[:, list(cols)]


def matched_ensemble(expr: pd.DataFrame, n_match: int, n_reps: int, seed: int = 0,
                     return_reps: bool = False, **genie3_kwargs):
    """Average GENIE3 over equal-sized subsamples so group size cannot drive edge weight."""
    if n_match > expr.shape[1]:
        raise ValueError(f"n_match={n_match} exceeds the {expr.shape[1]} available samples")
    rng = np.random.default_rng(seed)
    reps: List[pd.DataFrame] = []
    for rep in range(n_reps):
        sub = subsample(expr, n_match, rng)
        reps.append(genie3(sub, random_state=seed + 1000 * rep, **genie3_kwargs))
    mean = sum(reps) / float(n_reps)
    return (mean, reps) if return_reps else mean


def matched_networks(groups: Dict[str, pd.DataFrame], n_match: int, n_reps: int,
                     seed: int = 0, verbose: bool = False, return_reps: bool = False,
                     **genie3_kwargs):
    means: Dict[str, pd.DataFrame] = {}
    reps: Dict[str, List[pd.DataFrame]] = {}
    for i, (name, sub) in enumerate(sorted(groups.items())):
        if verbose:
            print(f"[network] {name}: {sub.shape[1]} samples -> {n_reps} x n={n_match}",
                  flush=True)
        res = matched_ensemble(sub, n_match, n_reps, seed=seed + 97 * i,
                               return_reps=return_reps, **genie3_kwargs)
        if return_reps:
            means[name], reps[name] = res
        else:
            means[name] = res
    return (means, reps) if return_reps else means


def hub_degree(edges: pd.DataFrame) -> pd.DataFrame:
    out_deg = edges.groupby("regulator").size()
    in_deg = edges.groupby("target").size()
    tab = pd.DataFrame({"out_degree": out_deg, "in_degree": in_deg}).fillna(0.0).astype(float)
    tab["total_degree"] = tab["out_degree"] + tab["in_degree"]
    return tab.sort_values("total_degree", ascending=False)


def top_hubs(edges: pd.DataFrame, k: int) -> List[str]:
    return list(hub_degree(edges).head(k).index)
