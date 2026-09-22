"""Differential regulatory rewiring between two matched subtype networks."""

from __future__ import annotations

from itertools import combinations
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd

from .network import hub_degree, link_list, matched_ensemble, proportion_normalize
from .stats_utils import bh_fdr

Edge = Tuple[str, str]
MIN_EDGES_PER_BIN = 500


def edge_pairs(edges: pd.DataFrame) -> Set[Edge]:
    return set(zip(edges["regulator"], edges["target"]))


def jaccard(a: Set[Edge], b: Set[Edge]) -> float:
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def compare_edge_sets(edges_a: pd.DataFrame, edges_b: pd.DataFrame,
                      name_a: str, name_b: str) -> Dict[str, object]:
    a, b = edge_pairs(edges_a), edge_pairs(edges_b)
    return {
        "group_a": name_a,
        "group_b": name_b,
        "n_a": len(a),
        "n_b": len(b),
        "shared": len(a & b),
        f"{name_a}_specific": len(a - b),
        f"{name_b}_specific": len(b - a),
        "jaccard": jaccard(a, b),
    }


def delta_matrix(vim_a: pd.DataFrame, vim_b: pd.DataFrame) -> pd.DataFrame:
    """Signed difference of per-target normalized importances."""
    ra, rb = proportion_normalize(vim_a), proportion_normalize(vim_b)
    rb = rb.reindex(index=ra.index, columns=ra.columns)
    return ra - rb


def _flat(frame: pd.DataFrame) -> pd.Series:
    stacked = frame.stack()
    stacked.index.names = ["regulator", "target"]
    return stacked[stacked.index.get_level_values(0) != stacked.index.get_level_values(1)]


def _flat_abs(vim_a: pd.DataFrame, vim_b: pd.DataFrame,
              index: pd.MultiIndex | None = None) -> pd.Series:
    delta = _flat(delta_matrix(vim_a, vim_b)).abs()
    return delta if index is None else delta.reindex(index)


def permutation_null(groups: Dict[str, pd.DataFrame], n_match: int, n_reps: int,
                     n_perm: int, seed: int = 0, verbose: bool = False,
                     **genie3_kwargs) -> pd.DataFrame:
    """Redo the whole comparison on shuffled subtype labels; costs n_perm * 2 * n_reps fits."""
    pooled = pd.concat(list(groups.values()), axis=1)
    rng = np.random.default_rng(seed)
    draws = []
    index = None
    for k in range(n_perm):
        shuffled = rng.permutation(np.asarray(pooled.columns))
        half = len(shuffled) // 2
        left = matched_ensemble(pooled.loc[:, list(shuffled[:half])], n_match, n_reps,
                                seed=seed + 31 * k, **genie3_kwargs)
        right = matched_ensemble(pooled.loc[:, list(shuffled[half:])], n_match, n_reps,
                                 seed=seed + 31 * k + 7, **genie3_kwargs)
        delta = _flat_abs(left, right, index)
        index = delta.index if index is None else index
        draws.append(delta.astype(np.float32))
        if verbose:
            print(f"  permutation {k + 1}/{n_perm}", flush=True)
    return pd.DataFrame(draws).T


def _binned_pvalues(observed: np.ndarray, base: np.ndarray,
                    null: np.ndarray, n_bins: int) -> Tuple[np.ndarray, np.ndarray]:
    """Compare each edge with the pooled null of edges at a similar importance level."""
    n_bins = max(1, min(n_bins, observed.size // MIN_EDGES_PER_BIN))
    cuts = np.quantile(base, np.linspace(0, 1, n_bins + 1))
    cuts[0], cuts[-1] = -np.inf, np.inf
    bin_id = np.clip(np.searchsorted(cuts, base, side="right") - 1, 0, n_bins - 1)

    p = np.ones(observed.size)
    null_mean = np.zeros(observed.size)
    for b in range(n_bins):
        members = np.flatnonzero(bin_id == b)
        if members.size == 0:
            continue
        pool = np.sort(null[members].ravel())
        above = pool.size - np.searchsorted(pool, observed[members], side="left")
        p[members] = (above + 1.0) / (pool.size + 1.0)
        null_mean[members] = float(pool.mean())
    return p, null_mean


def differential_edges(vim_a: pd.DataFrame, vim_b: pd.DataFrame, null: pd.DataFrame,
                       name_a: str = "A", name_b: str = "B", n_bins: int = 20,
                       seed: int = 0) -> pd.DataFrame:
    """Rank the edges by how far the two subtypes sit apart relative to the permutation null."""
    ra, rb = proportion_normalize(vim_a), proportion_normalize(vim_b)
    score_a = _flat(ra)
    score_b = _flat(rb.reindex(index=ra.index, columns=ra.columns))
    observed = score_a - score_b
    null = null.reindex(observed.index)
    if null.isna().to_numpy().any():
        raise ValueError("the null does not cover every edge of this comparison")

    base = ((score_a + score_b) / 2.0).to_numpy()
    p, null_mean = _binned_pvalues(observed.abs().to_numpy(), base,
                                   null.to_numpy(dtype=float), n_bins)

    out = pd.DataFrame({
        "regulator": observed.index.get_level_values(0),
        "target": observed.index.get_level_values(1),
        f"score_{name_a}": score_a.to_numpy(),
        f"score_{name_b}": score_b.to_numpy(),
        "delta": observed.to_numpy(),
        "abs_delta": observed.abs().to_numpy(),
        "null_mean_abs_delta": null_mean,
        "p_value": p,
    })
    out["q_value"] = bh_fdr(out["p_value"].to_numpy())
    out["favours"] = np.where(out["delta"] > 0, name_a, name_b)
    out = out.sort_values(["p_value", "abs_delta"], ascending=[True, False])
    return out.reset_index(drop=True)


def significant_edges(diff: pd.DataFrame, fdr: float = 0.05) -> pd.DataFrame:
    return diff[diff["q_value"] <= fdr].reset_index(drop=True)


def gene_rewiring(diff: pd.DataFrame) -> pd.DataFrame:
    """Per-gene rewiring: how much of a gene's regulatory neighbourhood moved."""
    reg = diff.groupby("regulator")["delta"].agg(["mean", lambda s: s.abs().mean(), "size"])
    reg.columns = ["mean_delta_out", "mean_abs_delta_out", "n_edges_out"]
    tgt = diff.groupby("target")["delta"].agg([lambda s: s.abs().mean(), "size"])
    tgt.columns = ["mean_abs_delta_in", "n_edges_in"]
    tab = reg.join(tgt, how="outer").fillna(0.0)
    tab["rewiring_score"] = tab[["mean_abs_delta_out", "mean_abs_delta_in"]].max(axis=1)
    tab.index.name = "gene"
    return tab.sort_values("rewiring_score", ascending=False)


def degree_shift(edges_a: pd.DataFrame, edges_b: pd.DataFrame,
                 name_a: str, name_b: str) -> pd.DataFrame:
    da = hub_degree(edges_a)["total_degree"].rename(name_a)
    db = hub_degree(edges_b)["total_degree"].rename(name_b)
    tab = pd.concat([da, db], axis=1).fillna(0.0)
    tab["shift"] = tab[name_a] - tab[name_b]
    return tab.sort_values("shift", key=lambda s: -s.abs())


def rewiring_summary(networks: Dict[str, pd.DataFrame], top_edges: int) -> pd.DataFrame:
    """Pairwise top-edge overlap across every subtype network."""
    tops = {name: link_list(v, top_n=top_edges) for name, v in networks.items()}
    rows: List[Dict[str, object]] = []
    for a, b in combinations(sorted(tops), 2):
        rows.append(compare_edge_sets(tops[a], tops[b], a, b))
    return pd.DataFrame(rows)
