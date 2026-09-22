"""Module detection on the inferred network and platform-free module activity scores."""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from .preprocessing import rank_normalize_samples, zscore_rows
from .signatures import load_tfs


def scale_adjacency(adj: pd.DataFrame) -> pd.DataFrame:
    """Map weights onto [0, 1] so the topological overlap is well defined."""
    values = adj.to_numpy(dtype=float)
    top = values.max()
    if top <= 0:
        return pd.DataFrame(np.zeros_like(values), index=adj.index, columns=adj.columns)
    scaled = np.clip(values / top, 0.0, 1.0)
    np.fill_diagonal(scaled, 0.0)
    return pd.DataFrame(scaled, index=adj.index, columns=adj.columns)


def topological_overlap(adj: pd.DataFrame) -> pd.DataFrame:
    """WGCNA topological overlap: two genes are close if they share neighbours."""
    a = scale_adjacency(adj).to_numpy()
    k = a.sum(axis=1)
    shared = a @ a
    denom = np.minimum.outer(k, k) + 1.0 - a
    tom = (shared + a) / np.where(denom <= 0, 1.0, denom)
    np.fill_diagonal(tom, 1.0)
    tom = np.clip(tom, 0.0, 1.0)
    return pd.DataFrame(tom, index=adj.index, columns=adj.columns)


def detect_modules(adj: pd.DataFrame, min_size: int = 12, max_modules: int = 12,
                   method: str = "complete") -> Dict[str, List[str]]:
    """Complete-linkage clustering of 1 - TOM; average linkage chains into one blob."""
    genes = list(adj.index)
    if len(genes) < min_size:
        return {}
    tom = topological_overlap(adj).to_numpy()
    dist = np.clip(1.0 - tom, 0.0, 1.0)
    np.fill_diagonal(dist, 0.0)
    dist = (dist + dist.T) / 2.0
    tree = linkage(squareform(dist, checks=False), method=method)
    labels = fcluster(tree, t=max_modules, criterion="maxclust")

    modules: Dict[str, List[str]] = {}
    index = 1
    for label in sorted(set(labels)):
        members = [genes[i] for i in np.flatnonzero(labels == label)]
        if len(members) >= min_size:
            modules[f"M{index}"] = sorted(members)
            index += 1
    return modules


def coassignment_matrix(partitions: Sequence[Dict[str, List[str]]],
                       genes: Sequence[str]) -> pd.DataFrame:
    """How often each pair of genes ends up in the same module across bootstrap runs."""
    idx = {g: i for i, g in enumerate(genes)}
    n = len(genes)
    counts = np.zeros((n, n))
    denom = np.zeros((n, n))
    for part in partitions:
        labels = np.full(n, -1)
        for k, name in enumerate(sorted(part)):
            for gene in part[name]:
                if gene in idx:
                    labels[idx[gene]] = k
        assigned = labels >= 0
        denom += np.outer(assigned, assigned)
        counts += (labels[:, None] == labels[None, :]) & np.outer(assigned, assigned)
    consensus = np.divide(counts, denom, out=np.zeros_like(counts), where=denom > 0)
    np.fill_diagonal(consensus, 1.0)
    return pd.DataFrame(consensus, index=list(genes), columns=list(genes))


def consensus_modules(consensus: pd.DataFrame, min_size: int = 12,
                      min_coassignment: float = 0.5,
                      method: str = "average") -> Dict[str, List[str]]:
    """Cut the co-assignment matrix by level, not cluster count, so module size follows the data."""
    genes = list(consensus.index)
    if len(genes) < min_size:
        return {}
    dist = np.clip(1.0 - consensus.to_numpy(), 0.0, 1.0)
    np.fill_diagonal(dist, 0.0)
    dist = (dist + dist.T) / 2.0
    tree = linkage(squareform(dist, checks=False), method=method)
    labels = fcluster(tree, t=1.0 - min_coassignment, criterion="distance")

    modules: Dict[str, List[str]] = {}
    index = 1
    for label in sorted(set(labels)):
        members = [genes[i] for i in np.flatnonzero(labels == label)]
        if len(members) >= min_size:
            modules[f"M{index}"] = sorted(members)
            index += 1
    return modules


def module_coherence(consensus: pd.DataFrame,
                     modules: Dict[str, List[str]]) -> pd.Series:
    """Mean co-assignment inside a module; its stability score."""
    out = {}
    for name, genes in modules.items():
        block = consensus.loc[genes, genes].to_numpy()
        iu = np.triu_indices_from(block, k=1)
        out[name] = float(block[iu].mean()) if iu[0].size else float("nan")
    return pd.Series(out, name="coherence")


def module_activity(expr: pd.DataFrame, genes: Sequence[str],
                    method: str = "rank") -> pd.Series:
    """Per-sample activity of a gene set; 'rank' transfers across platforms."""
    present = [g for g in genes if g in expr.index]
    if not present:
        return pd.Series(np.nan, index=expr.columns)
    if method == "rank":
        return rank_normalize_samples(expr).loc[present].mean(axis=0)
    if method == "zscore":
        return zscore_rows(expr.loc[present]).mean(axis=0)
    raise ValueError("method must be 'rank' or 'zscore'")


def activity_table(expr: pd.DataFrame, modules: Dict[str, List[str]],
                   method: str = "rank") -> pd.DataFrame:
    return pd.DataFrame(
        {name: module_activity(expr, genes, method) for name, genes in modules.items()},
        index=expr.columns,
    )


def module_summary(modules: Dict[str, List[str]], adj: pd.DataFrame) -> pd.DataFrame:
    tfs = set(load_tfs())
    rows = []
    for name, genes in modules.items():
        block = adj.loc[genes, genes].to_numpy()
        n = len(genes)
        rows.append({
            "module": name,
            "size": n,
            "n_tf": sum(g in tfs for g in genes),
            "mean_internal_weight": float(block.sum() / max(n * (n - 1), 1)),
            "hub": max(genes, key=lambda g: float(adj.loc[g, genes].sum())),
        })
    return pd.DataFrame(rows).set_index("module").sort_values("size", ascending=False)


def save_modules(modules: Dict[str, List[str]], path: str) -> str:
    rows = [{"module": m, "gene": g} for m, genes in modules.items() for g in genes]
    pd.DataFrame(rows).to_csv(path, sep="\t", index=False)
    return path


def load_modules(path: str) -> Dict[str, List[str]]:
    df = pd.read_csv(path, sep="\t")
    return {m: sorted(g["gene"].tolist()) for m, g in df.groupby("module")}
