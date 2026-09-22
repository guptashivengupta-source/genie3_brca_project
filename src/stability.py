"""Bootstrap resampling to separate reproducible hubs and modules from noise."""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from .network import genie3, hub_degree, link_list, subsample


def bootstrap_networks(expr: pd.DataFrame, n_boot: int, fraction: float = 0.8,
                       seed: int = 0, size: int | None = None, verbose: bool = False,
                       **genie3_kwargs) -> List[pd.DataFrame]:
    rng = np.random.default_rng(seed)
    size = size or max(int(round(expr.shape[1] * fraction)), 5)
    nets: List[pd.DataFrame] = []
    for b in range(n_boot):
        sub = subsample(expr, size, rng, replace=True)
        nets.append(genie3(sub, random_state=seed + 31 * b, **genie3_kwargs))
        if verbose:
            print(f"  bootstrap {b + 1}/{n_boot}", flush=True)
    return nets


def hub_frequency(networks: Sequence[pd.DataFrame], top_edges: int,
                  top_k: int) -> pd.DataFrame:
    """How often each gene lands in the top-k hubs across bootstrap networks."""
    counts: Dict[str, int] = {}
    degrees: Dict[str, List[float]] = {}
    for vim in networks:
        deg = hub_degree(link_list(vim, top_n=top_edges))["total_degree"]
        for gene, value in deg.items():
            degrees.setdefault(gene, []).append(float(value))
        for gene in deg.head(top_k).index:
            counts[gene] = counts.get(gene, 0) + 1

    n = len(networks)
    rows = []
    for gene, values in degrees.items():
        rows.append({
            "gene": gene,
            "bootstrap_frequency": counts.get(gene, 0) / n,
            "mean_degree": float(np.mean(values)),
            "sd_degree": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            "n_bootstrap": n,
        })
    tab = pd.DataFrame(rows).set_index("gene")
    return tab.sort_values(["bootstrap_frequency", "mean_degree"], ascending=False)


def stable_hubs(freq: pd.DataFrame, threshold: float) -> List[str]:
    return list(freq.index[freq["bootstrap_frequency"] >= threshold])


def adjusted_rand(labels_a: Sequence[int], labels_b: Sequence[int]) -> float:
    a = np.asarray(labels_a)
    b = np.asarray(labels_b)
    if a.size != b.size:
        raise ValueError("label vectors must describe the same genes")
    contingency = pd.crosstab(a, b).to_numpy()

    def comb2(x):
        return x * (x - 1) / 2.0

    sum_ij = comb2(contingency).sum()
    sum_i = comb2(contingency.sum(axis=1)).sum()
    sum_j = comb2(contingency.sum(axis=0)).sum()
    total = comb2(float(a.size))
    expected = sum_i * sum_j / total
    maximum = 0.5 * (sum_i + sum_j)
    return 0.0 if maximum == expected else float((sum_ij - expected) / (maximum - expected))


def module_jaccard(reference: Dict[str, List[str]],
                   other: Dict[str, List[str]]) -> pd.DataFrame:
    """Best-matching bootstrap module for each reference module."""
    rows = []
    for name, genes in reference.items():
        ref = set(genes)
        best_name, best_score = None, 0.0
        for cand_name, cand in other.items():
            c = set(cand)
            score = len(ref & c) / len(ref | c) if ref | c else 0.0
            if score > best_score:
                best_name, best_score = cand_name, score
        rows.append({"module": name, "best_match": best_name, "jaccard": best_score})
    return pd.DataFrame(rows).set_index("module")


def module_stability(reference: Dict[str, List[str]],
                     bootstraps: Sequence[Dict[str, List[str]]]) -> pd.DataFrame:
    scores = pd.concat(
        [module_jaccard(reference, b)["jaccard"].rename(i) for i, b in enumerate(bootstraps)],
        axis=1,
    )
    return pd.DataFrame({
        "mean_jaccard": scores.mean(axis=1),
        "sd_jaccard": scores.std(axis=1, ddof=1) if scores.shape[1] > 1 else 0.0,
        "n_bootstrap": scores.shape[1],
        "module_size": pd.Series({k: len(v) for k, v in reference.items()}),
    }).sort_values("mean_jaccard", ascending=False)


def partition_ari(reference: Dict[str, List[str]],
                  bootstraps: Sequence[Dict[str, List[str]]]) -> pd.Series:
    """ARI between the reference partition and each bootstrap partition."""
    genes = sorted({g for v in reference.values() for g in v})
    ref_labels = _label_vector(reference, genes)
    return pd.Series(
        [adjusted_rand(ref_labels, _label_vector(b, genes)) for b in bootstraps],
        name="ari",
    )


def _label_vector(modules: Dict[str, List[str]], genes: Sequence[str]) -> np.ndarray:
    lookup = {g: i for i, (_, members) in enumerate(sorted(modules.items()))
              for g in members}
    return np.array([lookup.get(g, -1) for g in genes])
