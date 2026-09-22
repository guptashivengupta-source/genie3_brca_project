"""Carry the TCGA modules over to METABRIC without refitting anything."""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from .modules import activity_table
from .survival import fit_module_models


def common_genes(expr_a: pd.DataFrame, expr_b: pd.DataFrame) -> List[str]:
    return sorted(set(expr_a.index) & set(expr_b.index))


def module_coverage(modules: Dict[str, List[str]], expr: pd.DataFrame) -> pd.DataFrame:
    present = set(expr.index)
    rows = []
    for name, genes in modules.items():
        kept = [g for g in genes if g in present]
        rows.append({"module": name, "size": len(genes), "mapped": len(kept),
                     "coverage": len(kept) / len(genes) if genes else 0.0,
                     "missing": ",".join(sorted(set(genes) - present))})
    return pd.DataFrame(rows).set_index("module")


def restrict_modules(modules: Dict[str, List[str]], expr: pd.DataFrame,
                     min_coverage: float = 0.5,
                     min_size: int = 5) -> Dict[str, List[str]]:
    """Keep the modules that survive gene mapping; never add genes on the way in."""
    present = set(expr.index)
    out: Dict[str, List[str]] = {}
    for name, genes in modules.items():
        kept = [g for g in genes if g in present]
        if len(kept) >= min_size and len(kept) / max(len(genes), 1) >= min_coverage:
            out[name] = kept
    return out


def _mean_abs_corr(expr: pd.DataFrame, genes: Sequence[str]) -> float:
    block = expr.loc[[g for g in genes if g in expr.index]]
    if len(block) < 2:
        return float("nan")
    corr = np.corrcoef(block.to_numpy(dtype=float))
    iu = np.triu_indices_from(corr, k=1)
    return float(np.nanmean(np.abs(corr[iu])))


def module_preservation(expr: pd.DataFrame, modules: Dict[str, List[str]],
                        n_perm: int = 500, seed: int = 0) -> pd.DataFrame:
    """Within-module correlation against random gene sets of the same size."""
    rng = np.random.default_rng(seed)
    pool = np.asarray(expr.index)
    rows = []
    for name, genes in modules.items():
        observed = _mean_abs_corr(expr, genes)
        size = len([g for g in genes if g in set(expr.index)])
        if size < 2 or not np.isfinite(observed):
            continue
        null = np.array([_mean_abs_corr(expr, rng.choice(pool, size, replace=False))
                         for _ in range(n_perm)])
        sd = null.std(ddof=1)
        rows.append({
            "module": name,
            "size": size,
            "mean_abs_corr": observed,
            "null_mean": float(null.mean()),
            "z_density": float((observed - null.mean()) / sd) if sd > 0 else np.nan,
            "p_value": float((np.sum(null >= observed) + 1) / (n_perm + 1)),
        })
    return pd.DataFrame(rows).set_index("module").sort_values("z_density", ascending=False)


def transfer_activity(expr: pd.DataFrame, modules: Dict[str, List[str]]) -> pd.DataFrame:
    """Rank-based activity, so microarray and RNA-seq scores live on the same scale."""
    return activity_table(expr, modules, method="rank")


def validate_survival(activity: pd.DataFrame, surv: pd.DataFrame,
                      covariates: pd.DataFrame | None = None) -> pd.DataFrame:
    return fit_module_models(activity, surv, covariates)


def concordance(discovery: pd.DataFrame, validation: pd.DataFrame,
                alpha: float = 0.05) -> pd.DataFrame:
    """Replication means the same direction and still significant."""
    d = discovery.set_index("module")
    v = validation.set_index("module")
    shared = [m for m in d.index if m in v.index]
    rows = []
    for m in shared:
        hr_d, hr_v = d.loc[m, "hr_uni"], v.loc[m, "hr_uni"]
        rows.append({
            "module": m,
            "hr_discovery": hr_d,
            "p_discovery": d.loc[m, "p_uni"],
            "hr_validation": hr_v,
            "p_validation": v.loc[m, "p_uni"],
            "same_direction": bool((hr_d - 1.0) * (hr_v - 1.0) > 0),
            "replicated": bool((hr_d - 1.0) * (hr_v - 1.0) > 0
                               and v.loc[m, "p_uni"] <= alpha),
        })
    return pd.DataFrame(rows).set_index("module")
