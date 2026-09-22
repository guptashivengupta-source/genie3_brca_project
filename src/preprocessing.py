"""Expression filtering, gene selection and composition adjustment."""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

from .signatures import COMPOSITION


def clean_symbols(expr: pd.DataFrame) -> pd.DataFrame:
    """Drop the unmapped Xena rows such as "?|652919"."""
    ok = [g for g in expr.index if g and not g.startswith("?") and "|" not in g]
    return expr.loc[ok]


def filter_low_expression(expr: pd.DataFrame, min_value: float = 1.0,
                          min_samples: int = 20) -> pd.DataFrame:
    keep = (expr >= min_value).sum(axis=1) >= min_samples
    return expr.loc[keep]


def log2_transform(expr: pd.DataFrame, already_log: bool = False) -> pd.DataFrame:
    return expr if already_log else np.log2(expr.clip(lower=0) + 1.0)


def drop_constant(expr: pd.DataFrame, eps: float = 1e-8) -> pd.DataFrame:
    return expr.loc[expr.std(axis=1) > eps]


def select_variable_genes(expr: pd.DataFrame, n: int,
                          force_include: Iterable[str] = ()) -> List[str]:
    """Top-n by variance over all samples, so every subtype shares one gene set."""
    forced = [g for g in force_include if g in expr.index]
    ranked = expr.var(axis=1).sort_values(ascending=False)
    chosen = list(dict.fromkeys(forced + list(ranked.index)))[:n]
    return sorted(chosen)


def zscore_rows(expr: pd.DataFrame) -> pd.DataFrame:
    sd = expr.std(axis=1).replace(0.0, np.nan)
    return expr.sub(expr.mean(axis=1), axis=0).div(sd, axis=0).fillna(0.0)


def signature_score(expr: pd.DataFrame, genes: Sequence[str]) -> pd.Series:
    present = [g for g in genes if g in expr.index]
    if not present:
        return pd.Series(0.0, index=expr.columns)
    return zscore_rows(expr.loc[present]).mean(axis=0)


def composition_scores(expr: pd.DataFrame,
                       axes: Sequence[str] | None = None) -> pd.DataFrame:
    axes = list(axes) if axes is not None else list(COMPOSITION)
    return pd.DataFrame(
        {a: signature_score(expr, COMPOSITION[a]) for a in axes},
        index=expr.columns,
    )


def residualize(expr: pd.DataFrame, covariates: pd.DataFrame) -> pd.DataFrame:
    """Remove the covariate axes from every gene by OLS, keeping the gene mean."""
    cov = covariates.reindex(expr.columns).astype(float)
    cov = cov.fillna(cov.mean())
    X = np.column_stack([np.ones(len(cov)), cov.to_numpy()])
    Y = expr.to_numpy(dtype=float).T
    beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
    resid = Y - X @ beta
    out = resid.T + expr.mean(axis=1).to_numpy()[:, None]
    return pd.DataFrame(out, index=expr.index, columns=expr.columns)


def align(expr: pd.DataFrame, meta: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    common = [s for s in expr.columns if s in set(meta.index)]
    if not common:
        raise ValueError("expression and metadata share no sample identifiers")
    return expr.loc[:, common], meta.loc[common]


def split_by_group(expr: pd.DataFrame, labels: pd.Series,
                   keep: Sequence[str], min_n: int) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    for g in keep:
        cols = [s for s in expr.columns if labels.get(s) == g]
        if len(cols) >= min_n:
            out[g] = expr.loc[:, cols]
    return out


def rank_normalize_samples(expr: pd.DataFrame) -> pd.DataFrame:
    """Within-sample percentile ranks; makes scores comparable across platforms."""
    ranks = expr.rank(axis=0, method="average")
    return ranks / float(len(expr.index))
