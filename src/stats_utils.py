"""Empirical p-values and multiple-testing correction."""

from __future__ import annotations

import numpy as np


def empirical_p(observed: np.ndarray, null: np.ndarray) -> np.ndarray:
    """Right-tailed p with the +1 correction, so p is never exactly zero."""
    observed = np.asarray(observed, dtype=float)
    null = np.asarray(null, dtype=float)
    if null.ndim == 1:
        null = np.tile(null, (observed.size, 1))
    hits = (null >= observed[:, None]).sum(axis=1)
    return (hits + 1.0) / (null.shape[1] + 1.0)


def bh_fdr(pvalues: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values."""
    p = np.asarray(pvalues, dtype=float)
    n = p.size
    if n == 0:
        return p
    order = np.argsort(p)
    ranked = p[order] * n / np.arange(1, n + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(ranked, 0.0, 1.0)
    return out
