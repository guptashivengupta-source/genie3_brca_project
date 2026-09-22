"""Kaplan-Meier, log-rank and Cox proportional hazards on plain numpy/scipy."""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from .stats_utils import bh_fdr

ENDPOINTS = {"OS": ("OS", "OS.time"), "DSS": ("DSS", "DSS.time"),
             "DFI": ("DFI", "DFI.time"), "PFI": ("PFI", "PFI.time")}


def prepare_endpoint(clinical: pd.DataFrame, endpoint: str = "OS",
                     max_followup: int | None = None) -> pd.DataFrame:
    """Pull one endpoint out of a clinical table and administratively censor it."""
    if endpoint not in ENDPOINTS:
        raise ValueError(f"endpoint must be one of {sorted(ENDPOINTS)}")
    ev_col, tm_col = ENDPOINTS[endpoint]
    out = pd.DataFrame({
        "time": pd.to_numeric(clinical[tm_col], errors="coerce"),
        "event": pd.to_numeric(clinical[ev_col], errors="coerce"),
    }, index=clinical.index).dropna()
    out = out[out["time"] > 0]
    if max_followup is not None:
        censored = out["time"] > max_followup
        out.loc[censored, "event"] = 0
        out.loc[censored, "time"] = max_followup
    out["event"] = out["event"].astype(int)
    return out


def kaplan_meier(time: Sequence[float], event: Sequence[int]) -> pd.DataFrame:
    t = np.asarray(time, dtype=float)
    e = np.asarray(event, dtype=int)
    rows = []
    surv = 1.0
    var = 0.0
    for tt in np.unique(t[e == 1]):
        at_risk = int((t >= tt).sum())
        deaths = int(((t == tt) & (e == 1)).sum())
        surv *= 1.0 - deaths / at_risk
        if at_risk > deaths:
            var += deaths / (at_risk * (at_risk - deaths))
        se = surv * np.sqrt(var)
        rows.append({"time": float(tt), "at_risk": at_risk, "events": deaths,
                     "survival": surv, "se": se,
                     "lower": max(surv - 1.96 * se, 0.0),
                     "upper": min(surv + 1.96 * se, 1.0)})
    return pd.DataFrame(rows)


def median_survival(km: pd.DataFrame) -> float:
    below = km[km["survival"] <= 0.5]
    return float(below["time"].iloc[0]) if len(below) else float("nan")


def logrank_test(time: Sequence[float], event: Sequence[int],
                 group: Sequence) -> Dict[str, object]:
    t = np.asarray(time, dtype=float)
    e = np.asarray(event, dtype=int)
    g = np.asarray(group)
    levels = list(pd.unique(g))
    if len(levels) < 2:
        raise ValueError("log-rank needs at least two groups")
    k = len(levels)

    observed = np.zeros(k)
    expected = np.zeros(k)
    cov = np.zeros((k, k))
    for tt in np.unique(t[e == 1]):
        at_risk = np.array([float(((g == lv) & (t >= tt)).sum()) for lv in levels])
        n = at_risk.sum()
        d = float(((t == tt) & (e == 1)).sum())
        if n <= 1 or d == 0:
            continue
        observed += np.array([float(((g == lv) & (t == tt) & (e == 1)).sum()) for lv in levels])
        p = at_risk / n
        expected += d * p
        factor = d * (n - d) / (n - 1.0)
        cov += factor * (np.diag(p) - np.outer(p, p))

    diff = (observed - expected)[:-1]
    stat = float(diff @ np.linalg.pinv(cov[:-1, :-1]) @ diff)
    df = k - 1
    return {"statistic": stat, "df": df, "p_value": float(stats.chi2.sf(stat, df)),
            "observed": dict(zip(map(str, levels), observed)),
            "expected": dict(zip(map(str, levels), expected))}


def _efron(beta: np.ndarray, X: np.ndarray, time: np.ndarray,
           event: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    n, p = X.shape
    eta = X @ beta
    eta = eta - eta.max()
    w = np.exp(eta)

    order = np.argsort(-time, kind="mergesort")
    Xo, wo, to, eo, etao = X[order], w[order], time[order], event[order], eta[order]
    wx = wo[:, None] * Xo
    wxx = wx[:, :, None] * Xo[:, None, :]
    cw, cwx, cwxx = np.cumsum(wo), np.cumsum(wx, axis=0), np.cumsum(wxx, axis=0)

    nll = 0.0
    grad = np.zeros(p)
    hess = np.zeros((p, p))
    i = 0
    while i < n:
        j = i
        while j + 1 < n and to[j + 1] == to[i]:
            j += 1
        dead = np.flatnonzero(eo[i:j + 1] == 1) + i
        if dead.size:
            d = dead.size
            tw, twx, twxx = wo[dead].sum(), wx[dead].sum(axis=0), wxx[dead].sum(axis=0)
            for k in range(d):
                f = k / d
                a = cw[j] - f * tw
                ax = cwx[j] - f * twx
                axx = cwxx[j] - f * twxx
                nll += np.log(a)
                grad += ax / a
                hess += axx / a - np.outer(ax, ax) / a ** 2
            nll -= etao[dead].sum()
            grad -= Xo[dead].sum(axis=0)
        i = j + 1
    return nll, grad, hess


def cox_ph(X: pd.DataFrame, time: Sequence[float], event: Sequence[int],
           max_iter: int = 50, tol: float = 1e-8) -> pd.DataFrame:
    """Newton-Raphson fit of the Cox model with Efron handling of tied event times."""
    Xv = X.to_numpy(dtype=float)
    t = np.asarray(time, dtype=float)
    e = np.asarray(event, dtype=int)
    if Xv.shape[0] != t.size:
        raise ValueError("covariates and survival times have different lengths")
    if e.sum() < 2:
        raise ValueError("need at least two events to fit a Cox model")

    center = Xv.mean(axis=0)
    scale = Xv.std(axis=0)
    scale[scale < 1e-9] = 1.0
    Z = (Xv - center) / scale

    beta = np.zeros(Z.shape[1])
    nll, grad, hess = _efron(beta, Z, t, e)
    for _ in range(max_iter):
        step = np.linalg.solve(hess + 1e-9 * np.eye(len(beta)), grad)
        new_nll = np.inf
        alpha = 1.0
        for _ in range(20):
            cand = beta - alpha * step
            new_nll, new_grad, new_hess = _efron(cand, Z, t, e)
            if np.isfinite(new_nll) and new_nll <= nll:
                break
            alpha /= 2.0
        if not np.isfinite(new_nll) or new_nll > nll:
            break
        beta, improvement = cand, nll - new_nll
        nll, grad, hess = new_nll, new_grad, new_hess
        if improvement < tol:
            break

    cov = np.linalg.pinv(hess)
    coef = beta / scale
    se = np.sqrt(np.clip(np.diag(cov), 0, None)) / scale
    z = np.divide(coef, se, out=np.zeros_like(coef), where=se > 0)
    return pd.DataFrame({
        "coef": coef,
        "se": se,
        "hazard_ratio": np.exp(coef),
        "ci_lower": np.exp(coef - 1.96 * se),
        "ci_upper": np.exp(coef + 1.96 * se),
        "z": z,
        "p_value": 2.0 * stats.norm.sf(np.abs(z)),
    }, index=list(X.columns))


def cox_score_test(X: pd.DataFrame, time: Sequence[float],
                   event: Sequence[int]) -> Dict[str, float]:
    Xv = X.to_numpy(dtype=float)
    _, grad, hess = _efron(np.zeros(Xv.shape[1]), Xv, np.asarray(time, float),
                           np.asarray(event, int))
    stat = float(grad @ np.linalg.pinv(hess) @ grad)
    df = Xv.shape[1]
    return {"statistic": stat, "df": df, "p_value": float(stats.chi2.sf(stat, df))}


def design_matrix(meta: pd.DataFrame, columns: Sequence[str],
                  reference: Dict[str, str] | None = None) -> pd.DataFrame:
    """Numeric columns kept as they are, categoricals one-hot against a reference level."""
    reference = reference or {}
    blocks: List[pd.DataFrame] = []
    for col in columns:
        s = meta[col]
        if pd.api.types.is_numeric_dtype(s):
            blocks.append(s.astype(float).to_frame(col))
            continue
        s = s.astype("string")
        levels = sorted(x for x in s.dropna().unique())
        base = reference.get(col, levels[0] if levels else None)
        for level in levels:
            if level == base:
                continue
            blocks.append((s == level).astype(float).to_frame(f"{col}[{level}]"))
    if not blocks:
        raise ValueError("no usable covariates")
    out = pd.concat(blocks, axis=1)
    return out.loc[:, out.std() > 0]


def fit_module_models(activity: pd.DataFrame, surv: pd.DataFrame,
                      covariates: pd.DataFrame | None = None) -> pd.DataFrame:
    """Univariable and covariate-adjusted Cox model per module, with BH correction."""
    rows = []
    for module in activity.columns:
        frame = pd.DataFrame({module: activity[module]}).join(surv, how="inner")
        if covariates is not None:
            frame = frame.join(covariates, how="inner")
        frame = frame.dropna()
        if frame["event"].sum() < 5:
            continue
        x = frame[[module]].copy()
        x[module] = (x[module] - x[module].mean()) / (x[module].std() or 1.0)
        uni = cox_ph(x, frame["time"], frame["event"]).loc[module]

        record = {
            "module": module,
            "n": len(frame),
            "events": int(frame["event"].sum()),
            "hr_uni": uni["hazard_ratio"],
            "ci_lower_uni": uni["ci_lower"],
            "ci_upper_uni": uni["ci_upper"],
            "p_uni": uni["p_value"],
        }
        if covariates is not None:
            full = x.join(frame[list(covariates.columns)])
            adj = cox_ph(full, frame["time"], frame["event"]).loc[module]
            record.update({
                "hr_adj": adj["hazard_ratio"],
                "ci_lower_adj": adj["ci_lower"],
                "ci_upper_adj": adj["ci_upper"],
                "p_adj_model": adj["p_value"],
            })
        rows.append(record)

    tab = pd.DataFrame(rows)
    if tab.empty:
        return tab
    tab["q_uni"] = bh_fdr(tab["p_uni"].to_numpy())
    if "p_adj_model" in tab:
        tab["q_adj_model"] = bh_fdr(tab["p_adj_model"].to_numpy())
    return tab.sort_values("p_uni").reset_index(drop=True)


def split_by_median(score: pd.Series) -> pd.Series:
    return pd.Series(np.where(score >= score.median(), "high", "low"), index=score.index)
