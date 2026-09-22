"""Fixtures: small matrices with known structure, plus a full synthetic two-cohort dataset."""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))


def make_expression(n_samples: int, n_genes: int, seed: int,
                    planted: bool = True) -> pd.DataFrame:
    """G000 drives G001; G002 and G003 follow a second factor; the rest is noise."""
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, (n_genes, n_samples))
    if planted:
        X[1] = 2.0 * X[0] + 0.15 * rng.normal(0, 1, n_samples)
        factor = rng.normal(0, 1, n_samples)
        X[2] = factor + 0.2 * rng.normal(0, 1, n_samples)
        X[3] = factor + 0.2 * rng.normal(0, 1, n_samples)
    genes = [f"G{i:03d}" for i in range(n_genes)]
    samples = [f"S{i:03d}" for i in range(n_samples)]
    return pd.DataFrame(X + 8.0, index=genes, columns=samples)


def make_blocks(n_samples: int, block_sizes, seed: int, strength: float = 0.9) -> pd.DataFrame:
    """One latent factor per block, so the block members co-express."""
    rng = np.random.default_rng(seed)
    rows, names = [], []
    for b, size in enumerate(block_sizes):
        factor = rng.normal(0, 1, n_samples)
        rows.append(strength * np.outer(np.ones(size), factor)
                    + np.sqrt(1 - strength ** 2) * rng.normal(0, 1, (size, n_samples)))
        names += [f"B{b}_{i:02d}" for i in range(size)]
    return pd.DataFrame(np.vstack(rows) + 8.0, index=names,
                        columns=[f"S{i:03d}" for i in range(n_samples)])


def make_hub(n_samples: int, n_targets: int, n_noise: int, seed: int) -> pd.DataFrame:
    """HUB drives n_targets genes, so it is a hub by degree and not just one strong edge."""
    rng = np.random.default_rng(seed)
    driver = rng.normal(0, 1, n_samples)
    targets = np.array([1.8 * driver + 0.25 * rng.normal(0, 1, n_samples)
                        for _ in range(n_targets)])
    noise = rng.normal(0, 1, (n_noise, n_samples))
    values = np.vstack([driver[None, :], targets, noise]) + 8.0
    names = (["HUB"] + [f"T{i:02d}" for i in range(n_targets)]
             + [f"N{i:02d}" for i in range(n_noise)])
    return pd.DataFrame(values, index=names,
                        columns=[f"S{i:03d}" for i in range(n_samples)])


@pytest.fixture(scope="session")
def expr_small() -> pd.DataFrame:
    return make_expression(60, 12, seed=1)


@pytest.fixture(scope="session")
def expr_medium() -> pd.DataFrame:
    return make_expression(120, 30, seed=2)


@pytest.fixture(scope="session")
def synthetic_root(tmp_path_factory) -> str:
    from make_synthetic import make

    return make(str(tmp_path_factory.mktemp("cohorts")), seed=11)


@pytest.fixture
def survival_data():
    """Exponential times with a known log hazard ratio of 0.8 on x."""
    rng = np.random.default_rng(5)
    n = 400
    x = rng.normal(0, 1, n)
    event_time = rng.exponential(1.0 / np.exp(0.8 * x))
    censor_time = rng.exponential(1.5, n)
    time = np.minimum(event_time, censor_time)
    event = (event_time <= censor_time).astype(int)
    return pd.DataFrame({"x": x}), time, event
