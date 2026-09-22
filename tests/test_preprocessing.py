"""Filtering, gene selection and the composition adjustment."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conftest import make_expression
from src.preprocessing import (
    clean_symbols, composition_scores, drop_constant, filter_low_expression,
    rank_normalize_samples, residualize, select_variable_genes, split_by_group,
)


def test_low_expression_genes_are_dropped():
    expr = pd.DataFrame({"S1": [0.0, 5.0], "S2": [0.0, 6.0], "S3": [0.1, 7.0]},
                        index=["silent", "expressed"])
    kept = filter_low_expression(expr, min_value=1.0, min_samples=2)
    assert list(kept.index) == ["expressed"]


def test_unmapped_xena_symbols_are_dropped():
    expr = pd.DataFrame(np.ones((3, 2)), index=["ESR1", "?|652919", "FOO|123"],
                        columns=["a", "b"])
    assert list(clean_symbols(expr).index) == ["ESR1"]


def test_constant_genes_are_dropped():
    expr = pd.DataFrame({"S1": [1.0, 3.0], "S2": [1.0, 9.0]}, index=["flat", "varying"])
    assert list(drop_constant(expr).index) == ["varying"]


def test_gene_selection_returns_exactly_n_and_is_order_independent(expr_medium):
    chosen = select_variable_genes(expr_medium, 10)
    shuffled = select_variable_genes(expr_medium.sample(frac=1.0, random_state=0), 10)
    assert len(chosen) == 10
    assert chosen == shuffled


def test_gene_selection_is_global_not_per_subtype(expr_medium):
    """Both halves must run on the same gene set."""
    left = expr_medium.iloc[:, :60]
    right = expr_medium.iloc[:, 60:]
    genes = select_variable_genes(expr_medium, 10)
    assert set(genes).issubset(left.index) and set(genes).issubset(right.index)


def test_forced_genes_survive_selection(expr_medium):
    chosen = select_variable_genes(expr_medium, 5, force_include=["G029"])
    assert "G029" in chosen


def test_residualize_removes_the_covariate(expr_medium):
    rng = np.random.default_rng(0)
    covariate = pd.DataFrame({"purity": rng.normal(0, 1, expr_medium.shape[1])},
                             index=expr_medium.columns)
    contaminated = expr_medium.add(2.0 * covariate["purity"], axis=1)
    before = contaminated.T.corrwith(covariate["purity"]).abs().mean()
    after = residualize(contaminated, covariate).T.corrwith(covariate["purity"]).abs().mean()
    assert before > 0.5
    assert after < 1e-8


def test_residualize_keeps_gene_means(expr_medium):
    cov = composition_scores(expr_medium)
    out = residualize(expr_medium, cov)
    assert np.allclose(out.mean(axis=1), expr_medium.mean(axis=1))


def test_composition_scores_cover_every_axis(expr_medium):
    scores = composition_scores(expr_medium)
    assert list(scores.index) == list(expr_medium.columns)
    assert {"adipocyte", "stromal", "immune", "proliferation"} == set(scores.columns)


def test_rank_normalization_survives_a_monotone_rescaling(expr_medium):
    """Lets a TCGA module score be compared with a METABRIC one."""
    rescaled = np.sign(expr_medium - 8) * np.abs(expr_medium - 8) ** 0.6 * 3.0 + 2.0
    assert np.allclose(rank_normalize_samples(expr_medium),
                       rank_normalize_samples(rescaled))


def test_split_by_group_enforces_the_minimum_size(expr_medium):
    labels = pd.Series(["A"] * 100 + ["B"] * 20, index=expr_medium.columns)
    groups = split_by_group(expr_medium, labels, ["A", "B"], min_n=50)
    assert set(groups) == {"A"}
    assert groups["A"].shape[1] == 100


def test_align_rejects_disjoint_identifiers():
    from src.preprocessing import align

    expr = make_expression(10, 5, seed=3)
    meta = pd.DataFrame(index=["other-1", "other-2"])
    with pytest.raises(ValueError):
        align(expr, meta)
