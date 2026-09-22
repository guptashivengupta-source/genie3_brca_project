"""Differential rewiring must find a planted difference and stay quiet without one."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conftest import make_expression
from src.network import link_list, matched_ensemble
from src.rewiring import (
    compare_edge_sets, degree_shift, differential_edges, edge_pairs, gene_rewiring,
    jaccard, permutation_null, rewiring_summary, significant_edges,
)

GENES = 30
SAMPLES = 90
N_MATCH = 60
N_REPS = 3
N_PERM = 15
KW = dict(n_trees=40)


def cohort(seed: int, planted: bool, suffix: str = ""):
    expr = make_expression(SAMPLES, GENES, seed=seed, planted=planted)
    expr.columns = [c + suffix for c in expr.columns]
    return expr


def compare(expr_a, expr_b, seed):
    va = matched_ensemble(expr_a, N_MATCH, N_REPS, seed=seed, **KW)
    vb = matched_ensemble(expr_b, N_MATCH, N_REPS, seed=seed + 1, **KW)
    null = permutation_null({"A": expr_a, "B": expr_b}, N_MATCH, N_REPS, N_PERM,
                            seed=seed + 2, **KW)
    return differential_edges(va, vb, null, "A", "B", seed=0)


@pytest.fixture(scope="module")
def rewired():
    """Same genes in both groups, but only group A carries the G000 -> G001 edge."""
    return compare(cohort(31, True), cohort(42, False, "b"), seed=100)


@pytest.fixture(scope="module")
def null_result():
    return compare(cohort(41, False), cohort(42, False, "b"), seed=200)


def test_planted_edge_is_flagged_as_rewired(rewired):
    hit = rewired[(rewired["regulator"] == "G000") & (rewired["target"] == "G001")].iloc[0]
    assert hit["q_value"] <= 0.05
    assert hit["favours"] == "A"


def test_planted_edge_is_near_the_top_of_the_ranking(rewired):
    position = rewired.index[(rewired["regulator"] == "G000") &
                             (rewired["target"] == "G001")][0]
    assert position < 10


def test_two_groups_from_the_same_process_give_no_hits(null_result):
    assert len(significant_edges(null_result, 0.05)) == 0


def test_p_values_are_roughly_uniform_under_the_null(null_result):
    p = null_result["p_value"]
    assert 0.02 <= (p <= 0.05).mean() <= 0.12
    assert 0.05 <= (p <= 0.10).mean() <= 0.20


def test_delta_sign_points_at_the_right_group(rewired):
    assert ((rewired["delta"] > 0) == (rewired["favours"] == "A")).all()


def test_every_ordered_pair_is_tested_once(rewired):
    assert len(rewired) == GENES * (GENES - 1)
    assert not rewired.duplicated(["regulator", "target"]).any()


def test_null_must_cover_every_edge():
    a, b = cohort(41, False), cohort(42, False, "b")
    va = matched_ensemble(a, N_MATCH, 2, seed=0, **KW)
    null = permutation_null({"A": a, "B": b}, N_MATCH, 2, 3, seed=1, **KW).iloc[10:]
    with pytest.raises(ValueError):
        differential_edges(va, va, null, "A", "B")


def test_edge_set_arithmetic():
    a = pd.DataFrame({"regulator": ["X", "Y"], "target": ["A", "B"]})
    b = pd.DataFrame({"regulator": ["X", "Z"], "target": ["A", "C"]})
    assert edge_pairs(a) == {("X", "A"), ("Y", "B")}
    assert jaccard(edge_pairs(a), edge_pairs(b)) == pytest.approx(1 / 3)
    summary = compare_edge_sets(a, b, "a", "b")
    assert summary["shared"] == 1 and summary["a_specific"] == 1 and summary["b_specific"] == 1


def test_jaccard_of_disjoint_and_identical_sets():
    assert jaccard(set(), set()) == 0.0
    assert jaccard({("X", "A")}, {("X", "A")}) == 1.0
    assert jaccard({("X", "A")}, {("Y", "B")}) == 0.0


def test_gene_rewiring_puts_the_planted_pair_on_top(rewired):
    table = gene_rewiring(significant_edges(rewired, 0.05))
    assert {"G000", "G001"} & set(table.head(4).index)


def test_degree_shift_is_antisymmetric():
    a = matched_ensemble(cohort(31, True), N_MATCH, 2, seed=0, **KW)
    b = matched_ensemble(cohort(42, False), N_MATCH, 2, seed=1, **KW)
    edges_a, edges_b = link_list(a, top_n=30), link_list(b, top_n=30)
    ab = degree_shift(edges_a, edges_b, "A", "B")
    ba = degree_shift(edges_b, edges_a, "B", "A")
    assert np.allclose(ab["shift"].sort_index(), -ba["shift"].sort_index())


def test_rewiring_summary_covers_every_pair():
    nets = {name: matched_ensemble(cohort(seed, False), N_MATCH, 1, seed=seed, **KW)
            for name, seed in (("A", 51), ("B", 52), ("C", 53))}
    assert len(rewiring_summary(nets, top_edges=20)) == 3
