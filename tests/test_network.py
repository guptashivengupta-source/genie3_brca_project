"""GENIE3 inference, and the sample-size matching the subtype comparison depends on."""

from __future__ import annotations

import numpy as np
import pytest

from conftest import make_expression
from src.network import (
    genie3, hub_degree, link_list, matched_ensemble, matched_networks, rank_normalize,
    symmetric_adjacency, top_hubs,
)


def test_planted_regulator_is_the_top_edge(expr_small):
    edges = link_list(genie3(expr_small, n_trees=60, random_state=0), top_n=1)
    assert set(edges.loc[0, ["regulator", "target"]]) == {"G000", "G001"}


def test_no_self_edges(expr_small):
    vim = genie3(expr_small, n_trees=30, random_state=0)
    assert np.allclose(np.diag(vim.to_numpy()), 0.0)
    edges = link_list(vim)
    assert not (edges["regulator"] == edges["target"]).any()


def test_same_seed_gives_the_same_network(expr_small):
    a = genie3(expr_small, n_trees=30, random_state=7)
    b = genie3(expr_small, n_trees=30, random_state=7)
    assert a.equals(b)


def test_different_seed_gives_a_different_network(expr_small):
    a = genie3(expr_small, n_trees=30, random_state=7)
    b = genie3(expr_small, n_trees=30, random_state=8)
    assert not np.allclose(a.to_numpy(), b.to_numpy())


def test_only_declared_regulators_appear(expr_small):
    regulators = ["G000", "G002", "G004"]
    vim = genie3(expr_small, regulators=regulators, n_trees=20, random_state=0)
    assert list(vim.index) == regulators
    assert set(link_list(vim)["regulator"]).issubset(regulators)


def test_unknown_regulators_are_ignored_not_fatal(expr_small):
    vim = genie3(expr_small, regulators=["G000", "NOT_A_GENE"], n_trees=20, random_state=0)
    assert list(vim.index) == ["G000"]


def test_unknown_tree_method_is_rejected(expr_small):
    with pytest.raises(ValueError):
        genie3(expr_small, tree_method="XGB")


def test_extra_trees_runs_and_differs_from_random_forest(expr_small):
    rf = genie3(expr_small, tree_method="RF", n_trees=40, random_state=0)
    et = genie3(expr_small, tree_method="ET", n_trees=40, random_state=0)
    assert rf.shape == et.shape
    assert not np.allclose(rf.to_numpy(), et.to_numpy())


def test_rank_normalize_is_bounded_and_per_target(expr_small):
    ranks = rank_normalize(genie3(expr_small, n_trees=20, random_state=0))
    assert ranks.to_numpy().min() >= 0.0 and ranks.to_numpy().max() <= 1.0
    assert np.allclose(ranks.max(axis=0), 1.0)


def test_link_list_is_sorted_and_truncated(expr_small):
    edges = link_list(genie3(expr_small, n_trees=20, random_state=0), top_n=15)
    assert len(edges) == 15
    assert edges["importance"].is_monotonic_decreasing


def test_symmetric_adjacency_is_symmetric_with_empty_diagonal(expr_small):
    adj = symmetric_adjacency(genie3(expr_small, n_trees=20, random_state=0)).to_numpy()
    assert np.allclose(adj, adj.T)
    assert np.allclose(np.diag(adj), 0.0)


def test_hub_degree_counts_both_directions(expr_small):
    edges = link_list(genie3(expr_small, n_trees=20, random_state=0), top_n=20)
    table = hub_degree(edges)
    assert np.isclose(table["total_degree"].sum(), 2 * len(edges))
    assert table["total_degree"].is_monotonic_decreasing
    assert len(top_hubs(edges, 3)) == 3


def test_matched_ensemble_refuses_to_oversample():
    expr = make_expression(20, 8, seed=4)
    with pytest.raises(ValueError):
        matched_ensemble(expr, n_match=50, n_reps=2, n_trees=10)


def test_matched_ensemble_returns_the_replicates():
    expr = make_expression(40, 8, seed=4)
    mean, reps = matched_ensemble(expr, n_match=30, n_reps=4, seed=0, return_reps=True,
                                  n_trees=10)
    assert len(reps) == 4
    assert np.allclose(mean.to_numpy(), sum(r.to_numpy() for r in reps) / 4)


def test_replicates_differ_even_when_the_group_is_exactly_n_match():
    """Without drawing with replacement a minority subtype would look perfectly stable."""
    expr = make_expression(40, 8, seed=4)
    _, reps = matched_ensemble(expr, n_match=40, n_reps=3, seed=0, return_reps=True,
                               n_trees=10)
    assert not np.allclose(reps[0].to_numpy(), reps[1].to_numpy())


def test_matching_removes_the_advantage_of_the_larger_group():
    """A big group must not get systematically sharper edges than a small one."""
    small = make_expression(60, 15, seed=21)
    large = make_expression(240, 15, seed=22)

    unmatched_small = genie3(small, n_trees=60, random_state=0)
    unmatched_large = genie3(large, n_trees=60, random_state=0)
    gap_unmatched = abs(unmatched_large.to_numpy().max() - unmatched_small.to_numpy().max())

    matched = matched_networks({"small": small, "large": large}, n_match=60, n_reps=4,
                               seed=0, n_trees=60)
    gap_matched = abs(matched["large"].to_numpy().max() - matched["small"].to_numpy().max())
    assert gap_matched < gap_unmatched
