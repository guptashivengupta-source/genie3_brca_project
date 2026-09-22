"""Bootstrap hub stability and the partition-agreement measures."""

from __future__ import annotations

import numpy as np
import pytest

from conftest import make_blocks, make_expression, make_hub
from src.network import genie3
from src.stability import (
    adjusted_rand, bootstrap_networks, hub_frequency, module_jaccard, module_stability,
    partition_ari, stable_hubs,
)


@pytest.fixture(scope="module")
def bootstrap_nets():
    expr = make_hub(80, n_targets=6, n_noise=10, seed=61)
    return bootstrap_networks(expr, n_boot=10, seed=0, size=60, n_trees=40)


def test_bootstrap_networks_all_share_the_gene_set(bootstrap_nets):
    assert len(bootstrap_nets) == 10
    assert all(net.shape == bootstrap_nets[0].shape for net in bootstrap_nets)
    assert all(list(net.index) == list(bootstrap_nets[0].index) for net in bootstrap_nets)


def test_bootstrap_networks_are_not_all_identical(bootstrap_nets):
    assert not np.allclose(bootstrap_nets[0].to_numpy(), bootstrap_nets[1].to_numpy())


def test_every_stable_hub_comes_from_the_planted_part(bootstrap_nets):
    """GENIE3 spreads importance over correlated targets, so the set ranks high, not the driver."""
    freq = hub_frequency(bootstrap_nets, top_edges=40, top_k=5)
    top = freq.sort_values("bootstrap_frequency", ascending=False).head(5)
    assert all(g == "HUB" or g.startswith("T") for g in top.index)
    assert freq["bootstrap_frequency"].max() == 1.0


def test_hub_frequency_reports_the_spread(bootstrap_nets):
    freq = hub_frequency(bootstrap_nets, top_edges=40, top_k=5)
    assert (freq["sd_degree"] >= 0).all()
    assert (freq["n_bootstrap"] == 10).all()


def test_stable_hubs_respects_the_threshold(bootstrap_nets):
    freq = hub_frequency(bootstrap_nets, top_edges=40, top_k=5)
    assert set(stable_hubs(freq, 1.0)).issubset(stable_hubs(freq, 0.5))
    assert stable_hubs(freq, 1.01) == []


def test_adjusted_rand_on_known_partitions():
    assert adjusted_rand([0, 0, 1, 1], [0, 0, 1, 1]) == pytest.approx(1.0)
    assert adjusted_rand([0, 0, 1, 1], [1, 1, 0, 0]) == pytest.approx(1.0)
    assert adjusted_rand([0, 0, 1, 1], [0, 1, 0, 1]) == pytest.approx(-0.5)
    assert adjusted_rand([0, 0, 0, 0], [0, 0, 0, 0]) == pytest.approx(0.0)


def test_adjusted_rand_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        adjusted_rand([0, 1], [0, 1, 1])


def test_module_jaccard_finds_the_best_match():
    reference = {"M1": ["a", "b", "c"], "M2": ["d", "e"]}
    other = {"X": ["a", "b", "c", "z"], "Y": ["d", "e"]}
    table = module_jaccard(reference, other)
    assert table.loc["M1", "best_match"] == "X"
    assert table.loc["M1", "jaccard"] == pytest.approx(0.75)
    assert table.loc["M2", "jaccard"] == pytest.approx(1.0)


def test_module_stability_of_a_perfectly_reproduced_partition():
    reference = {"M1": ["a", "b", "c"], "M2": ["d", "e", "f"]}
    table = module_stability(reference, [reference, reference, reference])
    assert (table["mean_jaccard"] == 1.0).all()
    assert (table["sd_jaccard"] == 0.0).all()
    assert list(table["module_size"]) == [3, 3]


def test_partition_ari_is_one_when_nothing_moves():
    reference = {"M1": ["a", "b"], "M2": ["c", "d"]}
    assert partition_ari(reference, [reference, reference]).tolist() == [1.0, 1.0]


def test_partition_ari_drops_when_genes_move():
    reference = {"M1": ["a", "b", "c", "d"], "M2": ["e", "f", "g", "h"]}
    shuffled = {"M1": ["a", "b", "e", "f"], "M2": ["c", "d", "g", "h"]}
    assert partition_ari(reference, [shuffled]).iloc[0] < 0.5


def test_noise_genes_are_not_stable_hubs(bootstrap_nets):
    freq = hub_frequency(bootstrap_nets, top_edges=40, top_k=5)
    noise = [g for g in freq.index if g.startswith("N")]
    planted = [g for g in freq.index if g not in noise]
    assert freq.loc[noise, "bootstrap_frequency"].max() < 0.7
    assert freq.loc[planted, "mean_degree"].min() > freq.loc[noise, "mean_degree"].max()


def test_block_genes_cluster_together_across_bootstraps():
    expr = make_blocks(90, [8, 8], seed=70)
    nets = bootstrap_networks(expr, n_boot=5, seed=0, size=70, n_trees=40)
    freq = hub_frequency(nets, top_edges=30, top_k=8)
    assert freq["bootstrap_frequency"].max() >= 0.6


def test_genie3_and_bootstrap_agree_on_the_gene_order():
    expr = make_expression(60, 10, seed=71)
    direct = genie3(expr, n_trees=20, random_state=0)
    boot = bootstrap_networks(expr, n_boot=1, seed=0, size=50, n_trees=20)[0]
    assert list(direct.index) == list(boot.index)
    assert list(direct.columns) == list(boot.columns)
