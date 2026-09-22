"""Module detection, consensus modules and the activity score used for validation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conftest import make_blocks
from src.modules import (
    activity_table, coassignment_matrix, consensus_modules, detect_modules, load_modules,
    module_activity, module_coherence, module_summary, save_modules, scale_adjacency,
    topological_overlap,
)


def block_adjacency(sizes=(15, 15), noise=0.05, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = sum(sizes)
    names = [f"B{b}_{i:02d}" for b, size in enumerate(sizes) for i in range(size)]
    adj = rng.uniform(0, noise, (n, n))
    start = 0
    for size in sizes:
        block = slice(start, start + size)
        adj[block, block] = rng.uniform(0.7, 1.0, (size, size))
        start += size
    adj = (adj + adj.T) / 2
    np.fill_diagonal(adj, 0.0)
    return pd.DataFrame(adj, index=names, columns=names)


def test_scale_adjacency_lands_in_the_unit_interval():
    scaled = scale_adjacency(block_adjacency() * 17.0).to_numpy()
    assert scaled.min() >= 0.0 and scaled.max() <= 1.0
    assert np.allclose(np.diag(scaled), 0.0)


def test_scale_adjacency_of_an_empty_network():
    empty = pd.DataFrame(np.zeros((3, 3)), index=list("abc"), columns=list("abc"))
    assert (scale_adjacency(empty).to_numpy() == 0).all()


def test_topological_overlap_is_symmetric_and_bounded():
    tom = topological_overlap(block_adjacency()).to_numpy()
    assert np.allclose(tom, tom.T)
    assert tom.min() >= 0.0 and tom.max() <= 1.0
    assert np.allclose(np.diag(tom), 1.0)


def test_topological_overlap_is_higher_inside_a_block():
    tom = topological_overlap(block_adjacency())
    inside = tom.iloc[0, 1]
    across = tom.iloc[0, -1]
    assert inside > across


def test_detect_modules_recovers_planted_blocks():
    modules = detect_modules(block_adjacency(), min_size=5, max_modules=4)
    assert len(modules) == 2
    for genes in modules.values():
        prefixes = {g.split("_")[0] for g in genes}
        assert len(prefixes) == 1


def test_detect_modules_drops_everything_below_min_size():
    assert detect_modules(block_adjacency(), min_size=100, max_modules=4) == {}


def test_detect_modules_on_a_tiny_network():
    small = block_adjacency(sizes=(2, 2))
    assert detect_modules(small, min_size=10, max_modules=2) == {}


def test_coassignment_counts_only_assigned_genes():
    partitions = [{"M1": ["a", "b"]}, {"M1": ["a", "b"]}, {"M1": ["a"], "M2": ["b"]}]
    cons = coassignment_matrix(partitions, ["a", "b", "c"])
    assert cons.loc["a", "b"] == pytest.approx(2 / 3)
    assert cons.loc["a", "c"] == 0.0
    assert cons.loc["a", "a"] == 1.0


def test_consensus_modules_recover_blocks_from_noisy_partitions():
    rng = np.random.default_rng(3)
    genes = [f"B0_{i:02d}" for i in range(12)] + [f"B1_{i:02d}" for i in range(12)]
    partitions = []
    for _ in range(20):
        noisy = {"M1": [g for g in genes if g.startswith("B0")],
                 "M2": [g for g in genes if g.startswith("B1")]}
        swap = rng.integers(0, 12)
        noisy["M1"][swap], noisy["M2"][swap] = noisy["M2"][swap], noisy["M1"][swap]
        partitions.append(noisy)
    cons = coassignment_matrix(partitions, genes)
    modules = consensus_modules(cons, min_size=5, min_coassignment=0.5)
    assert len(modules) == 2
    for members in modules.values():
        assert len({g.split("_")[0] for g in members}) == 1


def test_consensus_modules_drop_genes_that_never_settle():
    """A gene that lands somewhere different every time belongs in no module."""
    rng = np.random.default_rng(4)
    genes = [f"B0_{i:02d}" for i in range(12)] + [f"DRIFTER{i}" for i in range(6)]
    partitions = []
    for _ in range(20):
        drifters = [g for g in genes if g.startswith("DRIFTER")]
        rng.shuffle(drifters)
        partitions.append({"M1": genes[:12] + drifters[:3], "M2": drifters[3:]})
    modules = consensus_modules(coassignment_matrix(partitions, genes), min_size=5,
                                min_coassignment=0.8)
    assert len(modules) == 1
    assert not any(g.startswith("DRIFTER") for g in modules["M1"])


def test_a_stricter_coassignment_cut_never_grows_a_module():
    rng = np.random.default_rng(5)
    genes = [f"G{i:02d}" for i in range(24)]
    partitions = [{"M1": list(rng.choice(genes, 12, replace=False))} for _ in range(20)]
    cons = coassignment_matrix(partitions, genes)
    loose = consensus_modules(cons, min_size=3, min_coassignment=0.2)
    strict = consensus_modules(cons, min_size=3, min_coassignment=0.8)
    assert sum(len(g) for g in strict.values()) <= sum(len(g) for g in loose.values())


def test_module_coherence_ranks_a_solid_module_above_a_shaky_one():
    genes = list("abcdef")
    solid = [{"M1": ["a", "b", "c"], "M2": ["d", "e", "f"]}] * 10
    shaky = [{"M1": ["a", "b", "d"], "M2": ["c", "e", "f"]}]
    cons = coassignment_matrix(solid + shaky, genes)
    coherence = module_coherence(cons, {"solid": ["a", "b"], "mixed": ["a", "d"]})
    assert coherence["solid"] > coherence["mixed"]


def test_module_activity_survives_a_monotone_rescaling():
    """TCGA counts and METABRIC array intensities must give the same rank score."""
    expr = make_blocks(40, [10, 10], seed=5)
    genes = [g for g in expr.index if g.startswith("B0")]
    rescaled = np.sign(expr - 8) * np.abs(expr - 8) ** 0.5 * 2.5 + 3.0
    assert np.allclose(module_activity(expr, genes), module_activity(rescaled, genes))


def test_zscore_activity_does_not_survive_that_rescaling():
    expr = make_blocks(40, [10, 10], seed=5)
    genes = [g for g in expr.index if g.startswith("B0")]
    rescaled = np.sign(expr - 8) * np.abs(expr - 8) ** 0.5 * 2.5 + 3.0
    assert not np.allclose(module_activity(expr, genes, method="zscore"),
                           module_activity(rescaled, genes, method="zscore"))


def test_module_activity_ignores_genes_that_are_missing():
    expr = make_blocks(30, [6], seed=6)
    present = module_activity(expr, list(expr.index))
    padded = module_activity(expr, list(expr.index) + ["NOT_MEASURED"])
    assert np.allclose(present, padded)


def test_module_activity_of_an_absent_gene_set_is_missing():
    expr = make_blocks(30, [6], seed=6)
    assert module_activity(expr, ["NOPE"]).isna().all()


def test_unknown_activity_method_is_rejected():
    expr = make_blocks(30, [6], seed=6)
    with pytest.raises(ValueError):
        module_activity(expr, list(expr.index), method="mean")


def test_activity_table_has_one_column_per_module():
    expr = make_blocks(30, [6, 6], seed=7)
    modules = {"M1": list(expr.index[:6]), "M2": list(expr.index[6:])}
    table = activity_table(expr, modules)
    assert list(table.columns) == ["M1", "M2"]
    assert list(table.index) == list(expr.columns)


def test_module_summary_reports_size_and_hub():
    adj = block_adjacency()
    modules = detect_modules(adj, min_size=5, max_modules=4)
    summary = module_summary(modules, adj)
    assert set(summary.columns) >= {"size", "n_tf", "mean_internal_weight", "hub"}
    assert (summary["size"] >= 5).all()
    assert all(summary.loc[m, "hub"] in modules[m] for m in summary.index)


def test_modules_round_trip_through_disk(tmp_path):
    modules = {"M1": ["a", "b"], "M2": ["c"]}
    path = str(tmp_path / "modules.tsv")
    save_modules(modules, path)
    assert load_modules(path) == modules
