"""Carrying modules to the validation cohort must not quietly refit anything."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conftest import make_blocks
from src.validation import (
    common_genes, concordance, module_coverage, module_preservation, restrict_modules,
    transfer_activity, validate_survival,
)


@pytest.fixture(scope="module")
def cohorts():
    """Same block structure, different scale, and the validation cohort loses two genes."""
    discovery = make_blocks(120, [12, 12], seed=81)
    validation = make_blocks(200, [12, 12], seed=82)
    validation = np.sign(validation - 8) * np.abs(validation - 8) ** 0.7 * 1.5 + 5.0
    return discovery, validation.drop(index=["B0_00", "B1_00"])


@pytest.fixture(scope="module")
def modules(cohorts):
    discovery, _ = cohorts
    return {"M1": [g for g in discovery.index if g.startswith("B0")],
            "M2": [g for g in discovery.index if g.startswith("B1")]}


def test_common_genes_is_the_intersection(cohorts):
    discovery, validation = cohorts
    shared = common_genes(discovery, validation)
    assert "B0_00" not in shared
    assert len(shared) == len(validation.index)


def test_coverage_reports_what_was_lost(cohorts, modules):
    _, validation = cohorts
    coverage = module_coverage(modules, validation)
    assert coverage.loc["M1", "mapped"] == 11
    assert coverage.loc["M1", "coverage"] == pytest.approx(11 / 12)
    assert coverage.loc["M1", "missing"] == "B0_00"


def test_restrict_modules_never_invents_genes(cohorts, modules):
    _, validation = cohorts
    kept = restrict_modules(modules, validation)
    for name, genes in kept.items():
        assert set(genes).issubset(modules[name])
        assert set(genes).issubset(validation.index)


def test_restrict_modules_drops_poorly_mapped_modules(cohorts, modules):
    _, validation = cohorts
    assert restrict_modules(modules, validation, min_coverage=0.99) == {}


def test_restrict_modules_drops_modules_that_became_too_small(cohorts, modules):
    _, validation = cohorts
    assert restrict_modules(modules, validation, min_size=50) == {}


def test_a_planted_module_is_preserved_in_the_validation_cohort(cohorts, modules):
    _, validation = cohorts
    kept = restrict_modules(modules, validation)
    preservation = module_preservation(validation, kept, n_perm=200, seed=0)
    assert (preservation["z_density"] > 5).all()
    assert (preservation["p_value"] < 0.01).all()


def test_a_random_gene_set_is_not_preserved(cohorts):
    _, validation = cohorts
    rng = np.random.default_rng(0)
    scrambled = {"random": list(rng.choice(validation.index, 12, replace=False))}
    preservation = module_preservation(validation, scrambled, n_perm=200, seed=0)
    assert preservation.loc["random", "p_value"] > 0.01


def test_transfer_activity_uses_the_fixed_gene_list(cohorts, modules):
    _, validation = cohorts
    kept = restrict_modules(modules, validation)
    activity = transfer_activity(validation, kept)
    assert list(activity.columns) == list(kept)
    assert activity.to_numpy().min() >= 0.0 and activity.to_numpy().max() <= 1.0


def test_transfer_activity_tracks_its_own_module(cohorts, modules):
    _, validation = cohorts
    kept = restrict_modules(modules, validation)
    activity = transfer_activity(validation, kept)
    own = np.corrcoef(activity["M1"], validation.loc[kept["M1"]].mean(axis=0))[0, 1]
    other = np.corrcoef(activity["M1"], validation.loc[kept["M2"]].mean(axis=0))[0, 1]
    assert own > 0.6
    assert own > other


def test_validate_survival_returns_one_row_per_module(cohorts, modules):
    _, validation = cohorts
    rng = np.random.default_rng(2)
    kept = restrict_modules(modules, validation)
    activity = transfer_activity(validation, kept)
    n = validation.shape[1]
    surv = pd.DataFrame({"time": rng.exponential(1.0, n),
                         "event": rng.integers(0, 2, n)}, index=validation.columns)
    result = validate_survival(activity, surv)
    assert set(result["module"]) == set(kept)


def test_concordance_flags_a_replicated_module():
    discovery = pd.DataFrame({"module": ["M1", "M2"], "hr_uni": [1.6, 0.7],
                              "p_uni": [0.001, 0.01]})
    validation = pd.DataFrame({"module": ["M1", "M2"], "hr_uni": [1.4, 1.2],
                               "p_uni": [0.02, 0.3]})
    table = concordance(discovery, validation, alpha=0.05)
    assert table.loc["M1", "same_direction"] and table.loc["M1", "replicated"]
    assert not table.loc["M2", "same_direction"]
    assert not table.loc["M2", "replicated"]


def test_concordance_needs_direction_and_significance():
    discovery = pd.DataFrame({"module": ["M1"], "hr_uni": [1.6], "p_uni": [0.001]})
    validation = pd.DataFrame({"module": ["M1"], "hr_uni": [1.5], "p_uni": [0.4]})
    table = concordance(discovery, validation, alpha=0.05)
    assert table.loc["M1", "same_direction"]
    assert not table.loc["M1", "replicated"]


def test_concordance_ignores_modules_missing_from_validation():
    discovery = pd.DataFrame({"module": ["M1", "M2"], "hr_uni": [1.6, 1.1],
                              "p_uni": [0.001, 0.2]})
    validation = pd.DataFrame({"module": ["M1"], "hr_uni": [1.4], "p_uni": [0.02]})
    assert list(concordance(discovery, validation).index) == ["M1"]
