"""Kaplan-Meier, log-rank, Cox and the multiple-testing correction."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.stats_utils import bh_fdr, empirical_p
from src.survival import (
    cox_ph, cox_score_test, design_matrix, fit_module_models, kaplan_meier, logrank_test,
    median_survival, prepare_endpoint, split_by_median,
)


def test_kaplan_meier_matches_a_hand_computed_curve():
    # 5 at risk, death at t=1; 4 at risk, censor at t=2; 3 at risk, death at t=3
    time = [1, 2, 3, 4, 5]
    event = [1, 0, 1, 0, 0]
    km = kaplan_meier(time, event)
    assert list(km["time"]) == [1.0, 3.0]
    assert km.loc[0, "survival"] == pytest.approx(0.8)
    assert km.loc[1, "survival"] == pytest.approx(0.8 * 2 / 3)


def test_kaplan_meier_confidence_band_brackets_the_estimate():
    km = kaplan_meier([1, 2, 3, 4, 5, 6], [1, 1, 0, 1, 0, 1])
    assert (km["lower"] <= km["survival"]).all()
    assert (km["survival"] <= km["upper"]).all()
    assert (km["survival"].diff().dropna() <= 0).all()


def test_median_survival_is_the_first_time_below_half():
    km = kaplan_meier([1, 2, 3, 4], [1, 1, 1, 1])
    assert median_survival(km) == 2.0


def test_median_survival_is_undefined_when_the_curve_stays_high():
    km = kaplan_meier([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    assert np.isnan(median_survival(km))


def test_cox_recovers_a_known_hazard_ratio(survival_data):
    X, time, event = survival_data
    fit = cox_ph(X, time, event)
    assert fit.loc["x", "ci_lower"] < np.exp(0.8) < fit.loc["x", "ci_upper"]
    assert fit.loc["x", "p_value"] < 1e-6


def test_cox_is_invariant_to_rescaling_a_covariate(survival_data):
    X, time, event = survival_data
    plain = cox_ph(X, time, event).loc["x", "coef"]
    scaled = cox_ph(X * 10.0, time, event).loc["x", "coef"]
    assert scaled == pytest.approx(plain / 10.0, rel=1e-4)


def test_cox_score_test_equals_the_logrank_for_a_binary_covariate(survival_data):
    """Textbook identity, and the sharpest check that both implementations are right."""
    X, time, event = survival_data
    group = (X["x"] > 0).astype(float)
    score = cox_score_test(group.to_frame("g"), time, event)
    lr = logrank_test(time, event, group.to_numpy())
    assert score["statistic"] == pytest.approx(lr["statistic"], rel=1e-8)
    assert score["p_value"] == pytest.approx(lr["p_value"], rel=1e-8)


def test_cox_needs_events():
    X = pd.DataFrame({"x": [0.0, 1.0, 2.0]})
    with pytest.raises(ValueError):
        cox_ph(X, [1, 2, 3], [0, 0, 0])


def test_cox_rejects_mismatched_lengths():
    X = pd.DataFrame({"x": [0.0, 1.0, 2.0]})
    with pytest.raises(ValueError):
        cox_ph(X, [1, 2], [1, 1])


def test_cox_handles_tied_event_times():
    rng = np.random.default_rng(12)
    n = 200
    x = rng.normal(0, 1, n)
    time = np.round(rng.exponential(1.0 / np.exp(0.7 * x)) * 5)
    event = rng.integers(0, 2, n)
    fit = cox_ph(pd.DataFrame({"x": x}), np.maximum(time, 1), event)
    assert np.isfinite(fit.loc["x", "coef"])
    assert fit.loc["x", "coef"] > 0


def test_logrank_separates_two_very_different_arms():
    time = list(range(1, 21)) + list(range(50, 70))
    event = [1] * 40
    group = ["early"] * 20 + ["late"] * 20
    result = logrank_test(time, event, group)
    assert result["p_value"] < 1e-6
    assert result["df"] == 1


def test_logrank_on_identical_arms_is_not_significant():
    rng = np.random.default_rng(3)
    time = rng.exponential(1.0, 300)
    event = np.ones(300, dtype=int)
    group = np.array(["a", "b"] * 150)
    assert logrank_test(time, event, group)["p_value"] > 0.05


def test_logrank_supports_three_arms():
    rng = np.random.default_rng(4)
    time = rng.exponential(1.0, 300)
    event = np.ones(300, dtype=int)
    group = np.array(["a", "b", "c"] * 100)
    assert logrank_test(time, event, group)["df"] == 2


def test_logrank_needs_two_groups():
    with pytest.raises(ValueError):
        logrank_test([1, 2], [1, 1], ["a", "a"])


def test_prepare_endpoint_censors_at_the_horizon():
    clinical = pd.DataFrame({"OS": [1, 1, 0], "OS.time": [100, 5000, 200]},
                            index=["a", "b", "c"])
    surv = prepare_endpoint(clinical, "OS", max_followup=1000)
    assert surv.loc["b", "event"] == 0
    assert surv.loc["b", "time"] == 1000
    assert surv.loc["a", "event"] == 1


def test_prepare_endpoint_drops_missing_and_non_positive_times():
    clinical = pd.DataFrame({"OS": [1, np.nan, 1], "OS.time": [100, 200, 0]},
                            index=["a", "b", "c"])
    assert list(prepare_endpoint(clinical, "OS").index) == ["a"]


def test_unknown_endpoint_is_rejected():
    with pytest.raises(ValueError):
        prepare_endpoint(pd.DataFrame({"OS": [1], "OS.time": [1]}), "RFS")


def test_design_matrix_one_hot_encodes_against_a_reference():
    meta = pd.DataFrame({"age": [50, 60, 70], "stage": ["I", "II", "III"]})
    X = design_matrix(meta, ["age", "stage"], reference={"stage": "I"})
    assert list(X.columns) == ["age", "stage[II]", "stage[III]"]
    assert X.loc[1, "stage[II]"] == 1.0


def test_design_matrix_drops_a_column_with_no_variation():
    meta = pd.DataFrame({"age": [50, 60], "site": ["x", "x"]})
    assert list(design_matrix(meta, ["age", "site"]).columns) == ["age"]


def test_design_matrix_needs_at_least_one_usable_column():
    with pytest.raises(ValueError):
        design_matrix(pd.DataFrame({"a": [1, 2]}), [])


def test_fit_module_models_finds_a_harmful_module():
    rng = np.random.default_rng(9)
    n = 500
    activity = pd.DataFrame({"risky": rng.normal(0, 1, n), "inert": rng.normal(0, 1, n)},
                            index=[f"P{i}" for i in range(n)])
    event_time = rng.exponential(1.0 / np.exp(0.8 * activity["risky"]))
    censor = rng.exponential(1.5, n)
    surv = pd.DataFrame({"time": np.minimum(event_time, censor),
                         "event": (event_time <= censor).astype(int)},
                        index=activity.index)
    result = fit_module_models(activity, surv).set_index("module")
    assert result.loc["risky", "hr_uni"] > 1.3
    assert result.loc["risky", "q_uni"] < 0.01
    assert result.loc["inert", "p_uni"] > 0.05


def test_fit_module_models_adjusts_for_covariates():
    rng = np.random.default_rng(10)
    n = 400
    score = rng.normal(0, 1, n)
    index = [f"P{i}" for i in range(n)]
    activity = pd.DataFrame({"M1": score}, index=index)
    event_time = rng.exponential(1.0 / np.exp(0.6 * score))
    surv = pd.DataFrame({"time": event_time, "event": np.ones(n, dtype=int)}, index=index)
    covariates = pd.DataFrame({"age": rng.normal(60, 10, n)}, index=index)
    result = fit_module_models(activity, surv, covariates)
    assert {"hr_adj", "q_adj_model"}.issubset(result.columns)
    assert result.loc[0, "hr_adj"] > 1.2


def test_median_split_balances_the_arms():
    score = pd.Series(np.arange(10.0), index=[f"P{i}" for i in range(10)])
    arms = split_by_median(score)
    assert (arms == "high").sum() == 5 and (arms == "low").sum() == 5


def test_benjamini_hochberg_on_known_values():
    assert np.allclose(bh_fdr(np.array([0.001, 0.01, 0.03, 0.5])),
                       [0.004, 0.02, 0.04, 0.5])


def test_benjamini_hochberg_is_monotone_and_bounded():
    rng = np.random.default_rng(1)
    p = rng.uniform(0, 1, 200)
    q = bh_fdr(p)
    assert (q >= p - 1e-12).all() and q.max() <= 1.0
    order = np.argsort(p)
    assert np.all(np.diff(q[order]) >= -1e-12)


def test_benjamini_hochberg_of_an_empty_input():
    assert bh_fdr(np.array([])).size == 0


def test_empirical_p_is_never_zero():
    p = empirical_p(np.array([100.0]), np.array([0.0, 1.0, 2.0]))
    assert p[0] == pytest.approx(0.25)
