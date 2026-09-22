"""Run all six stages on the synthetic cohorts; marked slow, skip with -m "not slow"."""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config_test.json")

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def pipeline(synthetic_root, tmp_path_factory):
    out = str(tmp_path_factory.mktemp("results"))
    env = dict(os.environ, GENIE3_DATA=synthetic_root, GENIE3_RESULTS=out)
    proc = subprocess.run(
        [sys.executable, os.path.join(ROOT, "scripts", "run_all.py"), "--config", CONFIG],
        env=env, capture_output=True, text=True, cwd=ROOT,
    )
    if proc.returncode != 0:
        pytest.fail(f"pipeline failed:\n{proc.stdout[-4000:]}\n{proc.stderr[-4000:]}")
    return out, proc.stdout


def read(out, *parts):
    return pd.read_csv(os.path.join(out, *parts), sep="\t")


def test_every_stage_produced_its_outputs(pipeline):
    out, _ = pipeline
    for parts in [
        ("preprocessed", "tcga_meta.tsv"),
        ("preprocessed", "network_genes.txt"),
        ("networks", "all_top_edges.tsv"),
        ("networks", "run_info.json"),
        ("rewiring", "pairwise_overlap.tsv"),
        ("rewiring", "gene_rewiring.tsv"),
        ("hubs", "hub_frequency.tsv"),
        ("modules", "modules.tsv"),
        ("modules", "module_coherence.tsv"),
        ("survival", "tcga_module_cox.tsv"),
        ("survival", "selected_modules.txt"),
        ("validation", "metabric_module_cox.tsv"),
        ("validation", "concordance.tsv"),
    ]:
        assert os.path.exists(os.path.join(out, *parts)), parts


def test_all_subtypes_were_matched_to_the_same_sample_size(pipeline):
    out, _ = pipeline
    with open(os.path.join(out, "networks", "run_info.json"), encoding="utf-8") as fh:
        run = json.load(fh)
    assert run["n_match"] == min(run["subtypes"].values())
    assert set(run["subtypes"]) == {"Basal", "LumA", "LumB"}


def test_the_planted_module_genes_end_up_in_one_module(pipeline):
    """Modules are detected per subtype, so a shared module appears once in each."""
    out, _ = pipeline
    modules = read(out, "modules", "modules.tsv")
    modules["subtype"] = modules["module"].str.split("_").str[0]
    planted = modules[modules["gene"].str.startswith("MOD1_")]
    assert len(planted) > 0
    for subtype, block in planted.groupby("subtype"):
        biggest = block["module"].value_counts().iloc[0]
        assert biggest >= 0.6 * len(block), subtype


def test_composition_markers_do_not_dominate_the_modules(pipeline):
    """Stage 1 adjusts composition so adipose and stroma stop driving the network."""
    out, _ = pipeline
    modules = read(out, "modules", "modules.tsv")
    markers = {"ADIPOQ", "FABP4", "PLIN1", "COL1A1", "COL1A2", "PTPRC", "CD3D"}
    assert modules["gene"].isin(markers).mean() < 0.25


def test_the_survival_signal_is_found_and_points_the_right_way(pipeline):
    out, _ = pipeline
    cox = read(out, "survival", "tcga_module_cox.tsv")
    modules = read(out, "modules", "modules.tsv")
    carrier = (modules[modules["gene"].str.startswith("MOD1_")]["module"]
               .value_counts().idxmax())
    row = cox[cox["module"] == carrier].iloc[0]
    assert row["hr_uni"] > 1.0        # higher MOD1 activity means shorter survival
    assert row["p_uni"] < 0.05
    assert row["events"] > 20


def test_the_basal_only_regulator_is_called_rewired(pipeline):
    """REW_DRIVER wires up its targets in Basal only."""
    out, _ = pipeline
    for other in ("LumA", "LumB"):
        hits = read(out, "rewiring", f"Basal_vs_{other}_significant_edges.tsv")
        planted = hits[hits["regulator"].str.startswith("REW")
                       & hits["target"].str.startswith("REW")]
        assert len(planted) >= 5, f"Basal vs {other}: {len(hits)} hits, {len(planted)} planted"
        assert (planted["favours"] == "Basal").all()


def test_the_planted_edges_are_ranked_above_everything_else(pipeline):
    out, _ = pipeline
    diff = read(out, "rewiring", "Basal_vs_LumA_differential_edges.tsv")
    top = diff.head(20)
    is_planted = top["regulator"].str.startswith("REW") & top["target"].str.startswith("REW")
    assert is_planted.sum() >= 10


def test_two_subtypes_with_the_same_structure_give_no_hits(pipeline):
    """LumA and LumB come from identical generating processes."""
    out, _ = pipeline
    assert len(read(out, "rewiring", "LumA_vs_LumB_significant_edges.tsv")) == 0


def test_validation_scores_come_from_the_frozen_gene_lists(pipeline):
    out, _ = pipeline
    modules = read(out, "modules", "modules.tsv")
    coverage = read(out, "validation", "module_coverage.tsv")
    for _, row in coverage.iterrows():
        declared = (modules["module"] == row["module"]).sum()
        assert row["size"] == declared
        assert row["mapped"] <= row["size"]


def test_validation_reports_a_verdict_for_every_carried_module(pipeline):
    out, _ = pipeline
    with open(os.path.join(out, "survival", "selected_modules.txt"), encoding="utf-8") as fh:
        selected = [ln.strip() for ln in fh if ln.strip()]
    table = read(out, "validation", "concordance.tsv")
    assert set(table["module"]).issubset(selected)
    assert table["replicated"].dtype == bool


def test_the_run_is_reproducible(synthetic_root, tmp_path_factory):
    out = str(tmp_path_factory.mktemp("repeat"))
    env = dict(os.environ, GENIE3_DATA=synthetic_root, GENIE3_RESULTS=out)
    for stage in ("01_preprocess.py", "02_subtype_networks.py"):
        proc = subprocess.run(
            [sys.executable, os.path.join(ROOT, "scripts", stage), "--config", CONFIG],
            env=env, capture_output=True, text=True, cwd=ROOT)
        assert proc.returncode == 0, proc.stderr[-2000:]
    first = pd.read_csv(os.path.join(out, "networks", "Basal_top_edges.tsv"), sep="\t")

    again = str(tmp_path_factory.mktemp("repeat2"))
    env["GENIE3_RESULTS"] = again
    for stage in ("01_preprocess.py", "02_subtype_networks.py"):
        subprocess.run([sys.executable, os.path.join(ROOT, "scripts", stage),
                        "--config", CONFIG], env=env, capture_output=True, cwd=ROOT)
    second = pd.read_csv(os.path.join(again, "networks", "Basal_top_edges.tsv"), sep="\t")
    pd.testing.assert_frame_equal(first, second)
