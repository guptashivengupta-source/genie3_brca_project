"""Write a two-cohort dataset with a known answer, for the tests and for demos.

Planted: three modules (module 2 holds together in Basal only), a sparse Basal-only
regulator, and module 1 driving survival. The validation cohort uses a different scale.

    python scripts/make_synthetic.py --out data/synthetic
"""

from __future__ import annotations

import argparse
import gzip
import os

import numpy as np
import pandas as pd

from _common import ROOT  # noqa: F401  (puts the project on sys.path)

from src.signatures import COMPOSITION

SUBTYPES = {"LumA": 120, "LumB": 90, "Basal": 150}
MODULE_SIZE = 20
N_REWIRED_TARGETS = 5
N_NOISE = 160


def _module_block(n_samples: int, size: int, strength: float,
                  rng: np.random.Generator) -> np.ndarray:
    factor = rng.normal(0, 1, n_samples)
    loadings = rng.uniform(0.6, 1.0, size)
    noise = rng.normal(0, 1, (size, n_samples))
    return strength * np.outer(loadings, factor) + np.sqrt(1 - strength ** 2) * noise


def build_cohort(rng: np.random.Generator, subtypes: dict, basal_only_module: bool = True):
    labels = np.concatenate([[name] * n for name, n in subtypes.items()])
    n = len(labels)
    blocks, names = [], []

    for module, strength in ((1, 0.85), (2, 0.80), (3, 0.75)):
        rows = np.empty((MODULE_SIZE, n))
        for name in subtypes:
            take = labels == name
            # module 2 only coheres in Basal, which is the rewiring the pipeline should find
            weak = basal_only_module and module == 2 and name != "Basal"
            rows[:, take] = _module_block(int(take.sum()), MODULE_SIZE,
                                          0.15 if weak else strength, rng)
        blocks.append(rows)
        names += [f"MOD{module}_G{i:02d}" for i in range(MODULE_SIZE)]

    # sparse Basal-only regulator; dense blocks dilute per-edge importance too much to detect
    driver = rng.normal(0, 1.4, n)
    wired = labels == "Basal"
    rewired = np.empty((N_REWIRED_TARGETS + 1, n))
    rewired[0] = driver
    for i in range(1, N_REWIRED_TARGETS + 1):
        rewired[i, wired] = 1.9 * driver[wired] + 0.2 * rng.normal(0, 1, int(wired.sum()))
        rewired[i, ~wired] = rng.normal(0, 1.4, int((~wired).sum()))
    blocks.append(rewired)
    names += ["REW_DRIVER"] + [f"REW_T{i:02d}" for i in range(1, N_REWIRED_TARGETS + 1)]

    composition = rng.normal(0, 1, n)
    for axis, markers in COMPOSITION.items():
        rows = 0.9 * np.outer(np.ones(len(markers)), composition) + \
            rng.normal(0, 0.4, (len(markers), n))
        blocks.append(rows)
        names += list(markers)

    blocks.append(rng.normal(0, 1, (N_NOISE, n)) * rng.uniform(0.3, 1.2, (N_NOISE, 1)))
    names += [f"NOISE{i:03d}" for i in range(N_NOISE)]

    values = np.vstack(blocks) + 8.0
    expr = pd.DataFrame(values, index=names, columns=None)
    expr = expr.loc[~expr.index.duplicated()]
    return expr, pd.Series(labels)


def survival_from_module(expr: pd.DataFrame, rng: np.random.Generator, beta: float = 0.9):
    score = expr.loc[[g for g in expr.index if g.startswith("MOD1_")]].mean(axis=0)
    score = (score - score.mean()) / score.std()
    time = rng.exponential(1.0 / np.exp(beta * score.to_numpy())) * 1200.0
    censor = rng.exponential(2000.0, len(score))
    observed = np.minimum(time, censor)
    return observed.round(), (time <= censor).astype(int)


def write_discovery(out: str, seed: int) -> None:
    rng = np.random.default_rng(seed)
    expr, labels = build_cohort(rng, SUBTYPES)
    samples = [f"TCGA-XX-{i:04d}-01" for i in range(expr.shape[1])]
    expr.columns = samples
    labels.index = samples

    time, event = survival_from_module(expr, rng)
    os.makedirs(os.path.join(out, "tcga"), exist_ok=True)
    with gzip.open(os.path.join(out, "tcga", "HiSeqV2.gz"), "wt", encoding="utf-8") as fh:
        expr.rename_axis("sample").to_csv(fh, sep="\t")

    clinical = pd.DataFrame({
        "PAM50Call_RNAseq": labels,
        "age_at_initial_pathologic_diagnosis": rng.integers(32, 85, len(samples)),
        "pathologic_stage": rng.choice(["Stage I", "Stage IIA", "Stage IIIA"], len(samples)),
        "ER_Status_nature2012": rng.choice(["Positive", "Negative"], len(samples)),
        "PR_Status_nature2012": rng.choice(["Positive", "Negative"], len(samples)),
        "HER2_Final_Status_nature2012": rng.choice(["Positive", "Negative"], len(samples)),
    }, index=pd.Index(samples, name="sampleID"))
    clinical.to_csv(os.path.join(out, "tcga", "BRCA_clinicalMatrix.tsv"), sep="\t")

    survival = pd.DataFrame({"OS": event, "OS.time": time, "DSS": event, "DSS.time": time,
                             "DFI": event, "DFI.time": time, "PFI": event, "PFI.time": time},
                            index=pd.Index(samples, name="sample"))
    survival.to_csv(os.path.join(out, "tcga", "BRCA_survival.tsv"), sep="\t")


def write_validation(out: str, seed: int, drop_fraction: float = 0.1) -> None:
    rng = np.random.default_rng(seed + 1)
    sizes = {name: int(n * 1.4) for name, n in SUBTYPES.items()}
    expr, labels = build_cohort(rng, sizes)
    samples = [f"MB-{i:04d}" for i in range(expr.shape[1])]
    expr.columns = samples
    labels.index = samples

    time, event = survival_from_module(expr, rng)
    # microarray-like: different scale and a compressed dynamic range
    expr = np.sign(expr - 8.0) * np.abs(expr - 8.0) ** 0.7 * 1.6 + 6.0
    keep = rng.random(expr.shape[0]) > drop_fraction
    expr = expr.loc[keep]

    os.makedirs(os.path.join(out, "metabric"), exist_ok=True)
    expr.rename_axis("gene").to_csv(os.path.join(out, "metabric", "expression.tsv"), sep="\t")
    pd.DataFrame({
        "CLAUDIN_SUBTYPE": labels,
        "AGE_AT_DIAGNOSIS": rng.integers(30, 90, len(samples)),
        "GRADE": rng.choice(["1", "2", "3"], len(samples)),
        "OS": event, "OS.time": time, "DFI": event, "DFI.time": time,
    }, index=pd.Index(samples, name="sample")).to_csv(
        os.path.join(out, "metabric", "clinical.tsv"), sep="\t")


def make(out: str, seed: int = 11) -> str:
    write_discovery(out, seed)
    write_validation(out, seed)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "synthetic"))
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()
    print(f"wrote synthetic cohorts to {make(args.out, args.seed)}")


if __name__ == "__main__":
    main()
