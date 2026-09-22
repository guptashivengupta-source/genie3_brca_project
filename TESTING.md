# How to test this

Three levels, each self-contained. Level 1 needs no data and no downloads.

Commands are PowerShell. On bash, replace `$env:X="y"` with `X=y` on the same line as the
command, and `\` with `/` in paths.

## 0. Install

```powershell
cd genie3_brca_project
pip install -r requirements.txt
```

Python 3.9 or newer. Everything is numpy, pandas, scipy, scikit-learn, joblib, matplotlib
and pytest; there is no bioconductor dependency and nothing to compile.

## 1. Does the code work at all (5 minutes, no data)

```powershell
python -m pytest -m "not slow" -q
```

Expect `114 passed`. These test the statistics directly on small matrices with a known
answer, so a failure here is a real bug, not a data problem.

If you want to see what they check:

```powershell
python -m pytest -m "not slow" -v
```

The test names are written to be read as claims, for example
`test_matching_removes_the_advantage_of_the_larger_group` and
`test_cox_score_test_equals_the_logrank_for_a_binary_covariate`.

## 2. Does the whole pipeline work (15 minutes, still no data)

This builds a fake two-cohort dataset with a known answer and runs all six stages on it.

```powershell
python -m pytest -m slow -q
```

Expect `11 passed`. The planted structure is: three modules, one that only holds together
in Basal, a sparse Basal-only regulator, and one module that drives survival. The tests
assert that the pipeline finds exactly those and nothing else.

To watch it run instead of just seeing a pass:

```powershell
python scripts\make_synthetic.py --out data\synthetic
$env:GENIE3_DATA="data\synthetic"
$env:GENIE3_RESULTS="results_demo"
python scripts\run_all.py --config tests\config_test.json
```

Then look at `results_demo\`. Clear the variables afterwards with
`Remove-Item Env:GENIE3_DATA, Env:GENIE3_RESULTS` so later runs use the real data.

This run ends with `3/3 modules replicated in METABRIC`. That is the planted answer, not a
result: the fake validation cohort carries the same survival signal by construction. The
real data ends with `0/3`. Seeing both is the point, because it shows the validation step
can report either verdict.

Everything together:

```powershell
python -m pytest -q
```

Expect `125 passed`, about 28 minutes.

## 3. Reproduce the real analysis (about 3 hours)

```powershell
python scripts\download_data.py tcga
python scripts\01_preprocess.py
python scripts\download_data.py metabric --genes results\preprocessed\network_genes.txt
python scripts\run_all.py --config config.json
python scripts\make_figures.py
```

The TCGA download is 64 MB from UCSC Xena. The METABRIC fetch only pulls the 500 genes
stage 1 selected, so it is a few MB. Stage 3 and stage 4 are the slow ones, roughly 75 and
68 minutes.

Check you got the same numbers:

| Where | Expect |
|---|---|
| `results\rewiring\pairwise_overlap.tsv` | LumA vs LumB Jaccard 0.248, Basal vs LumA 0.082 |
| stage 3 log | Basal vs LumA 45 edges, Basal vs LumB 51, LumA vs LumB 0 |
| `results\hubs\hub_frequency.tsv` | SOX10 at frequency 1.00 in all three subtypes |
| `results\modules\module_coherence.tsv` | one module per subtype above 0.5 |
| `results\survival\tcga_module_cox.tsv` | three modules, HR 0.77 to 0.82, all q below 0.05 |
| `results\validation\concordance.tsv` | all three preserved, none replicated |

The run is deterministic for a fixed seed, so these should match exactly.

## Reading a failure

- **A test in `tests/test_*.py` fails** and it is not `test_pipeline.py`: a real bug in
  `src/`. The test name says which claim broke.
- **`test_pipeline.py` fails**: run the six stages by hand as in section 2 and read the
  stage logs; the failure message quotes the last 4,000 characters of pipeline output.
- **Stage 4 exits with "no module passed the coherence threshold"**: the bootstrap
  networks are fine and only the clustering cut was wrong. Rerun with
  `python scripts\04_stability_modules.py --reuse-coassignment`, which reclusters the saved
  matrices in seconds instead of repeating an hour of fitting, or lower
  `module_stability_threshold` in `config.json`.
- **A download fails**: `data\README.md` has the direct URLs for manual download.
- **A worker process is killed during a test**: something else is using the cores. Set
  `"n_jobs": 2` in the config you are using.

## What the tests are actually for

The suite exists because most of the failure modes in this kind of analysis are silent.
Four checks are the ones that would stop a wrong paper:

- `test_matching_removes_the_advantage_of_the_larger_group` — without sample-size matching,
  a bigger subtype gets sharper importances and the rewiring is an artefact of cohort size.
- `test_two_groups_from_the_same_process_give_no_hits` and
  `test_p_values_are_roughly_uniform_under_the_null` — these caught three successive
  versions of the permutation null that were anti-conservative. The first reported 23 % of
  edges at p below 0.05 under a true null.
- `test_module_activity_survives_a_monotone_rescaling`, with its counterpart showing the
  z-score version does not — this is what licenses comparing a TCGA module score with a
  METABRIC one.
- `test_cox_score_test_equals_the_logrank_for_a_binary_covariate` — a textbook identity
  that only holds if both the Cox fit and the log-rank test are implemented correctly.
