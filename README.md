# Subtype-specific regulatory networks in breast cancer

GENIE3 networks are inferred separately for each PAM50 subtype in TCGA-BRCA, compared
against a permutation null, reduced to modules that survive bootstrapping, tested against
survival, and then carried unchanged into METABRIC.

**TCGA-BRCA discovers; METABRIC decides.** Nothing is fitted, selected or re-thresholded
in METABRIC.

## What is different here

Running GENIE3 once per subtype and listing the top 1,000 edges is easy, and it mostly
measures sample size and noise. Four things in this pipeline exist to stop that.

1. **Every network is inferred at the same sample size.** Luminal A has 421 tumours and
   Basal 141; a network built on more samples has sharper importances, so an unmatched
   comparison finds "rewiring" that is really just power. Each subtype network is the mean
   of `n_match_reps` GENIE3 fits on subsamples of the size of the smallest retained
   subtype (`src/network.py: matched_ensemble`).

2. **Edges are called differential against a null that repeats the whole procedure.**
   The subtype labels are shuffled, the pooled patients are split in two, and the same
   averaged networks are built again `n_permutations` times. That is the only null that
   works: averaging replicates shrinks subsampling noise but not the noise of which
   patients a cohort happens to contain, and a cheaper null (permuting replicate labels,
   or splitting one subtype in half) is measurably anti-conservative. `tests/test_rewiring.py`
   checks the calibration directly.

3. **Adipose, stromal and immune content are regressed out first.** Unadjusted TCGA-BRCA
   networks are led by `ADIPOQ`, `FABP4`, `PLIN1`, `CIDEC` adipocyte genes that say more
   about how much fat was in the block than about tumour-cell regulation. Stage 1 removes
   three marker-based composition axes before gene selection; 56 of the top 500 variable
   genes change as a result. `--no-adjust` reruns without it as a sensitivity analysis.

4. **Modules come from bootstrap co-assignment, and their score is rank-based.** A module
   is a set of genes that keep clustering together across bootstrap networks, scored by how
   often they do (`coherence`). The co-assignment tree is cut at a co-assignment level, not
   at a fixed number of clusters, so module size follows the data and genes that never
   settle are left unassigned rather than forced somewhere. Module activity is the mean
   within-sample percentile rank of its genes, which is invariant to any monotone
   rescaling so a module defined on TCGA RNA-seq counts transfers to METABRIC microarray
   intensities without renormalization.

Comparison methods and parameter sweeps are in **Sensitivity analyses** below.

## Data

| | Discovery | Validation |
|---|---|---|
| Cohort | TCGA-BRCA | METABRIC |
| Source | UCSC Xena TCGA hub | cBioPortal REST API |
| Platform | Illumina HiSeq RNA-seq, log2(norm_count+1) | Illumina HT-12 v3 microarray |
| Samples | 1,097 primary tumours, 841 with a PAM50 call | 1,980 with expression |
| Subtypes | LumA 421, LumB 192, Basal 141, Her2 67, Normal 23 | LumA 700, LumB 475, Her2 224, Basal 209 |
| Outcome | OS / DSS / DFI / PFI, 202 OS events | OS and RFS, 1,143 OS events |

Two consequences worth stating up front. Her2 has only 67 TCGA tumours, which would cap
the matched sample size for every network, so the primary analysis keeps subtypes with at
least 100 samples (`min_subtype_n`) and Her2 is a secondary run via `config_her2.json`.
And METABRIC contributes far more events than TCGA, so validation is better powered than
discovery which is the right way round.

```bash
python scripts/download_data.py tcga
python scripts/01_preprocess.py
python scripts/download_data.py metabric --genes results/preprocessed/network_genes.txt
```

The METABRIC fetch pulls only the genes stage 1 selected (452 of 500 map), so it is a few
MB rather than a full study download. Links and manual fallbacks are in `data/README.md`.

## Running it

```bash
pip install -r requirements.txt
python scripts/run_all.py --profile quick     # ~10 min, every stage, small settings
python scripts/run_all.py --config config.json
```

Stages also run on their own, in order:

| Stage | Script | Output |
|---|---|---|
| 1 | `01_preprocess.py` | analysis matrix, subtype labels, survival, composition scores |
| 2 | `02_subtype_networks.py` | one matched GENIE3 network per subtype, top edges |
| 3 | `03_rewiring.py` | permutation null, differential edges, per-gene rewiring |
| 4 | `04_stability_modules.py` | bootstrap hub frequencies, co-assignment, consensus modules |
| 5 | `05_survival.py` | Kaplan-Meier, log-rank, Cox in TCGA |
| 6 | `06_metabric_validation.py` | preservation, Cox and replication in METABRIC |

Every stage takes `--config`, `--profile quick|full` and `--seed`, and every run is
deterministic for a fixed seed.

### Runtime

One GENIE3 network over 500 genes at n=141 takes about 45 s on 12 cores at
`n_trees=100`. With the shipped `config.json`:

| Stage | Fits | Time |
|---|---:|---:|
| 2 | 3 x 5 | ~11 min |
| 3 | 10 x 2 x 5 | ~75 min |
| 4 | 3 x 30 | ~68 min |

Raise `n_trees` to 200 and `n_bootstrap` to 100 for a final run and expect roughly four
times that. `n_jobs` parallelizes over targets; set it to 2 or 4 if you are running
something else on the machine.

## Layout

```
config.json          every parameter, and what a run is reproducible from
config_her2.json     secondary run that keeps Her2 at the cost of n_match = 67
src/                 the analysis code, no paths and no argument parsing
scripts/             the six numbered stages, plus download and synthetic-data helpers
notebooks/           thin walk-throughs of each stage, calling the same src functions
tests/               unit tests plus an end-to-end run on synthetic cohorts
data/                downloaded cohorts and the reference TF list
results/             everything a stage writes; safe to delete and regenerate
```

`src/` holds no file paths and no `argparse`; the stages hold no statistics. That split is
what lets the tests exercise the science without touching the filesystem.

## Outputs worth reading

- `results/rewiring/{A}_vs_{B}_significant_edges.tsv` rewired edges at FDR 0.05, with the
  subtype each favours, the observed delta and the mean null delta for edges of that
  strength.
- `results/hubs/hub_frequency.tsv` every gene's bootstrap frequency, mean degree, SD,
  whether it is a TF, and whether it clears `hub_stability_threshold`.
- `results/modules/module_coherence.tsv` module size and mean bootstrap co-assignment.
  Only modules above `module_stability_threshold` reach stage 5.
- `results/survival/tcga_module_cox.tsv` univariable and covariate-adjusted hazard
  ratios with CIs and BH-adjusted p-values.
- `results/validation/concordance.tsv` the one table that matters: discovery HR,
  validation HR, same direction, replicated.

## Tests

```bash
python -m pytest                    # 125 tests, ~28 min
python -m pytest -m "not slow"      # 114 tests, ~12 min
```

The suite is written as scenarios rather than coverage. The ones that would actually catch
a wrong result:

| Scenario | File |
|---|---|
| A planted regulator comes out as the top edge | `test_network.py` |
| The larger group loses its edge-strength advantage once matched | `test_network.py` |
| A subtype-specific edge is called rewired at FDR 0.05 | `test_rewiring.py` |
| Two cohorts from one process give zero hits and uniform p-values | `test_rewiring.py` |
| A planted single-subtype regulator tops the rewiring ranking end to end | `test_pipeline.py` |
| A reproducible hub is found in every bootstrap, noise genes in few | `test_stability.py` |
| Consensus modules recover planted blocks from noisy partitions | `test_modules.py` |
| Rank activity survives a monotone rescaling, z-score activity does not | `test_modules.py` |
| Cox recovers a known hazard ratio, and its score test equals the log-rank | `test_survival.py` |
| Validation never adds a gene the module did not already have | `test_validation.py` |
| A planted module is preserved in the validation cohort, a random set is not | `test_validation.py` |
| Six stages run end to end and find the planted survival signal | `test_pipeline.py` |

`test_pipeline.py` runs against `scripts/make_synthetic.py`, which writes a two-cohort
dataset in the real file formats with a known answer: three modules, one that only coheres
in Basal, one that drives survival, and a validation cohort on a different scale missing
10% of genes. It is also the fastest way to demo the pipeline without downloading anything.

## What the shipped configuration found

Full run on the real cohorts, `config.json`, seed 1234. Reproduce with
`python scripts/run_all.py --config config.json`.

**Rewiring.** The two luminal subtypes share three times as many top edges as either shares
with Basal, and after the permutation null no LumA-vs-LumB edge survives at all.

| Comparison | Top-edge Jaccard | Edges at FDR 5% |
|---|---:|---:|
| Basal vs LumA | 0.082 | 45 (32 favour LumA) |
| Basal vs LumB | 0.072 | 51 (35 favour LumB) |
| LumA vs LumB | 0.248 | 0 |

Forty-five edges is a small number, and it is small because the null is honest. The earlier
within-group null on this same data reported thousands.

**Hubs.** 9 to 12 genes per subtype appear in the top 20 in at least 70% of 30 bootstrap
networks. They are the basal-keratin programme `SOX10` (frequency 1.00 in all three
subtypes), `KRT5`, `KRT14`, `KRT6B`, `KRT17`, `DSG3`, `DSC3`, `S100A7`, `KLK5`, `KLK6`,
`CLEC3A`, `SCGB2A2`, `TFAP2B`. No adipocyte gene survives, which is the composition
adjustment doing its job.

**Modules.** One module per subtype clears coherence 0.5: Basal_M1 (14 genes, 0.65),
LumA_M1 (55, 0.69), LumB_M1 (31, 0.76). Everything else is below, and partition ARI across
bootstraps is 0.08 to 0.15 at 500 genes and n=141, module *boundaries* are not
reproducible even where a tight core is.

**Survival.** All three modules are protective in TCGA (810 patients, 104 OS events) and
all three clear FDR 5%: LumA_M1 HR 0.77 (p=0.009), LumB_M1 HR 0.78 (p=0.010), Basal_M1
HR 0.82 (p=0.041).

**Validation.** All three modules are strongly preserved in METABRIC mean within-module
correlation far above size-matched random gene sets (z = 16.7 to 18.8, p < 0.005). The
survival association does not replicate. LumA_M1 (HR 0.95, p=0.22) and LumB_M1 (HR 0.94,
p=0.12) point the same way but attenuate to nothing; Basal_M1 reverses.

That is a negative result and it should be reported as one. METABRIC contributes 728 events
against TCGA's 104, so this is not a power problem the prognostic signal in TCGA does not
carry. The co-expression structure transfers; the outcome association does not. Any version
of this analysis that had picked its modules or thresholds after looking at METABRIC would
have found something, which is exactly why stage 5 freezes the shortlist before stage 6
runs.

## Sensitivity analyses

| Question | Command |
|---|---|
| Does the composition adjustment change the conclusions? | `01_preprocess.py --no-adjust` |
| Are the edges plausible as regulation, not just prediction? | `02_subtype_networks.py --tf-only` |
| Random Forest or Extra-Trees? | `"tree_method": "ET"` in a config |
| Is it an artefact of the top-edge cutoff? | `"top_edges": 500 / 1000 / 2000` |
| Is it an artefact of the gene count? | `01_preprocess.py --n-genes 300` |
| Does Her2 behave like the others? | `--config config_her2.json` |
| A different endpoint? | `05_survival.py --endpoint DSS` |

Use `--tag` to keep parallel runs side by side in `results/`.

If stage 4 ends with "no module passed the coherence threshold", the bootstrap networks were
fine and only the cut was wrong. `04_stability_modules.py --reuse-coassignment` reclusters
the saved co-assignment matrices in seconds rather than repeating an hour of fitting.

## What this does not establish

**Edge-level tests are blind to dense modules.** When twenty genes share one regulatory
programme, GENIE3 splits each gene's importance across its nineteen partners, so every
individual edge carries a small share and the per-edge difference stays small even when the
programme is present in one subtype and absent in another. The synthetic cohort makes this
concrete: a dense Basal-only block of twenty genes produces no significant edges at all,
while a sparse Basal-only regulator with five targets is detected and ranks at the very top.
So read the rewiring tables as a map of sparse, strong regulatory differences, and use the
module and hub results for the dense programmes. A modest count of significant edges is not
evidence that the subtypes are regulatorily similar.

GENIE3 edges are predictive associations from expression alone. A rewired edge means the
regulator predicts the target differently in one subtype, not that it binds its promoter;
confirming that needs ChIP-seq, motif or perturbation evidence, which is not in this
pipeline. The composition adjustment uses marker scores, not ESTIMATE or a deconvolution,
so residual stromal signal is likely. The PAM50 calls are taken as given. And the survival
analysis is observational: treatment is not modelled in TCGA and only partly recorded in
METABRIC, so a hazard ratio here is an association with outcome, not an effect of the
module.
