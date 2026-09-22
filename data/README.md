# Data

Nothing here is committed. `scripts/download_data.py` fetches both cohorts; the links below
are the manual fallback.

## TCGA-BRCA (discovery) - UCSC Xena, TCGA hub

`python scripts/download_data.py tcga` writes three files into `data/tcga/`.

| File | Source |
|---|---|
| `HiSeqV2.gz` | https://tcga.xenahubs.net/download/TCGA.BRCA.sampleMap/HiSeqV2.gz |
| `BRCA_clinicalMatrix.tsv` | https://tcga.xenahubs.net/download/TCGA.BRCA.sampleMap/BRCA_clinicalMatrix |
| `BRCA_survival.tsv` | https://tcga.xenahubs.net/download/survival/BRCA_survival.txt |

`HiSeqV2` is 20,530 genes x 1,218 samples, already log2(norm_count+1), so stage 1 does not
log-transform again. PAM50 calls come from `PAM50Call_RNAseq` in the clinical matrix.
Survival is the Liu et al. 2018 curated endpoint file: OS, DSS, DFI and PFI.

The same study on cBioPortal (`brca_tcga_pan_can_atlas_2018`) works too, but its bulk
download was not reachable from here and its clinical file needs different column names in
`scripts/01_preprocess.py: build_meta`.

## METABRIC (validation) - cBioPortal REST API

`python scripts/download_data.py metabric --genes results/preprocessed/network_genes.txt`
writes `data/metabric/expression.tsv` and `clinical.tsv`. It fetches only the genes stage 1
selected, so run stage 1 first.

- expression: profile `brca_metabric_mrna`, sample list `brca_metabric_all`
- clinical: patient- and sample-level attributes, pivoted wide
- PAM50 is `CLAUDIN_SUBTYPE`; `claudin-low`, `Normal` and `NC` are kept in the file and
  dropped by the analysis
- `OS_MONTHS` / `OS_STATUS` and `RFS_MONTHS` / `RFS_STATUS` are converted to `OS`/`OS.time`
  and `DFI`/`DFI.time` in days, so both cohorts share the TCGA column names

Full study download (needs ~100 MB): https://www.cbioportal.org/study/summary?id=brca_metabric

## Synthetic cohorts

`python scripts/make_synthetic.py --out data/synthetic` writes both cohorts in the same
formats with a known answer. Point the pipeline at them with `GENIE3_DATA` and
`GENIE3_RESULTS`.
