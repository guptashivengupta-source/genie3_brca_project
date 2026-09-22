"""Stage 1: build the TCGA-BRCA analysis matrix, subtype labels and survival table.

    python scripts/01_preprocess.py
    python scripts/01_preprocess.py --profile quick --no-adjust
"""

from __future__ import annotations

import pandas as pd

from _common import base_parser, get_config, log

from src.io_utils import data_path, read_matrix, read_table, results_path, write_gene_list, write_table
from src.preprocessing import (
    clean_symbols, composition_scores, drop_constant, filter_low_expression,
    residualize, select_variable_genes,
)

STAGE_COLLAPSE = {
    "Stage I": "I", "Stage IA": "I", "Stage IB": "I",
    "Stage II": "II", "Stage IIA": "II", "Stage IIB": "II",
    "Stage III": "III", "Stage IIIA": "III", "Stage IIIB": "III", "Stage IIIC": "III",
    "Stage IV": "IV",
}


def primary_tumours(expr: pd.DataFrame) -> pd.DataFrame:
    return expr.loc[:, [c for c in expr.columns if c.endswith("-01")]]


def build_meta(clinical: pd.DataFrame, survival: pd.DataFrame) -> pd.DataFrame:
    meta = pd.DataFrame(index=clinical.index)
    meta["subtype"] = clinical["PAM50Call_RNAseq"]
    meta["age"] = pd.to_numeric(clinical["age_at_initial_pathologic_diagnosis"], errors="coerce")
    meta["stage"] = clinical["pathologic_stage"].map(STAGE_COLLAPSE)
    meta["er"] = clinical.get("ER_Status_nature2012")
    meta["pr"] = clinical.get("PR_Status_nature2012")
    meta["her2"] = clinical.get("HER2_Final_Status_nature2012")
    for col in ["OS", "OS.time", "DSS", "DSS.time", "DFI", "DFI.time", "PFI", "PFI.time"]:
        meta[col] = survival[col].reindex(meta.index)
    return meta


def main() -> None:
    ap = base_parser(__doc__)
    ap.add_argument("--no-adjust", action="store_true",
                    help="skip the composition adjustment (sensitivity analysis)")
    ap.add_argument("--n-genes", type=int)
    args = ap.parse_args()
    cfg = get_config(args)
    if args.n_genes:
        cfg.n_network_genes = args.n_genes
    adjust = cfg.adjust_composition and not args.no_adjust

    log("reading TCGA-BRCA")
    expr = primary_tumours(read_matrix(data_path("tcga", "HiSeqV2.gz")))
    clinical = read_table(data_path("tcga", "BRCA_clinicalMatrix.tsv"), index_col=0)
    survival = read_table(data_path("tcga", "BRCA_survival.tsv"), index_col=0)
    log(f"expression {expr.shape[0]} genes x {expr.shape[1]} primary tumours")

    meta = build_meta(clinical, survival)
    keep = [s for s in expr.columns if s in meta.index and pd.notna(meta.loc[s, "subtype"])]
    expr, meta = expr.loc[:, keep], meta.loc[keep]
    log(f"subtype-labelled samples: {dict(meta['subtype'].value_counts())}")

    expr = drop_constant(filter_low_expression(clean_symbols(expr), cfg.min_count, cfg.min_samples))
    log(f"{expr.shape[0]} genes pass the expression filter")

    composition = composition_scores(expr)
    write_table(composition, results_path("preprocessed", "composition.tsv"))

    if adjust:
        adjusted = residualize(expr, composition[cfg.composition_axes])
        log(f"regressed out {cfg.composition_axes}")
    else:
        adjusted = expr
        log("composition adjustment disabled")

    genes = select_variable_genes(adjusted, cfg.n_network_genes)
    genes_raw = select_variable_genes(expr, cfg.n_network_genes)
    log(f"{len(genes)} network genes, {len(set(genes) & set(genes_raw))} shared with the unadjusted list")

    write_table(adjusted.loc[genes], results_path("preprocessed", "tcga_expression.tsv.gz"))
    write_table(expr.loc[genes], results_path("preprocessed", "tcga_expression_raw.tsv.gz"))
    write_table(meta, results_path("preprocessed", "tcga_meta.tsv"))
    write_gene_list(genes, results_path("preprocessed", "network_genes.txt"))
    write_gene_list(genes_raw, results_path("preprocessed", "network_genes_unadjusted.txt"))
    log("wrote results/preprocessed/")


if __name__ == "__main__":
    main()
