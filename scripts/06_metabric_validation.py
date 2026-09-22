"""Stage 6: replay the frozen TCGA modules in METABRIC; nothing is refitted here.

    python scripts/06_metabric_validation.py
"""

from __future__ import annotations

import pandas as pd

from _common import base_parser, get_config, log

from src.io_utils import data_path, read_gene_list, read_matrix, read_table, results_path, write_table
from src.modules import load_modules
from src.survival import design_matrix, prepare_endpoint
from src.validation import (
    concordance, module_coverage, module_preservation, restrict_modules,
    transfer_activity, validate_survival,
)

COVARIATES = {"age": "AGE_AT_DIAGNOSIS", "grade": "GRADE", "subtype": "CLAUDIN_SUBTYPE"}


def build_meta(clinical: pd.DataFrame) -> pd.DataFrame:
    meta = pd.DataFrame(index=clinical.index)
    meta["age"] = pd.to_numeric(clinical[COVARIATES["age"]], errors="coerce")
    meta["grade"] = clinical[COVARIATES["grade"]].astype("string")
    meta["subtype"] = clinical[COVARIATES["subtype"]].astype("string")
    for col in ("OS", "OS.time", "DFI", "DFI.time"):
        meta[col] = pd.to_numeric(clinical[col], errors="coerce")
    return meta


def main() -> None:
    ap = base_parser(__doc__)
    ap.add_argument("--tag", default="")
    ap.add_argument("--endpoint", default=None, choices=["OS", "DFI"])
    ap.add_argument("--all-modules", action="store_true",
                    help="ignore the stage 5 shortlist and test every module")
    args = ap.parse_args()
    cfg = get_config(args)
    endpoint = args.endpoint or ("OS" if cfg.survival_endpoint in ("OS", "DSS") else "DFI")

    expr = read_matrix(data_path("metabric", "expression.tsv"))
    clinical = read_table(data_path("metabric", "clinical.tsv"), index_col=0)
    meta = build_meta(clinical)
    log(f"METABRIC {expr.shape[0]} genes x {expr.shape[1]} samples")

    modules = load_modules(results_path("modules", f"modules{args.tag}.tsv"))
    if not args.all_modules:
        shortlist = set(read_gene_list(results_path("survival", f"selected_modules{args.tag}.txt")))
        modules = {k: v for k, v in modules.items() if k in shortlist}
    log(f"carrying {len(modules)} modules over from TCGA")

    coverage = module_coverage(modules, expr)
    write_table(coverage, results_path("validation", f"module_coverage{args.tag}.tsv"))
    modules = restrict_modules(modules, expr, min_coverage=0.5, min_size=5)
    log(f"{len(modules)} modules keep at least half of their genes after mapping")
    if not modules:
        raise SystemExit("no module survived gene mapping")

    preservation = module_preservation(expr, modules, n_perm=max(cfg.n_permutations, 200),
                                       seed=cfg.seed)
    write_table(preservation, results_path("validation", f"module_preservation{args.tag}.tsv"))
    log("\n" + preservation[["size", "mean_abs_corr", "z_density", "p_value"]].to_string())

    activity = transfer_activity(expr, modules)
    write_table(activity, results_path("validation", f"metabric_module_activity{args.tag}.tsv"))

    surv = prepare_endpoint(meta, endpoint, cfg.max_followup_days)
    covariates = design_matrix(meta.loc[surv.index], ["age", "grade", "subtype"]).dropna()
    log(f"{len(surv)} patients, {int(surv['event'].sum())} {endpoint} events")

    result = validate_survival(activity, surv, covariates)
    result["endpoint"] = endpoint
    write_table(result, results_path("validation", f"metabric_module_cox{args.tag}.tsv"), index=False)
    log("\n" + result[["module", "n", "events", "hr_uni", "p_uni", "q_uni"]].to_string(index=False))

    discovery = read_table(results_path("survival", f"tcga_module_cox{args.tag}.tsv"))
    table = concordance(discovery, result, alpha=cfg.fdr)
    write_table(table, results_path("validation", f"concordance{args.tag}.tsv"))
    log("\n" + table.to_string())
    log(f"{int(table['replicated'].sum())}/{len(table)} modules replicated in METABRIC")


if __name__ == "__main__":
    main()
