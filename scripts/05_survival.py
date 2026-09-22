"""Stage 5: test module activity against outcome in TCGA, the discovery cohort.

    python scripts/05_survival.py
    python scripts/05_survival.py --endpoint DSS --adjusted
"""

from __future__ import annotations

import pandas as pd

from _common import base_parser, get_config, log

from src.io_utils import read_matrix, read_table, results_path, write_gene_list, write_table
from src.modules import activity_table, load_modules
from src.survival import (
    design_matrix, fit_module_models, kaplan_meier, logrank_test, median_survival,
    prepare_endpoint, split_by_median,
)


def main() -> None:
    ap = base_parser(__doc__)
    ap.add_argument("--tag", default="")
    ap.add_argument("--endpoint", default=None, choices=["OS", "DSS", "DFI", "PFI"])
    ap.add_argument("--adjusted", action="store_true",
                    help="score modules on the composition-adjusted matrix")
    args = ap.parse_args()
    cfg = get_config(args)
    endpoint = args.endpoint or cfg.survival_endpoint

    matrix = "tcga_expression.tsv.gz" if args.adjusted else "tcga_expression_raw.tsv.gz"
    expr = read_matrix(results_path("preprocessed", matrix))
    meta = read_table(results_path("preprocessed", "tcga_meta.tsv"), index_col=0)
    modules = load_modules(results_path("modules", f"modules{args.tag}.tsv"))
    log(f"{len(modules)} modules, endpoint {endpoint}, matrix {matrix}")

    activity = activity_table(expr, modules, method="rank")
    write_table(activity, results_path("survival", f"tcga_module_activity{args.tag}.tsv"))

    surv = prepare_endpoint(meta, endpoint, cfg.max_followup_days)
    covariates = design_matrix(meta.loc[surv.index], cfg.covariates).dropna()
    log(f"{len(surv)} patients with follow-up, {int(surv['event'].sum())} events, "
        f"{covariates.shape[1]} covariate columns")

    cox = fit_module_models(activity, surv, covariates)
    cox["endpoint"] = endpoint
    write_table(cox, results_path("survival", f"tcga_module_cox{args.tag}.tsv"), index=False)
    log("\n" + cox[["module", "n", "events", "hr_uni", "p_uni", "q_uni"]].head(10).to_string(index=False))

    km_rows, logrank_rows = [], []
    for module in cox["module"]:
        score = activity[module].reindex(surv.index).dropna()
        groups = split_by_median(score)
        frame = surv.loc[score.index]
        test = logrank_test(frame["time"], frame["event"], groups)
        logrank_rows.append({"module": module, "statistic": test["statistic"],
                             "p_value": test["p_value"]})
        for arm in ("high", "low"):
            take = frame.loc[groups[groups == arm].index]
            km = kaplan_meier(take["time"], take["event"])
            km["module"], km["arm"] = module, arm
            km_rows.append(km)
            if module == cox["module"].iloc[0]:
                log(f"{module} {arm}: n={len(take)}, median {median_survival(km):.0f} days")

    write_table(pd.concat(km_rows, ignore_index=True),
                results_path("survival", f"tcga_km{args.tag}.tsv"), index=False)
    write_table(pd.DataFrame(logrank_rows),
                results_path("survival", f"tcga_logrank{args.tag}.tsv"), index=False)

    selected = cox.loc[cox["q_uni"] <= cfg.fdr, "module"].tolist()
    write_gene_list(selected or cox["module"].head(3).tolist(),
                    results_path("survival", f"selected_modules{args.tag}.txt"))
    log(f"{len(selected)} modules pass FDR {cfg.fdr}; these go to METABRIC unchanged")


if __name__ == "__main__":
    main()
