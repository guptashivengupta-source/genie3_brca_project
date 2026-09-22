"""Stage 2: one GENIE3 network per PAM50 subtype, all inferred at the same sample size.

    python scripts/02_subtype_networks.py
    python scripts/02_subtype_networks.py --profile quick --tf-only
"""

from __future__ import annotations

import json

import pandas as pd

from _common import base_parser, get_config, log

from src.io_utils import read_matrix, read_table, results_path, write_table
from src.network import annotate_edges, link_list, matched_networks
from src.preprocessing import split_by_group
from src.signatures import load_tfs


def main() -> None:
    ap = base_parser(__doc__)
    ap.add_argument("--tf-only", action="store_true",
                    help="restrict candidate regulators to the human TF list")
    ap.add_argument("--raw", action="store_true",
                    help="use the composition-unadjusted matrix")
    ap.add_argument("--tag", default="", help="suffix for the output files")
    args = ap.parse_args()
    cfg = get_config(args)
    tf_only = cfg.tf_only or args.tf_only
    tag = args.tag or ("_tf" if tf_only else "")

    matrix = "tcga_expression_raw.tsv.gz" if args.raw else "tcga_expression.tsv.gz"
    expr = read_matrix(results_path("preprocessed", matrix))
    meta = read_table(results_path("preprocessed", "tcga_meta.tsv"), index_col=0)

    groups = split_by_group(expr, meta["subtype"], cfg.subtypes, cfg.min_subtype_n)
    if len(groups) < 2:
        raise SystemExit("fewer than two subtypes pass min_subtype_n")
    n_match = cfg.n_match or min(g.shape[1] for g in groups.values())
    log(f"{len(groups)} subtypes, matching every network at n={n_match}")

    tfs = load_tfs()
    regulators = [g for g in expr.index if g in set(tfs)] if tf_only else None
    if tf_only:
        log(f"{len(regulators)} of {expr.shape[0]} genes are candidate regulators")

    means = matched_networks(
        groups, n_match=n_match, n_reps=cfg.n_match_reps, seed=cfg.seed,
        verbose=True, regulators=regulators,
        tree_method=cfg.tree_method, n_trees=cfg.n_trees,
        max_features=cfg.max_features, n_jobs=cfg.n_jobs,
    )

    edge_tables = []
    for name, vim in means.items():
        write_table(vim, results_path("networks", f"{name}{tag}_vim.tsv.gz"))
        edges = annotate_edges(link_list(vim, top_n=cfg.top_edges), tfs, name)
        write_table(edges, results_path("networks", f"{name}{tag}_top_edges.tsv"), index=False)
        edge_tables.append(edges)
        log(f"{name}: top edge {edges.iloc[0]['regulator']} -> {edges.iloc[0]['target']}, "
            f"{edges['regulator_is_tf'].mean():.0%} TF regulators")

    write_table(pd.concat(edge_tables, ignore_index=True),
                results_path("networks", f"all_top_edges{tag}.tsv"), index=False)

    run = {"n_match": n_match, "n_reps": cfg.n_match_reps, "tf_only": tf_only,
           "raw_matrix": args.raw, "subtypes": {k: v.shape[1] for k, v in groups.items()},
           "genes": expr.shape[0], "adjusted": not args.raw,
           **{k: getattr(cfg, k) for k in
              ("tree_method", "n_trees", "max_features", "seed", "top_edges")}}
    with open(results_path("networks", f"run_info{tag}.json"), "w", encoding="utf-8") as fh:
        json.dump(run, fh, indent=2)
    log("wrote results/networks/")


if __name__ == "__main__":
    main()
