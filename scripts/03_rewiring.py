"""Stage 3: call the edges that differ between subtypes against a permutation null.

    python scripts/03_rewiring.py
    python scripts/03_rewiring.py --pairs Basal:LumA
"""

from __future__ import annotations

import json
from itertools import combinations

import pandas as pd

from _common import base_parser, get_config, log

from src.io_utils import read_matrix, read_table, results_path, write_table
from src.preprocessing import split_by_group
from src.rewiring import (
    differential_edges, gene_rewiring, permutation_null, rewiring_summary, significant_edges,
)
from src.signatures import load_tfs


def main() -> None:
    ap = base_parser(__doc__)
    ap.add_argument("--tag", default="")
    ap.add_argument("--pairs", nargs="*", default=None,
                    help="restrict to specific comparisons, e.g. Basal:LumA")
    args = ap.parse_args()
    cfg = get_config(args)

    with open(results_path("networks", f"run_info{args.tag}.json"), encoding="utf-8") as fh:
        run = json.load(fh)

    expr = read_matrix(results_path("preprocessed", "tcga_expression.tsv.gz"))
    meta = read_table(results_path("preprocessed", "tcga_meta.tsv"), index_col=0)
    groups = split_by_group(expr, meta["subtype"], cfg.subtypes, cfg.min_subtype_n)
    vims = {name: read_matrix(results_path("networks", f"{name}{args.tag}_vim.tsv.gz"))
            for name in groups}
    if len(vims) < 2:
        raise SystemExit("run scripts/02_subtype_networks.py first")
    log(f"loaded {len(vims)} networks: {sorted(vims)}")

    overlap = rewiring_summary(vims, cfg.top_edges)
    write_table(overlap, results_path("rewiring", "pairwise_overlap.tsv"), index=False)
    log("top-edge Jaccard:\n" +
        overlap[["group_a", "group_b", "shared", "jaccard"]].to_string(index=False))

    log(f"building the null: {cfg.n_permutations} permutations x 2 x {run['n_reps']} fits")
    null = permutation_null(
        groups, n_match=run["n_match"], n_reps=run["n_reps"], n_perm=cfg.n_permutations,
        seed=cfg.seed + 1, verbose=True,
        regulators=list(vims[sorted(vims)[0]].index),
        tree_method=run["tree_method"], n_trees=run["n_trees"],
        max_features=run["max_features"], n_jobs=cfg.n_jobs,
    )
    write_table(null.describe(), results_path("rewiring", "null_summary.tsv"))

    pairs = [tuple(p.split(":")) for p in args.pairs] if args.pairs else \
        list(combinations(sorted(vims), 2))

    tfs = set(load_tfs())
    rewiring_tables = []
    for a, b in pairs:
        diff = differential_edges(vims[a], vims[b], null, a, b, seed=cfg.seed)
        diff["regulator_is_tf"] = diff["regulator"].isin(tfs)
        hits = significant_edges(diff, cfg.fdr)
        write_table(diff.head(20000),
                    results_path("rewiring", f"{a}_vs_{b}_differential_edges.tsv"), index=False)
        write_table(hits, results_path("rewiring", f"{a}_vs_{b}_significant_edges.tsv"),
                    index=False)
        log(f"{a} vs {b}: {len(hits)} edges at FDR {cfg.fdr} "
            f"({hits['favours'].value_counts().to_dict() if len(hits) else 'none'})")

        genes = gene_rewiring(hits if len(hits) else diff.head(cfg.top_edges))
        genes["comparison"] = f"{a}_vs_{b}"
        genes["is_tf"] = [g in tfs for g in genes.index]
        rewiring_tables.append(genes.reset_index())

    write_table(pd.concat(rewiring_tables, ignore_index=True),
                results_path("rewiring", "gene_rewiring.tsv"), index=False)
    log("wrote results/rewiring/")


if __name__ == "__main__":
    main()
