"""Stage 4: bootstrap the subtype networks and keep the hubs and modules that survive.

    python scripts/04_stability_modules.py
    python scripts/04_stability_modules.py --profile quick
"""

from __future__ import annotations

from itertools import combinations

import pandas as pd

from _common import base_parser, get_config, log

from src.io_utils import read_matrix, read_table, results_path, write_table
from src.modules import (
    coassignment_matrix, consensus_modules, detect_modules, module_coherence,
    module_summary, save_modules,
)
from src.network import symmetric_adjacency
from src.preprocessing import split_by_group
from src.rewiring import jaccard
from src.signatures import load_tfs
from src.stability import bootstrap_networks, hub_frequency, partition_ari, stable_hubs


def main() -> None:
    ap = base_parser(__doc__)
    ap.add_argument("--tag", default="")
    ap.add_argument("--reuse-coassignment", action="store_true",
                    help="recluster the saved co-assignment matrices instead of "
                         "bootstrapping the networks again")
    args = ap.parse_args()
    cfg = get_config(args)
    tfs = set(load_tfs())

    expr = read_matrix(results_path("preprocessed", "tcga_expression.tsv.gz"))
    meta = read_table(results_path("preprocessed", "tcga_meta.tsv"), index_col=0)
    groups = split_by_group(expr, meta["subtype"], cfg.subtypes, cfg.min_subtype_n)
    n_match = cfg.n_match or min(g.shape[1] for g in groups.values())

    all_modules = {}
    hub_rows, coherence_rows, ari_rows = [], [], []

    for name, sub in sorted(groups.items()):
        consensus_file = results_path("modules", f"{name}{args.tag}_coassignment.tsv.gz")
        if args.reuse_coassignment:
            consensus = read_matrix(consensus_file)
            freq = read_table(results_path("hubs", f"{name}{args.tag}_hub_frequency.tsv"),
                              index_col=0)
            log(f"{name}: reusing the saved co-assignment matrix")
        else:
            log(f"{name}: {cfg.n_bootstrap} bootstrap networks at n={n_match}")
            nets = bootstrap_networks(
                sub, n_boot=cfg.n_bootstrap, seed=cfg.seed + 7 * len(name), size=n_match,
                tree_method=cfg.tree_method, n_trees=cfg.n_trees,
                max_features=cfg.max_features, n_jobs=cfg.n_jobs,
            )
            freq = hub_frequency(nets, cfg.top_edges, cfg.hub_top_k)
            freq["subtype"] = name
            freq["is_tf"] = [g in tfs for g in freq.index]
            freq["stable"] = freq["bootstrap_frequency"] >= cfg.hub_stability_threshold
            write_table(freq, results_path("hubs", f"{name}{args.tag}_hub_frequency.tsv"))

            partitions = [detect_modules(symmetric_adjacency(net), cfg.min_module_size,
                                         cfg.max_modules) for net in nets]
            consensus = coassignment_matrix(partitions, list(expr.index))
            write_table(consensus, consensus_file)

        hub_rows.append(freq.reset_index())
        log(f"{name}: {len(stable_hubs(freq, cfg.hub_stability_threshold))} stable hubs "
            f"(>= {cfg.hub_stability_threshold:.0%} of bootstraps)")

        modules = consensus_modules(consensus, cfg.min_module_size,
                                    cfg.module_stability_threshold)
        if not modules:
            log(f"{name}: no consensus module reached min_module_size")
            continue

        coherence = module_coherence(consensus, modules).to_frame()
        coherence["subtype"] = name
        coherence["size"] = [len(modules[m]) for m in coherence.index]
        coherence["keeper"] = coherence["coherence"] >= cfg.module_stability_threshold
        coherence.index.name = "module"
        coherence_rows.append(coherence.reset_index())

        if not args.reuse_coassignment:
            ari = partition_ari(modules, partitions)
            ari_rows.append({"subtype": name, "mean_ari": float(ari.mean()),
                             "sd_ari": float(ari.std(ddof=1)) if len(ari) > 1 else 0.0})
        log(f"{name}: {len(modules)} modules, "
            f"{int(coherence['keeper'].sum())} above coherence "
            f"{cfg.module_stability_threshold}, sizes {[len(g) for g in modules.values()]}")

        vim = read_matrix(results_path("networks", f"{name}{args.tag}_vim.tsv.gz"))
        summary = module_summary(modules, symmetric_adjacency(vim))
        summary["coherence"] = coherence["coherence"]
        summary["subtype"] = name
        write_table(summary, results_path("modules", f"{name}{args.tag}_module_summary.tsv"))

        for module in modules:
            if coherence.loc[module, "keeper"]:
                all_modules[f"{name}_{module}"] = modules[module]

    if not all_modules:
        raise SystemExit("no module passed the coherence threshold; lower it or add bootstraps")

    save_modules(all_modules, results_path("modules", f"modules{args.tag}.tsv"))
    write_table(pd.concat(hub_rows, ignore_index=True),
                results_path("hubs", f"hub_frequency{args.tag}.tsv"), index=False)
    write_table(pd.concat(coherence_rows, ignore_index=True),
                results_path("modules", f"module_coherence{args.tag}.tsv"), index=False)
    if ari_rows:
        write_table(pd.DataFrame(ari_rows),
                    results_path("modules", f"partition_ari{args.tag}.tsv"), index=False)

    overlap = [{"module_a": a, "module_b": b,
                "jaccard": jaccard(set(all_modules[a]), set(all_modules[b]))}
               for a, b in combinations(sorted(all_modules), 2)]
    write_table(pd.DataFrame(overlap), results_path("modules", f"module_overlap{args.tag}.tsv"),
                index=False)
    log(f"kept {len(all_modules)} modules across {len(groups)} subtypes")


if __name__ == "__main__":
    main()
