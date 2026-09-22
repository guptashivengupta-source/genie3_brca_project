"""Draw the figures for the paper from whatever the pipeline has already written.

    python scripts/make_figures.py
"""

from __future__ import annotations

import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from _common import base_parser, get_config, log  # noqa: E402

from src.io_utils import read_table, results_path  # noqa: E402
from src.survival import kaplan_meier, logrank_test, prepare_endpoint, split_by_median  # noqa: E402

plt.rcParams.update({"figure.dpi": 150, "font.size": 8, "axes.spines.top": False,
                     "axes.spines.right": False})


def save(fig, name: str) -> str:
    path = results_path("figures", name)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    log(f"wrote {os.path.relpath(path, results_path())}")
    return path


def figure_overlap(tag: str) -> None:
    table = read_table(results_path("rewiring", "pairwise_overlap.tsv"))
    labels = [f"{a}\nvs {b}" for a, b in zip(table["group_a"], table["group_b"])]
    fig, ax = plt.subplots(figsize=(4, 2.6))
    ax.bar(labels, table["jaccard"], color="#4c72b0")
    ax.set_ylabel("Jaccard of top edges")
    ax.set_title("Top-edge overlap between subtype networks")
    save(fig, f"fig1_edge_overlap{tag}.png")


def figure_rewiring_volcano(tag: str) -> None:
    import glob

    paths = sorted(glob.glob(results_path("rewiring", "*_differential_edges.tsv")))
    if not paths:
        return
    fig, axes = plt.subplots(1, len(paths), figsize=(3.2 * len(paths), 2.8), squeeze=False)
    for ax, path in zip(axes[0], paths):
        diff = read_table(path)
        name = os.path.basename(path).replace("_differential_edges.tsv", "")
        ax.scatter(diff["delta"], -np.log10(diff["p_value"]), s=2, alpha=0.25,
                   color="#999999", rasterized=True)
        hits = diff[diff["q_value"] <= 0.05]
        ax.scatter(hits["delta"], -np.log10(hits["p_value"]), s=3, alpha=0.6, color="#c44e52")
        ax.set_xlabel("delta importance")
        ax.set_ylabel("-log10 p")
        ax.set_title(f"{name.replace('_vs_', ' vs ')}  ({len(hits)} at FDR 5%)")
    save(fig, f"fig2_rewiring_volcano{tag}.png")


def figure_hub_stability(tag: str) -> None:
    freq = read_table(results_path("hubs", f"hub_frequency{tag}.tsv"))
    subtypes = sorted(freq["subtype"].unique())
    fig, axes = plt.subplots(1, len(subtypes), figsize=(3.0 * len(subtypes), 3.0),
                             squeeze=False, sharex=True)
    for ax, name in zip(axes[0], subtypes):
        block = freq[freq["subtype"] == name].nlargest(15, "bootstrap_frequency")
        colours = ["#dd8452" if tf else "#4c72b0" for tf in block["is_tf"]]
        ax.barh(block["gene"][::-1], block["bootstrap_frequency"][::-1], color=colours[::-1])
        ax.set_xlim(0, 1)
        ax.set_title(name)
        ax.set_xlabel("bootstrap frequency")
    fig.suptitle("Most reproducible hubs (orange = transcription factor)", y=1.02)
    save(fig, f"fig3_hub_stability{tag}.png")


def figure_module_coherence(tag: str) -> None:
    table = read_table(results_path("modules", f"module_coherence{tag}.tsv"))
    fig, ax = plt.subplots(figsize=(4.2, 2.8))
    for name, block in table.groupby("subtype"):
        ax.scatter(block["size"], block["coherence"], label=name, s=18)
    ax.axhline(0.5, ls="--", lw=0.8, color="#888888")
    ax.set_xlabel("module size")
    ax.set_ylabel("bootstrap coherence")
    ax.set_title("Module size against reproducibility")
    ax.legend(frameon=False, fontsize=7)
    save(fig, f"fig4_module_coherence{tag}.png")


def figure_kaplan_meier(tag: str, cfg) -> None:
    cox = read_table(results_path("survival", f"tcga_module_cox{tag}.tsv"))
    if cox.empty:
        return
    activity = read_table(results_path("survival", f"tcga_module_activity{tag}.tsv"),
                          index_col=0)
    meta = read_table(results_path("preprocessed", "tcga_meta.tsv"), index_col=0)
    surv = prepare_endpoint(meta, cfg.survival_endpoint, cfg.max_followup_days)

    modules = list(cox["module"].head(3))
    fig, axes = plt.subplots(1, len(modules), figsize=(3.0 * len(modules), 2.8),
                             squeeze=False, sharey=True)
    for ax, module in zip(axes[0], modules):
        score = activity[module].reindex(surv.index).dropna()
        arms = split_by_median(score)
        frame = surv.loc[score.index]
        for arm, colour in (("high", "#c44e52"), ("low", "#4c72b0")):
            take = frame.loc[arms[arms == arm].index]
            km = kaplan_meier(take["time"], take["event"])
            ax.step([0] + list(km["time"]), [1.0] + list(km["survival"]), where="post",
                    color=colour, label=f"{arm} (n={len(take)})")
        p = logrank_test(frame["time"], frame["event"], arms)["p_value"]
        ax.set_title(f"{module}\nlog-rank p = {p:.3g}")
        ax.set_xlabel(f"days ({cfg.survival_endpoint})")
        ax.legend(frameon=False, fontsize=7)
    axes[0][0].set_ylabel("survival probability")
    save(fig, f"fig5_kaplan_meier{tag}.png")


def figure_validation(tag: str) -> None:
    path = results_path("validation", f"concordance{tag}.tsv")
    if not os.path.exists(path):
        return
    table = read_table(path, index_col=0)
    fig, ax = plt.subplots(figsize=(3.4, 3.2))
    colours = ["#55a868" if r else "#999999" for r in table["replicated"]]
    ax.scatter(table["hr_discovery"], table["hr_validation"], c=colours, s=30)
    for name, row in table.iterrows():
        ax.annotate(name, (row["hr_discovery"], row["hr_validation"]), fontsize=6,
                    xytext=(3, 3), textcoords="offset points")
    limits = [0.6, 1.6]
    ax.plot(limits, limits, ls="--", lw=0.8, color="#888888")
    ax.axhline(1.0, lw=0.5, color="#cccccc")
    ax.axvline(1.0, lw=0.5, color="#cccccc")
    ax.set_xlabel("hazard ratio, TCGA")
    ax.set_ylabel("hazard ratio, METABRIC")
    ax.set_title("Discovery against validation (green = replicated)")
    save(fig, f"fig6_validation{tag}.png")


def main() -> None:
    ap = base_parser(__doc__)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    cfg = get_config(args)

    for name, draw in (("overlap", figure_overlap), ("volcano", figure_rewiring_volcano),
                       ("hubs", figure_hub_stability), ("modules", figure_module_coherence),
                       ("validation", figure_validation)):
        try:
            draw(args.tag)
        except FileNotFoundError:
            log(f"skipped {name}: its stage has not run yet")
    try:
        figure_kaplan_meier(args.tag, cfg)
    except FileNotFoundError:
        log("skipped kaplan-meier: stage 5 has not run yet")


if __name__ == "__main__":
    main()
