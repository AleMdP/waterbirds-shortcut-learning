"""Average vs worst-group accuracy of every finished run (both backbones).

Left panel : average accuracy on the (group-balanced) test set, as in the report tables.
Right panel: average accuracy with each group weighted by its share of the TRAINING set
             (convention of Sagawa et al., 2020) -> shows the real trade-off ERM vs robust methods.

    python scripts/plot_tradeoff.py
"""
import os
import glob
import json
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
from result_utils import CONFIGS, MODELS, MODEL_LABEL, results_dir, sweep_dir, weighted_avg

COLORS = {"erm": "#2b5c8f", "reweight": "#d95f02", "group_dro": "#1b9e77"}
MARKERS = {"resnet50": "o", "transformer": "^"}


def collect(model):
    main, sweep = [], []
    for cfg in CONFIGS:
        p = os.path.join(results_dir(model), f"{cfg}_results.json")
        if os.path.exists(p):
            main.append((cfg, json.load(open(p))))
    for p in sorted(glob.glob(os.path.join(sweep_dir(model), "dro_*_results.json"))):
        sweep.append(json.load(open(p)))
    return main, sweep


def draw(ax, xfun, data):
    for model in MODELS:
        main, sweep = data[model]
        if sweep:
            ax.scatter([xfun(m) for m in sweep], [m["worst_group_acc"] * 100 for m in sweep],
                       marker=MARKERS[model], s=28, color="#9a9a9a", alpha=0.8, zorder=2)
        for cfg, m in main:
            method, sampler = cfg.rsplit("_", 1)
            c = COLORS[method]
            ax.scatter(xfun(m), m["worst_group_acc"] * 100, marker=MARKERS[model], s=95,
                       facecolors=c if sampler == "balanced" else "none", edgecolors=c,
                       linewidths=1.8, zorder=3)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Worst-group test accuracy (%)", fontsize=11)


def main():
    data = {m: collect(m) for m in MODELS}
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6), sharey=True)

    draw(axes[0], lambda m: m["avg_acc"] * 100, data)
    axes[0].set_xlabel("Average test accuracy (%) - group-balanced test set", fontsize=11)
    axes[0].set_title("Plain average", fontsize=12, fontweight="bold")

    draw(axes[1], lambda m: weighted_avg(m) * 100, data)
    axes[1].set_xlabel("Average test accuracy (%) - weighted by training group frequencies", fontsize=11)
    axes[1].set_title("Training-distribution average", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("")

    handles = [Line2D([], [], marker="s", ls="", color=c, label=n) for n, c in
               [("ERM", COLORS["erm"]), ("Reweight", COLORS["reweight"]), ("Group DRO", COLORS["group_dro"])]]
    handles += [Line2D([], [], marker="o", ls="", markerfacecolor="k", color="k", label="Balanced sampler (filled)"),
                Line2D([], [], marker="o", ls="", markerfacecolor="none", color="k", label="Unbalanced (hollow)"),
                Line2D([], [], marker="o", ls="", color="k", label=MODEL_LABEL["resnet50"] + " (circle)"),
                Line2D([], [], marker="^", ls="", color="k", label=MODEL_LABEL["transformer"] + " (triangle)"),
                Line2D([], [], marker=".", ls="", color="#9a9a9a", label="Group DRO sweep runs")]
    axes[1].legend(handles=handles, loc="lower left", fontsize=8.5, frameon=True)

    fig.suptitle("Average vs worst-group accuracy (5 epochs, single run per point)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    os.makedirs("reports/figures", exist_ok=True)
    out = "reports/figures/tradeoff_avg_vs_worst.png"
    plt.savefig(out, dpi=300)
    print(f"Saved trade-off plot to: {out}")


if __name__ == "__main__":
    main()
