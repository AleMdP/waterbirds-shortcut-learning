"""Bar plots.
 
    python scripts/plot_results.py --model resnet50
    python scripts/plot_results.py --model transformer
 
1) <tag>_ablation_comparison.png : 3 methods x 2 samplers (avg vs worst-group, test).
2) group_dro_sweep_top5_<tag>.png: 5 Group DRO sweep runs with the best VALIDATION worst-group
   accuracy (selection never looks at the test set).
"""

import os
import json
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from result_utils import (CONFIGS, MODELS, MODEL_LABEL, MODEL_TAG, results_dir, sweep_dir, load_runs, rank_by_validation)

NAMES = {
    "erm_unbalanced": "ERM (Unbalanced)", "erm_balanced": "ERM (Balanced)",
    "reweight_unbalanced": "Reweight (Unbalanced)", "reweight_balanced": "Reweight (Balanced)",
    "group_dro_unbalanced": "Group DRO (Unbalanced)", "group_dro_balanced": "Group DRO (Balanced)",
}
 
 
def annotate(ax):
    for p in ax.patches:
        h = p.get_height()
        if h > 0:
            ax.annotate(f"{h:.1f}%", (p.get_x() + p.get_width() / 2., h), ha="center", va="bottom",
                        fontsize=8, fontweight="bold", xytext=(0, 3), textcoords="offset points")
 
 
def plot_ablation(model):
    rows = []
    for cfg in CONFIGS:
        p = os.path.join(results_dir(model), f"{cfg}_results.json")
        if not os.path.exists(p):
            print(f"[warning] {p} not found")
            continue
        m = json.load(open(p))
        rows.append({"Method": NAMES[cfg], "Metric": "Average Accuracy", "Accuracy": m["avg_acc"] * 100})
        rows.append({"Method": NAMES[cfg], "Metric": "Worst-Group Accuracy", "Accuracy": m["worst_group_acc"] * 100})
    if not rows:
        return
    plt.figure(figsize=(12, 6))
    sns.set_theme(style="whitegrid")
    ax = sns.barplot(data=pd.DataFrame(rows), x="Method", y="Accuracy", hue="Metric", palette=["#2b5c8f", "#d95f02"])
    plt.title(f"{MODEL_LABEL[model]}: Sampler & Method Mitigation Comparison (5 Epochs)", fontsize=14, fontweight="bold")
    plt.ylabel("Accuracy (%)", fontsize=12)
    plt.ylim(0, 105)
    plt.xticks(rotation=20, ha="right")
    annotate(ax)
    plt.tight_layout()
    out = f"reports/figures/{MODEL_TAG[model]}_ablation_comparison.png"
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"Saved {out}")
 
 
def plot_sweep_top5(model):
    runs = rank_by_validation(load_runs(sweep_dir(model)))[:5]
    if not runs:
        print(f"[warning] no sweep runs in {sweep_dir(model)}")
        return
    rows = []
    for r in runs:
        label = r["name"].replace("dro_", "")
        rows.append({"Config": label, "Metric": "Average Accuracy", "Accuracy": r["test_avg"] * 100})
        rows.append({"Config": label, "Metric": "Worst-Group Accuracy", "Accuracy": r["test_worst"] * 100})
    plt.figure(figsize=(11, 6))
    sns.set_theme(style="whitegrid")
    ax = sns.barplot(data=pd.DataFrame(rows), x="Config", y="Accuracy", hue="Metric", palette=["#2166ac", "#b2182b"])
    plt.title(f"{MODEL_LABEL[model]}: top-5 Group DRO configs by validation worst-group accuracy (test results)",
              fontsize=13, fontweight="bold")
    plt.ylabel("Accuracy (%)", fontsize=12)
    plt.ylim(0, 105)
    plt.xticks(rotation=20, ha="right")
    annotate(ax)
    plt.tight_layout()
    out = f"reports/figures/group_dro_sweep_top5_{MODEL_TAG[model]}.png"
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"Saved {out}")
 
 
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS, required=True)
    args = parser.parse_args()
    os.makedirs("reports/figures", exist_ok=True)
    plot_ablation(args.model)
    plot_sweep_top5(args.model)
 
 
if __name__ == "__main__":
    main()