"""Per-group TRAINING loss per epoch, one figure per run.
 
    python scripts/plot_loss_curves.py --model resnet50
    python scripts/plot_loss_curves.py --model transformer
 
Output: reports/figures/<resnet50|vit>_<method>_<sampler>_loss_curves.png
Files are read from explicit paths (no fuzzy matching), so every figure is tied to one run.
"""

import os
import json
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from result_utils import CONFIGS, MODELS, MODEL_LABEL, MODEL_TAG, results_dir

GROUP_LABELS = {
    0: "Group 0: Landbird / Land (Majority)",
    1: "Group 1: Landbird / Water (Minority)",
    2: "Group 2: Waterbird / Land (Minority)",
    3: "Group 3: Waterbird / Water (Majority)",
}
COLORS = ["#2b5c8f", "#d95f02", "#e7298a", "#1b9e77"]
NICE = {"erm": "ERM", "reweight": "Group-reweighted", "group_dro": "Group DRO"}
 
 
def plot_curves(history_path, title, out_path):
    if not os.path.exists(history_path):
        print(f"[skip] {history_path} not found")
        return
    with open(history_path) as f:
        history = json.load(f)
    epochs = [e.get("epoch", i + 1) for i, e in enumerate(history)]
 
    plt.figure(figsize=(9, 6))
    sns.set_theme(style="whitegrid")
    for g in range(4):
        plt.plot(epochs, [e[f"group_{g}_loss"] for e in history], marker="o", linewidth=2.5,
                 label=GROUP_LABELS[g], color=COLORS[g])
    plt.title(f"Per-Group Training Loss ({title})", fontsize=14, fontweight="bold")
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Cross-Entropy Loss (unweighted, training mode)", fontsize=12)
    plt.xticks(epochs)
    plt.legend(loc="best")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved {out_path}")
 
 
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS, required=True)
    args = parser.parse_args()
    for cfg in CONFIGS:
        method, sampler = cfg.rsplit("_", 1)
        plot_curves(os.path.join(results_dir(args.model), f"{cfg}_history.json"),
                    f"{MODEL_LABEL[args.model]}, {NICE[method]}, {sampler} loader",
                    f"reports/figures/{MODEL_TAG[args.model]}_{cfg}_loss_curves.png")
 
 
if __name__ == "__main__":
    main()