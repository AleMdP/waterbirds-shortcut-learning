import os
import json
import matplotlib.pyplot as plt
import seaborn as sns

def plot_curves(history_path, title, output_filename):
    if not os.path.exists(history_path):
        print(f"Skipping {title}: {history_path} not found.")
        return

    with open(history_path, "r") as f:
        history = json.load(f)

    epochs = [entry["epoch"] for entry in history]
    
    group_labels = {
        0: "Group 0: Landbird / Land (Majority)",
        1: "Group 1: Landbird / Water (Minority)",
        2: "Group 2: Waterbird / Land (Minority)",
        3: "Group 3: Waterbird / Water (Majority)"
    }
    
    colors = ["#2b5c8f", "#d95f02", "#e7298a", "#1b9e77"]

    plt.figure(figsize=(9, 6))
    sns.set_theme(style="whitegrid")

    for g in range(4):
        losses = [entry[f"group_{g}_loss"] for entry in history]
        plt.plot(epochs, losses, marker="o", linewidth=2.5, label=group_labels[g], color=colors[g])

    plt.title(f"Per-Group Loss Convergence ({title})", fontsize=14, fontweight="bold")
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Cross-Entropy Loss", fontsize=12)
    plt.legend(loc="upper right")
    plt.tight_layout()

    os.makedirs("reports/figures", exist_ok=True)
    out_path = os.path.join("reports/figures", output_filename)
    plt.savefig(out_path, dpi=300)
    print(f"Saved figure to: {out_path}")

def main():
    plot_curves("results/erm_history.json", "Standard ERM", "erm_per_group_loss_curves.png")
    plot_curves("results/reweighted_history.json", "Group-Reweighted", "reweighted_per_group_loss_curves.png")

if __name__ == "__main__":
    main()