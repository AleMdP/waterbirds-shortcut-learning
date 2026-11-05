import os
import json
import matplotlib.pyplot as plt
import seaborn as sns

def find_history_file(filename):
    """Busca el archivo de historial en la raíz o en subcarpetas comunes de results."""
    candidate_paths = [
        os.path.join("results", filename),
        os.path.join("results/vit", filename),
        os.path.join("results_3_epochs", filename),
        os.path.join("results_3_epochs/vit", filename),
        filename
    ]
    for path in candidate_paths:
        if os.path.exists(path):
            return path
    return None

def plot_curves(filename, title, output_filename):
    history_path = find_history_file(filename)
    
    if not history_path:
        print(f"Skipping {title}: '{filename}' not found in results directory or subdirectories.")
        return

    print(f"Loading {title} from: {history_path}")
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
    plot_curves("erm_history.json", "Standard ERM", "erm_per_group_loss_curves.png")
    plot_curves("reweighted_history.json", "Group-Reweighted", "reweighted_per_group_loss_curves.png")

if __name__ == "__main__":
    main()