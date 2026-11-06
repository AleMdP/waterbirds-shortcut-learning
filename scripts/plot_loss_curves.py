import os
import json
import matplotlib.pyplot as plt
import seaborn as sns

def find_matching_file(keyword):
    """Searches results/ recursively for any JSON file matching the keyword."""
    for root, _, files in os.walk("results"):
        for file in files:
            if file.endswith(".json") and keyword in file and "summary" not in file:
                return os.path.join(root, file)
    
    # Also check results_3_epochs if present
    if os.path.exists("results_3_epochs"):
        for root, _, files in os.walk("results_3_epochs"):
            for file in files:
                if file.endswith(".json") and keyword in file and "summary" not in file:
                    return os.path.join(root, file)
    return None

def plot_curves(keyword, title, output_filename):
    history_path = find_matching_file(keyword)
    
    if not history_path:
        print(f"Skipping {title}: No JSON file containing '{keyword}' found in results/.")
        return

    print(f"Loading {title} from: {history_path}")
    with open(history_path, "r") as f:
        data = json.load(f)

    # Handle both list of epoch histories or dict containing metrics/history
    if isinstance(data, dict) and "history" in data:
        history = data["history"]
    elif isinstance(data, list):
        history = data
    elif isinstance(data, dict) and any("group_0_loss" in k for k in data.keys()):
        # Single final evaluation result dictionary
        history = [data]
    else:
        print(f"Skipping {title}: Structure of {history_path} unrecognized.")
        return

    epochs = [entry.get("epoch", idx + 1) for idx, entry in enumerate(history)]
    
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
        losses = [entry.get(f"group_{g}_loss", 0.0) for entry in history]
        plt.plot(epochs, losses, marker="o", linewidth=2.5, label=group_labels[g], color=colors[g])

    plt.title(f"Per-Group Loss Convergence ({title})", fontsize=14, fontweight="bold")
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Cross-Entropy Loss", fontsize=12)
    plt.legend(loc="upper right")
    plt.tight_layout()

    os.makedirs("reports/figures", exist_ok=True)
    out_path = os.path.join("reports/figures", output_filename)
    plt.savefig(out_path, dpi=300)
    print(f"Successfully generated figure: {out_path}")

def main():
    plot_curves("erm", "Standard ERM", "erm_per_group_loss_curves.png")
    plot_curves("reweight", "Group-Reweighted", "reweighted_per_group_loss_curves.png")

if __name__ == "__main__":
    main()