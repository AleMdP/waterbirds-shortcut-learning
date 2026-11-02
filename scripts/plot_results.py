import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def main():
    os.makedirs("reports/figures", exist_ok=True)
    
    # Load JSON metrics
    erm_path = "results/erm_baseline_results.json"
    reweighted_path = "results/reweighted_results.json"
    
    if not os.path.exists(erm_path) or not os.path.exists(reweighted_path):
        raise FileNotFoundError("Missing result JSON files in results/ directory.")
        
    with open(erm_path, "r") as f:
        erm_data = json.load(f)
        
    with open(reweighted_path, "r") as f:
        reweighted_data = json.load(f)

    # Prepare DataFrame for plotting
    plot_data = [
        {"Method": "Standard ERM", "Metric": "Average Accuracy", "Accuracy": erm_data["avg_acc"] * 100},
        {"Method": "Standard ERM", "Metric": "Worst-Group Accuracy", "Accuracy": erm_data["worst_group_acc"] * 100},
        {"Method": "Group-Reweighted", "Metric": "Average Accuracy", "Accuracy": reweighted_data["avg_acc"] * 100},
        {"Method": "Group-Reweighted", "Metric": "Worst-Group Accuracy", "Accuracy": reweighted_data["worst_group_acc"] * 100},
    ]
    
    df = pd.DataFrame(plot_data)

    # Generate Bar Chart Comparison
    plt.figure(figsize=(8, 6))
    sns.set_theme(style="whitegrid")
    
    ax = sns.barplot(
        data=df, 
        x="Method", 
        y="Accuracy", 
        hue="Metric", 
        palette=["#2b5c8f", "#d95f02"]
    )
    
    plt.title("ERM vs. Group-Reweighted Performance Comparison", fontsize=14, fontweight="bold")
    plt.ylabel("Accuracy (%)", fontsize=12)
    plt.ylim(0, 100)
    
    # Add numerical labels above bars
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f"{height:.2f}%",
                        (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='bottom', fontsize=10, fontweight="bold",
                        xytext=(0, 3), textcoords='offset points')

    plt.tight_layout()
    output_path = "reports/figures/erm_vs_reweighted_comparison.png"
    plt.savefig(output_path, dpi=300)
    print(f"Comparison figure saved to {output_path}")

if __name__ == "__main__":
    main()