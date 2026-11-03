import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def main():
    os.makedirs("reports/figures", exist_ok=True)
    
    files = {
        "Standard ERM": "results/erm_baseline_results.json",
        "Group-Reweighted": "results/reweighted_results.json",
        "Group DRO (Balanced)": "results/group_dro_results.json",
        "Group DRO (Unbalanced)": "results/group_dro_unbalanced_results.json"
    }
    
    plot_data = []
    for method_name, path in files.items():
        if os.path.exists(path):
            with open(path, "r") as f:
                data = json.load(f)
            plot_data.append({"Method": method_name, "Metric": "Average Accuracy", "Accuracy": data["avg_acc"] * 100})
            plot_data.append({"Method": method_name, "Metric": "Worst-Group Accuracy", "Accuracy": data["worst_group_acc"] * 100})

    df = pd.DataFrame(plot_data)

    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")
    
    ax = sns.barplot(
        data=df, 
        x="Method", 
        y="Accuracy", 
        hue="Metric", 
        palette=["#2b5c8f", "#d95f02"]
    )
    
    plt.title("Waterbirds: Model Mitigation Benchmark Comparison", fontsize=14, fontweight="bold")
    plt.ylabel("Accuracy (%)", fontsize=12)
    plt.ylim(0, 105)
    plt.xticks(rotation=15)
    
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f"{height:.1f}%",
                        (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='bottom', fontsize=9, fontweight="bold",
                        xytext=(0, 3), textcoords='offset points')

    plt.tight_layout()
    output_path = "reports/figures/all_methods_comparison.png"
    plt.savefig(output_path, dpi=300)
    print(f"Updated 3-method comparison plot saved to {output_path}")

if __name__ == "__main__":
    main()