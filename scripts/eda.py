import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    parser = argparse.ArgumentParser(description="Exploratory Data Analysis for Waterbirds")
    parser.add_argument("--data_dir", type=str, default="data/waterbird_complete95_forest2water2", help="Path to dataset")
    parser.add_argument("--output_dir", type=str, default="reports/figures", help="Directory to save plots")
    args = parser.parse_args()

    metadata_path = os.path.join(args.data_dir, "metadata.csv")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"metadata.csv not found at {metadata_path}")

    os.makedirs(args.output_dir, exist_ok=True)
    df = pd.read_csv(metadata_path)

    # Split mapping
    split_names = {0: "Train", 1: "Val", 2: "Test"}
    df["split_name"] = df["split"].map(split_names)

    # Calculate 4 groups: y * 2 + place
    # y: 0 (Landbird), 1 (Waterbird)
    # place: 0 (Land), 1 (Water)
    df["group_id"] = df["y"] * 2 + df["place"]
    
    group_labels = {
        0: "Group 0: Landbird on Land (Majority)",
        1: "Group 1: Landbird on Water (Minority)",
        2: "Group 2: Waterbird on Land (Minority)",
        3: "Group 3: Waterbird on Water (Majority)"
    }
    df["group_label"] = df["group_id"].map(group_labels)

    print("==================================================")
    print("         WATERBIRDS DATASET SUMMARY REPORT        ")
    print("==================================================\n")

    for split_code, split_name in split_names.items():
        sub_df = df[df["split"] == split_code]
        total = len(sub_df)
        print(f"--- Split: {split_name} (Total: {total}) ---")
        
        counts = sub_df["group_id"].value_counts().sort_index()
        for g_id in range(4):
            count = counts.get(g_id, 0)
            prop = (count / total) * 100
            print(f"  {group_labels[g_id]:<40} | Count: {count:>5} | Proportion: {prop:>6.2f}%")
        print()

    # Generate and Save Visualization Plot
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")
    
    palette = ["#2b5c8f", "#d95f02", "#e7298a", "#1b9e77"]
    
    ax = sns.countplot(
        data=df, 
        x="split_name", 
        hue="group_id", 
        palette=palette,
        order=["Train", "Val", "Test"]
    )
    
    plt.title("Waterbirds Subgroup Sample Counts Across Splits", fontsize=14, fontweight="bold")
    plt.xlabel("Dataset Split", fontsize=12)
    plt.ylabel("Sample Count", fontsize=12)
    
    # Custom Legend
    handles, _ = ax.get_legend_handles_labels()
    plt.legend(handles=handles, labels=[group_labels[i] for i in range(4)], title="Subgroups", loc="upper right")
    
    plt.tight_layout()
    plot_path = os.path.join(args.output_dir, "eda_group_distribution.png")
    plt.savefig(plot_path, dpi=300)
    print(f"Exploratory chart saved to: {plot_path}\n")

if __name__ == "__main__":
    main()