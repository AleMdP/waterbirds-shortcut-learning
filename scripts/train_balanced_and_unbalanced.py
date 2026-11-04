import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import json
import torch
from train import prepare_data_bundle, run_experiment

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Initializing shared GPU & Data Bundle once on {device}...")
    data_bundle = prepare_data_bundle("data/waterbird_complete95_forest2water2")

    # Define the 3x2 ablation matrix: (method, use_balanced_sampler, exp_name)
    experiments = [
        # 1. Standard Unbalanced DataLoader (Shuffle=True)
        {"method": "erm",       "balanced_sampler": False, "exp_name": "vit/erm_unbalanced"},
        {"method": "reweight",  "balanced_sampler": False, "exp_name": "vit/reweight_unbalanced"},
        {"method": "group_dro", "balanced_sampler": False, "exp_name": "vit/group_dro_unbalanced"},
        
        # 2. Balanced DataLoader (WeightedRandomSampler)
        {"method": "erm",       "balanced_sampler": True,  "exp_name": "vit/erm_balanced"},
        {"method": "reweight",  "balanced_sampler": True,  "exp_name": "vit/reweight_balanced"},
        {"method": "group_dro", "balanced_sampler": True,  "exp_name": "vit/group_dro_balanced"},
    ]

    # Shared baseline hyperparameters (matching Partner B's 3-epoch benchmark)
    epochs = 3
    learning_rate = 1e-4
    weight_decay = 1e-4
    eta = 0.01
    model = "transformer"  # Use Vision Transformer for this ablation study

    summary_results = {}
    for idx, exp in enumerate(experiments, 1):
        print("\n" + "=" * 70)
        print(f"RUNNING EXPERIMENT [{idx}/{len(experiments)}]: {exp['exp_name']}")
        print("=" * 70)

        metrics = run_experiment(
            data_bundle=data_bundle,
            device=device,
            epochs=epochs,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            eta=eta,
            model=model,
            **exp
        )
        summary_results[exp["exp_name"]] = metrics

    # Compile and print a clean summary table from all generated JSON files
    print("\n" + "=" * 85)
    print("FINAL SUMMARY TABLE ACROSS ALL RUNS")
    print("=" * 85)
    header = f"| {'Experiment':<22} | {'Avg Acc':>8} | {'Worst Acc':>9} | {'G0 (L/L)':>8} | {'G1 (L/W)':>8} | {'G2 (W/L)':>8} | {'G3 (W/W)':>8} |"
    print(header)
    print("|" + "-" * 24 + "|" + "-" * 10 + "|" + "-" * 11 + "|" + "-" * 10 + "|" + "-" * 10 + "|" + "-" * 10 + "|" + "-" * 10 + "|")

    summary_data = {}
    for exp in experiments:
        name = exp["exp_name"]
        res_path = os.path.join("results", f"{name}_results.json")
        if os.path.exists(res_path):
            with open(res_path, "r") as f:
                m = json.load(f)
            summary_data[name] = m
            row = (
                f"| {name:<22} "
                f"| {m['avg_acc']*100:>7.2f}% "
                f"| {m['worst_group_acc']*100:>8.2f}% "
                f"| {m['group_0_acc']*100:>7.2f}% "
                f"| {m['group_1_acc']*100:>7.2f}% "
                f"| {m['group_2_acc']*100:>7.2f}% "
                f"| {m['group_3_acc']*100:>7.2f}% |"
            )
            print(row)

    with open("results/vit/balanced_and_unbalanced_summary.json", "w") as f:
        json.dump(summary_data, f, indent=4)
    print("\nSaved combined metrics to results/vit/balanced_and_unbalanced_summary.json")

if __name__ == "__main__":
    main()