import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import json
import torch
from train import prepare_data_bundle, run_experiment

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Initializing shared GPU & Data Bundle once on {device}...")
    data_bundle = prepare_data_bundle("data/waterbird_complete95_forest2water2")

    # Add any ablation or hyperparameter sweep configurations here!
    experiments = [
        # 1. Step size (eta) sweep on Group DRO + Balanced Sampler
        {"method": "group_dro", "balanced_sampler": True, "eta": 0.001, "weight_decay": 1e-4, "adj_coeff": 0.0, "exp_name": "dro_eta_0.001"},
        {"method": "group_dro", "balanced_sampler": True, "eta": 0.05,  "weight_decay": 1e-4, "adj_coeff": 0.0, "exp_name": "dro_eta_0.05"},
        {"method": "group_dro", "balanced_sampler": True, "eta": 0.01,  "weight_decay": 1e-4, "adj_coeff": 0.0, "exp_name": "dro_eta_0.01"},
        {"method": "group_dro", "balanced_sampler": True, "eta": 0.1,   "weight_decay": 1e-4, "adj_coeff": 0.0, "exp_name": "dro_eta_0.1"},
        {"method": "group_dro", "balanced_sampler": True, "eta": 0.5,   "weight_decay": 1e-4, "adj_coeff": 0.0, "exp_name": "dro_eta_0.5"},
        {"method": "group_dro", "balanced_sampler": True, "eta": 1.0,   "weight_decay": 1e-4, "adj_coeff": 0.0, "exp_name": "dro_eta_1.0"},
        
        # 2. Weight decay sweep (Sagawa et al. regularization test)
        {"method": "group_dro", "balanced_sampler": True, "eta": 0.01,  "weight_decay": 1e-3, "adj_coeff": 0.0, "exp_name": "dro_wd_1e-3"},
        {"method": "group_dro", "balanced_sampler": True, "eta": 0.01,  "weight_decay": 1e-2, "adj_coeff": 0.0, "exp_name": "dro_wd_1e-2"},
        {"method": "group_dro", "balanced_sampler": True, "eta": 0.01,  "weight_decay": 1e-1, "adj_coeff": 0.0, "exp_name": "dro_wd_1e-1"},
        
        # 3. Generalization adjustment (C / sqrt(n_g)) sweep
        {"method": "group_dro", "balanced_sampler": True, "eta": 0.01,  "weight_decay": 1e-2, "adj_coeff": 1.0, "exp_name": "dro_adj_1_wd_1e-2"},
        {"method": "group_dro", "balanced_sampler": True, "eta": 0.01,  "weight_decay": 1e-1, "adj_coeff": 1.0, "exp_name": "dro_adj_1_wd_1e-1"},
        {"method": "group_dro", "balanced_sampler": True, "eta": 0.01,  "weight_decay": 1e-2, "adj_coeff": 2.0, "exp_name": "dro_adj_2_wd_1e-2"},
        {"method": "group_dro", "balanced_sampler": True, "eta": 0.01,  "weight_decay": 1e-1, "adj_coeff": 2.0, "exp_name": "dro_adj_2_wd_1e-1"},
        {"method": "group_dro", "balanced_sampler": True, "eta": 0.01,  "weight_decay": 1e-3, "adj_coeff": 1.0, "learning_rate": 1e-3, "exp_name": "dro_adj_1_wd_1e-3_lr_1e-3"},
    ]

    model = "transformer"  # Use Vision Transformer for this ablation study

    summary_results = {}
    for exp in experiments:
        metrics = run_experiment(
            data_bundle=data_bundle,
            device=device,
            epochs=3,
            model=model,
            **exp
        )
        summary_results[exp["exp_name"]] = metrics

    # Print sorted Markdown table
    print("\n" + "=" * 85)
    print(f"| {'Experiment':<22} | {'Avg Acc':>8} | {'Worst Acc':>9} | {'G0 (L/L)':>8} | {'G1 (L/W)':>8} | {'G2 (W/L)':>8} | {'G3 (W/W)':>8} |")
    print("|" + "-"*24 + "|" + "-"*10 + "|" + "-"*11 + "|" + "-"*10 + "|" + "-"*10 + "|" + "-"*10 + "|" + "-"*10 + "|")
    
    sorted_exps = sorted(summary_results.items(), key=lambda x: x[1]["worst_group_acc"], reverse=True)
    for name, m in sorted_exps:
        print(
            f"| {name:<22} "
            f"| {m['avg_acc']*100:>7.2f}% "
            f"| {m['worst_group_acc']*100:>8.2f}% "
            f"| {m['group_0_acc']*100:>7.2f}% "
            f"| {m['group_1_acc']*100:>7.2f}% "
            f"| {m['group_2_acc']*100:>7.2f}% "
            f"| {m['group_3_acc']*100:>7.2f}% |"
        )

    with open("results/group_dro_results.json", "w") as f:
        json.dump(summary_results, f, indent=4)

if __name__ == "__main__":
    main()