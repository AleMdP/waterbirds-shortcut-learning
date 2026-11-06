"""Group DRO hyperparameter sweep (14 runs, balanced sampler).
 
    CUDA_VISIBLE_DEVICES=1 python scripts/train_sweep_group_dro.py --model resnet50
    CUDA_VISIBLE_DEVICES=1 python scripts/train_sweep_group_dro.py --model transformer
 
The table printed at the end is ranked by VALIDATION worst-group accuracy (not by test).
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import json
import torch
import argparse
from train import prepare_data_bundle, run_experiment
from result_utils import sweep_prefix

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["resnet50", "transformer"], required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--data_dir", type=str, default="data/waterbird_complete95_forest2water2")
    args = parser.parse_args()
 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Initializing shared GPU & Data Bundle once on {device}...")
    data_bundle = prepare_data_bundle(args.data_dir)
 
    p = sweep_prefix(args.model)  # "group_dro_sweep/" or "vit/group_dro_sweep/"
    B = {"method": "group_dro", "balanced_sampler": True}
 
    experiments = [
        # 1. Step size (eta) sweep on Group DRO + Balanced Sampler
        {**B, "eta": 0.001, "weight_decay": 1e-4, "adj_coeff": 0.0, "exp_name": f"{p}dro_eta_0.001"},
        {**B, "eta": 0.05,  "weight_decay": 1e-4, "adj_coeff": 0.0, "exp_name": f"{p}dro_eta_0.05"},
        {**B, "eta": 0.01,  "weight_decay": 1e-4, "adj_coeff": 0.0, "exp_name": f"{p}dro_eta_0.01"},
        {**B, "eta": 0.1,   "weight_decay": 1e-4, "adj_coeff": 0.0, "exp_name": f"{p}dro_eta_0.1"},
        {**B, "eta": 0.5,   "weight_decay": 1e-4, "adj_coeff": 0.0, "exp_name": f"{p}dro_eta_0.5"},
        {**B, "eta": 1.0,   "weight_decay": 1e-4, "adj_coeff": 0.0, "exp_name": f"{p}dro_eta_1.0"},
        # 2. Weight decay sweep (Sagawa et al. regularization test)
        {**B, "eta": 0.01,  "weight_decay": 1e-3, "adj_coeff": 0.0, "exp_name": f"{p}dro_wd_1e-3"},
        {**B, "eta": 0.01,  "weight_decay": 1e-2, "adj_coeff": 0.0, "exp_name": f"{p}dro_wd_1e-2"},
        {**B, "eta": 0.01,  "weight_decay": 1e-1, "adj_coeff": 0.0, "exp_name": f"{p}dro_wd_1e-1"},
        # 3. Generalization adjustment (C / sqrt(n_g)) sweep
        {**B, "eta": 0.01,  "weight_decay": 1e-2, "adj_coeff": 1.0, "exp_name": f"{p}dro_adj_1_wd_1e-2"},
        {**B, "eta": 0.01,  "weight_decay": 1e-1, "adj_coeff": 1.0, "exp_name": f"{p}dro_adj_1_wd_1e-1"},
        {**B, "eta": 0.01,  "weight_decay": 1e-2, "adj_coeff": 2.0, "exp_name": f"{p}dro_adj_2_wd_1e-2"},
        {**B, "eta": 0.01,  "weight_decay": 1e-1, "adj_coeff": 2.0, "exp_name": f"{p}dro_adj_2_wd_1e-1"},
        {**B, "eta": 0.01,  "weight_decay": 1e-3, "adj_coeff": 1.0, "learning_rate": 1e-3,
         "exp_name": f"{p}dro_adj_1_wd_1e-3_lr_1e-3"},
    ]
 
    summary = {}
    for exp in experiments:
        metrics = run_experiment(data_bundle=data_bundle, device=device, epochs=args.epochs,
                                 model=args.model, seed=args.seed, **exp)
        key = exp["exp_name"] if args.seed is None else f"{exp['exp_name']}_seed{args.seed}"
        summary[key] = metrics
 
    print("\n" + "=" * 100)
    print(f"| {'Experiment':<34} | {'Val worst':>9} | {'Val avg':>7} | {'Test worst':>10} | {'Test avg':>8} |")
    print("|" + "-" * 36 + "|" + "-" * 11 + "|" + "-" * 9 + "|" + "-" * 12 + "|" + "-" * 10 + "|")
    ranked = sorted(summary.items(),
                    key=lambda x: (-x[1]["best_val_worst_group_acc"], -x[1]["best_val_avg_acc"]))
    for name, m in ranked:
        print(f"| {name:<34} | {m['best_val_worst_group_acc']*100:>8.1f}% | {m['best_val_avg_acc']*100:>6.1f}% "
              f"| {m['worst_group_acc']*100:>9.2f}% | {m['avg_acc']*100:>7.2f}% |")
 
    suffix = "" if args.seed is None else f"_seed{args.seed}"
    out = f"results/{p}group_dro_results{suffix}.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(summary, f, indent=4)
    print(f"\nSaved sweep summary to {out}")
 
 
if __name__ == "__main__":
    main()