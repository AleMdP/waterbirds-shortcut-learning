"""Main experiment matrix: {ERM, Reweight, Group DRO} x {unbalanced, balanced sampler}.
 
    CUDA_VISIBLE_DEVICES=1 python scripts/train_balanced_and_unbalanced.py --model resnet50
    CUDA_VISIBLE_DEVICES=1 python scripts/train_balanced_and_unbalanced.py --model transformer
    # extra independent runs (files get a _seed<k> suffix):
    CUDA_VISIBLE_DEVICES=1 python scripts/train_balanced_and_unbalanced.py --model resnet50 --seeds 1 2
 
ResNet-50 results go to results/, ViT-B/16 results to results/vit/ (same layout as before).
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import json
import torch
import argparse
from train import prepare_data_bundle, run_experiment
from result_utils import exp_prefix

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["resnet50", "transformer"], required=True)
    parser.add_argument("--seeds", type=int, nargs="*", default=[],
                        help="optional list of seeds; without it each config is run once, unseeded")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--eta", type=float, default=0.01)
    parser.add_argument("--data_dir", type=str, default="data/waterbird_complete95_forest2water2")
    args = parser.parse_args()
 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Initializing shared GPU & Data Bundle once on {device}...")
    data_bundle = prepare_data_bundle(args.data_dir)
 
    prefix = exp_prefix(args.model)
 
    # (method, use_balanced_sampler, exp_name)
    experiments = [
        {"method": "erm",       "balanced_sampler": False, "exp_name": f"{prefix}erm_unbalanced"},
        {"method": "reweight",  "balanced_sampler": False, "exp_name": f"{prefix}reweight_unbalanced"},
        {"method": "group_dro", "balanced_sampler": False, "exp_name": f"{prefix}group_dro_unbalanced"},
        {"method": "erm",       "balanced_sampler": True,  "exp_name": f"{prefix}erm_balanced"},
        {"method": "reweight",  "balanced_sampler": True,  "exp_name": f"{prefix}reweight_balanced"},
        {"method": "group_dro", "balanced_sampler": True,  "exp_name": f"{prefix}group_dro_balanced"},
    ]
    seeds = args.seeds if args.seeds else [None]
 
    summary = {}
    total = len(experiments) * len(seeds)
    n = 0
    for seed in seeds:
        for exp in experiments:
            n += 1
            print("\n" + "=" * 70)
            print(f"RUNNING EXPERIMENT [{n}/{total}]: {exp['exp_name']} (seed={seed})")
            print("=" * 70)
            metrics = run_experiment(
                data_bundle=data_bundle, device=device, epochs=args.epochs,
                learning_rate=args.learning_rate, weight_decay=args.weight_decay, eta=args.eta,
                model=args.model, seed=seed, **exp
            )
            key = exp["exp_name"] if seed is None else f"{exp['exp_name']}_seed{seed}"
            summary[key] = metrics
 
    print("\n" + "=" * 90)
    print("FINAL SUMMARY TABLE")
    print("=" * 90)
    print(f"| {'Experiment':<32} | {'Avg':>7} | {'Worst':>7} | {'G0':>6} | {'G1':>6} | {'G2':>6} | {'G3':>6} |")
    print("|" + "-" * 34 + "|" + "-" * 9 + "|" + "-" * 9 + "|" + ("-" * 8 + "|") * 4)
    for name, m in summary.items():
        print(f"| {name:<32} | {m['avg_acc']*100:>6.2f}% | {m['worst_group_acc']*100:>6.2f}% "
              f"| {m['group_0_acc']*100:>5.1f}% | {m['group_1_acc']*100:>5.1f}% "
              f"| {m['group_2_acc']*100:>5.1f}% | {m['group_3_acc']*100:>5.1f}% |")
 
    suffix = "_seeds" if args.seeds else ""
    out = f"results/{prefix}balanced_and_unbalanced_summary{suffix}.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(summary, f, indent=4)
    print(f"\nSaved combined metrics to {out}")
 
 
if __name__ == "__main__":
    main()