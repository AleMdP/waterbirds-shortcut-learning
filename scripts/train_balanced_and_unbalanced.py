import os
import sys
import json
import subprocess

def main():
    # Define the 3x2 ablation matrix: (method, use_balanced_sampler, exp_name)
    experiments = [
        # 1. Standard Unbalanced DataLoader (Shuffle=True)
        {"method": "erm",       "balanced_sampler": False, "exp_name": "erm_unbalanced"},
        {"method": "reweight",  "balanced_sampler": False, "exp_name": "reweight_unbalanced"},
        {"method": "group_dro", "balanced_sampler": False, "exp_name": "group_dro_unbalanced"},
        
        # 2. Balanced DataLoader (WeightedRandomSampler)
        {"method": "erm",       "balanced_sampler": True,  "exp_name": "erm_balanced"},
        {"method": "reweight",  "balanced_sampler": True,  "exp_name": "reweight_balanced"},
        {"method": "group_dro", "balanced_sampler": True,  "exp_name": "group_dro_balanced"},
    ]

    # Shared baseline hyperparameters (matching Partner B's 3-epoch benchmark)
    epochs = 3
    lr = 1e-4
    weight_decay = 1e-4
    eta = 0.01

    train_script = os.path.join(os.path.dirname(__file__), "train.py")

    for idx, exp in enumerate(experiments, 1):
        print("\n" + "=" * 70)
        print(f"RUNNING EXPERIMENT [{idx}/{len(experiments)}]: {exp['exp_name']}")
        print("=" * 70)

        cmd = [
            sys.executable, train_script,
            "--method", exp["method"],
            "--epochs", str(epochs),
            "--lr", str(lr),
            "--weight_decay", str(weight_decay),
            "--eta", str(eta),
            "--exp_name", exp["exp_name"]
        ]
        if exp["balanced_sampler"]:
            cmd.append("--balanced_sampler")

        # Run scripts/train.py and stream output directly to terminal
        subprocess.run(cmd, check=True)

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

    with open("results/balanced_and_unbalanced_summary.json", "w") as f:
        json.dump(summary_data, f, indent=4)
    print("\nSaved combined metrics to results/balanced_and_unbalanced_summary.json")

if __name__ == "__main__":
    main()