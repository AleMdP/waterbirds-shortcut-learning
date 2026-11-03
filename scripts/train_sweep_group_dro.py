import os
import sys
import json
import subprocess

def main():
    # Hyperparameter sweep focused on Group DRO (with Balanced Sampler)
    sweeps = [
        # 1. Step size (eta) sweep
        {"method": "group_dro", "eta": 0.001, "wd": 1e-4, "adj": 0.0, "lr": 1e-4, "name": "dro_eta_0.001"},
        {"method": "group_dro", "eta": 0.01,  "wd": 1e-4, "adj": 0.0, "lr": 1e-4, "name": "dro_eta_0.01_base"},
        {"method": "group_dro", "eta": 0.1,   "wd": 1e-4, "adj": 0.0, "lr": 1e-4, "name": "dro_eta_0.1"},
        {"method": "group_dro", "eta": 1.0,   "wd": 1e-4, "adj": 0.0, "lr": 1e-4, "name": "dro_eta_1.0"},

        # 2. Strong Weight Decay sweep (Sagawa et al. key finding)
        {"method": "group_dro", "eta": 0.01,  "wd": 1e-2, "adj": 0.0, "lr": 1e-4, "name": "dro_wd_1e-2"},
        {"method": "group_dro", "eta": 0.01,  "wd": 1e-1, "adj": 0.0, "lr": 1e-4, "name": "dro_wd_1e-1"},
        {"method": "group_dro", "eta": 0.01,  "wd": 1.0,  "adj": 0.0, "lr": 1e-4, "name": "dro_wd_1.0"},

        # 3. Group-Size Generalization Adjustment (C / sqrt(n_g)) + Strong WD
        {"method": "group_dro", "eta": 0.01,  "wd": 1e-2, "adj": 2.0, "lr": 1e-4, "name": "dro_adj_2.0_wd_1e-2"},
        {"method": "group_dro", "eta": 0.01,  "wd": 1e-1, "adj": 2.0, "lr": 1e-4, "name": "dro_adj_2.0_wd_1e-1"},
        {"method": "group_dro", "eta": 0.01,  "wd": 1e-3, "adj": 1.0, "lr": 1e-3, "name": "dro_lr_1e-3_wd_1e-3_adj_1"},
    ]

    train_script = os.path.join(os.path.dirname(__file__), "train.py")

    for idx, cfg in enumerate(sweeps, 1):
        print(f"\n>>> Running Sweep [{idx}/{len(sweeps)}]: {cfg['name']} <<<")
        cmd = [
            sys.executable, train_script,
            "--method", cfg["method"],
            "--balanced_sampler",
            "--epochs", "3",
            "--lr", str(cfg["lr"]),
            "--weight_decay", str(cfg["wd"]),
            "--eta", str(cfg["eta"]),
            "--adj_coeff", str(cfg["adj"]),
            "--exp_name", cfg["name"]
        ]
        subprocess.run(cmd, check=True)

    print("\n" + "=" * 95)
    print("HYPERPARAMETER SWEEP RESULTS (SORTED BY WORST-GROUP ACCURACY)")
    print("=" * 95)
    print(f"| {'Sweep Run':<26} | {'Eta':>5} | {'WD':>6} | {'Adj':>4} | {'Avg Acc':>8} | {'Worst Acc':>9} | {'G0':>6} | {'G1':>6} | {'G2':>6} | {'G3':>6} |")
    print("|" + "-"*28 + "|" + "-"*7 + "|" + "-"*8 + "|" + "-"*6 + "|" + "-"*10 + "|" + "-"*11 + "|" + "-"*8 + "|" + "-"*8 + "|" + "-"*8 + "|" + "-"*8 + "|")

    rows = []
    for cfg in sweeps:
        res_path = os.path.join("results", f"{cfg['name']}_results.json")
        if os.path.exists(res_path):
            with open(res_path, "r") as f:
                m = json.load(f)
            rows.append((cfg, m))

    # Sort best worst-group accuracy first
    rows.sort(key=lambda x: x[1]["worst_group_acc"], reverse=True)

    sweep_summary = {}
    for cfg, m in rows:
        sweep_summary[cfg["name"]] = m
        print(
            f"| {cfg['name']:<26} "
            f"| {cfg['eta']:>5} "
            f"| {cfg['wd']:>6} "
            f"| {cfg['adj']:>4} "
            f"| {m['avg_acc']*100:>7.2f}% "
            f"| {m['worst_group_acc']*100:>8.2f}% "
            f"| {m['group_0_acc']*100:>5.1f}% "
            f"| {m['group_1_acc']*100:>5.1f}% "
            f"| {m['group_2_acc']*100:>5.1f}% "
            f"| {m['group_3_acc']*100:>5.1f}% |"
        )

    with open("results/group_dro_results.json", "w") as f:
        json.dump(sweep_summary, f, indent=4)

if __name__ == "__main__":
    main()