"""Mean +- std over repeated runs of the 6 main configurations.

Runs launched with `--seeds 1 2` (see train_balanced_and_unbalanced.py) are stored as
<name>_seed<k>_results.json next to the original run <name>_results.json.

    python scripts/aggregate_seeds.py --model resnet50
    python scripts/aggregate_seeds.py --model transformer --exclude_original
"""
import os
import glob
import json
import argparse
import numpy as np
from result_utils import CONFIGS, MODELS, MODEL_LABEL, results_dir, weighted_avg


def fmt(values):
    v = np.array(values) * 100
    if len(v) > 1:
        return f"{v.mean():.2f} ± {v.std(ddof=1):.2f}"
    return f"{v.mean():.2f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--exclude_original", action="store_true",
                        help="use only the *_seed*_results.json runs")
    args = parser.parse_args()

    rdir = results_dir(args.model)
    print(f"\n### {MODEL_LABEL[args.model]}: mean ± std over n runs\n")
    print("| Config | n | Worst-group acc | Avg acc | Avg acc (train-weighted) |")
    print("|---|---|---|---|---|")
    for cfg in CONFIGS:
        files = sorted(glob.glob(os.path.join(rdir, f"{cfg}_seed*_results.json")))
        if not args.exclude_original:
            orig = os.path.join(rdir, f"{cfg}_results.json")
            if os.path.exists(orig):
                files = [orig] + files
        if not files:
            continue
        runs = [json.load(open(f)) for f in files]
        print(f"| {cfg} | {len(runs)} | {fmt([r['worst_group_acc'] for r in runs])} "
              f"| {fmt([r['avg_acc'] for r in runs])} | {fmt([weighted_avg(r) for r in runs])} |")


if __name__ == "__main__":
    main()
