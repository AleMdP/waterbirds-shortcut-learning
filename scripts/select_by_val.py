"""Rank finished runs by VALIDATION worst-group accuracy (model selection without
touching the test set) and print a Markdown table.

Examples:
    python scripts/select_by_val.py --model resnet50          # main 6 runs + sweep
    python scripts/select_by_val.py --model transformer
    python scripts/select_by_val.py results/vit/group_dro_sweep   # any folder
"""
import argparse
from result_utils import (MODELS, results_dir, sweep_dir, load_runs, rank_by_validation)


def print_table(title, rows):
    print(f"\n### {title}\n")
    print("| Run | Best epoch | Val worst | Val avg | Test worst | Test avg | Test avg (train-weighted) |")
    print("|---|---|---|---|---|---|---|")
    for i, r in enumerate(rank_by_validation(rows)):
        star = " **(selected)**" if i == 0 else ""
        print(f"| {r['name']}{star} | {r['best_epoch']} | {r['val_worst']*100:.1f} | {r['val_avg']*100:.1f} "
              f"| {r['test_worst']*100:.2f} | {r['test_avg']*100:.2f} | {r['test_wavg']*100:.2f} |")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directories", nargs="*", help="folders with *_history.json / *_results.json")
    parser.add_argument("--model", choices=MODELS, default=None)
    args = parser.parse_args()

    targets = list(args.directories)
    if args.model:
        targets += [results_dir(args.model), sweep_dir(args.model)]
    if not targets:
        parser.error("give --model or at least one directory")
    for d in targets:
        rows = load_runs(d)
        if rows:
            print_table(d, rows)
        else:
            print(f"\n[warning] no runs found in {d}")


if __name__ == "__main__":
    main()
