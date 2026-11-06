"""Shared helpers for the analysis / plotting scripts (no torch needed)."""
import os
import json
import glob

# Number of training images per group (Sagawa et al., 2020): G0, G1, G2, G3
GROUP_TRAIN_COUNTS = [3498, 184, 56, 1057]

CONFIGS = [
    "erm_unbalanced", "reweight_unbalanced", "group_dro_unbalanced",
    "erm_balanced", "reweight_balanced", "group_dro_balanced",
]

MODELS = ["resnet50", "transformer"]
MODEL_LABEL = {"resnet50": "ResNet-50", "transformer": "ViT-B/16"}
MODEL_TAG = {"resnet50": "resnet50", "transformer": "vit"}
_MODEL_PREFIX = {"resnet50": "", "transformer": "vit/"}


def exp_prefix(model):
    """Prefix put in front of every experiment name (creates results/vit/...)."""
    return _MODEL_PREFIX[model]


def sweep_prefix(model):
    return exp_prefix(model) + "group_dro_sweep/"


def results_dir(model):
    return os.path.join("results", exp_prefix(model).rstrip("/"))


def sweep_dir(model):
    return os.path.join("results", sweep_prefix(model).rstrip("/"))


def weighted_avg(metrics):
    """Average accuracy with every group weighted by its share of the TRAINING set.

    Sagawa et al. (2020, App. C.1) report average test accuracy this way, because
    their validation/test sets are group-balanced while the training set is not.
    """
    total = float(sum(GROUP_TRAIN_COUNTS))
    return sum(GROUP_TRAIN_COUNTS[g] / total * metrics[f"group_{g}_acc"] for g in range(4))


def load_runs(directory):
    """Read every <name>_history.json + <name>_results.json pair in `directory`.

    Returns one dict per run with the validation worst-group accuracy of the epoch
    that was kept as checkpoint (last epoch reaching the max, as in train.py, which
    uses `>=`) and the corresponding TEST metrics.
    """
    rows = []
    for hp in sorted(glob.glob(os.path.join(directory, "*_history.json"))):
        rp = hp[: -len("_history.json")] + "_results.json"
        if not os.path.exists(rp):
            continue
        with open(hp) as f:
            hist = json.load(f)
        with open(rp) as f:
            res = json.load(f)
        vals = [e["val_worst_group_acc"] for e in hist]
        best = max(vals)
        best_idx = len(vals) - 1 - vals[::-1].index(best)
        rows.append({
            "name": os.path.basename(hp)[: -len("_history.json")],
            "best_epoch": best_idx + 1,
            "val_worst": best,
            "val_avg": hist[best_idx]["val_avg_acc"],
            "test_worst": res["worst_group_acc"],
            "test_avg": res["avg_acc"],
            "test_wavg": weighted_avg(res),
            "groups": [res[f"group_{g}_acc"] for g in range(4)],
        })
    return rows


def rank_by_validation(rows):
    """Best first: validation worst-group accuracy, ties broken by validation average."""
    return sorted(rows, key=lambda r: (-r["val_worst"], -r["val_avg"]))
