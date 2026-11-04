import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import sys
import json
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.dataset import WaterbirdsDataset, get_waterbirds_transforms
from src.models import get_waterbirds_model
from src.utils import GroupLogger
from src.loss import GroupDROLoss


def prepare_data_bundle(data_dir="data/waterbird_complete95_forest2water2", num_groups=4):
    """
    Loads train/val/test datasets and computes group counts, weights, and sample weights ONCE.
    """
    train_tf, eval_tf = get_waterbirds_transforms()
    metadata_path = os.path.join(data_dir, "metadata.csv")

    train_dataset = WaterbirdsDataset(metadata_path, data_dir, split="train", transform=train_tf)
    val_dataset = WaterbirdsDataset(metadata_path, data_dir, split="val", transform=eval_tf)
    test_dataset = WaterbirdsDataset(metadata_path, data_dir, split="test", transform=eval_tf)

    # Fast group extraction (uses dataframe if present, otherwise loops once)
    if hasattr(train_dataset, "df"):
        y_vals = torch.tensor(train_dataset.df["y"].values, dtype=torch.long)
        place_vals = torch.tensor(train_dataset.df["place"].values, dtype=torch.long)
        all_groups = y_vals * 2 + place_vals
        group_counts = torch.bincount(all_groups, minlength=num_groups).float()
    else:
        group_counts = torch.zeros(num_groups)
        all_groups = torch.zeros(len(train_dataset), dtype=torch.long)
        for idx in range(len(train_dataset)):
            _, _, g = train_dataset[idx]
            group_counts[g] += 1
            all_groups[idx] = g

    total_samples = len(train_dataset)
    group_weights = total_samples / (num_groups * group_counts)
    
    weight_per_group = 1.0 / group_counts
    sample_weights = weight_per_group[all_groups]

    return {
        "train_dataset": train_dataset,
        "val_dataset": val_dataset,
        "test_dataset": test_dataset,
        "group_counts": group_counts,
        "group_weights": group_weights,
        "sample_weights": sample_weights,
    }


def train_epoch(model, dataloader, optimizer, method, device, group_weights=None, dro_criterion=None):
    model.train()
    logger = GroupLogger()
    ce_criterion = nn.CrossEntropyLoss(reduction="none")
    
    if group_weights is not None:
        group_weights = group_weights.to(device)

    for images, targets, groups in dataloader:
        images, targets, groups = images.to(device), targets.to(device), groups.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        
        if method == "erm":
            raw_loss = ce_criterion(outputs, targets)
            loss = raw_loss.mean()
        elif method == "reweight":
            raw_loss = ce_criterion(outputs, targets)
            sample_weights = group_weights[groups]
            loss = (raw_loss * sample_weights).mean()
        elif method == "group_dro":
            loss, raw_loss = dro_criterion(outputs, targets, groups)
        else:
            raise ValueError(f"Unknown method: {method}")
            
        loss.backward()
        optimizer.step()
        
        logger.update(outputs, targets, groups, raw_losses=raw_loss.detach())
        
    metrics = logger.get_metrics()
    if method == "group_dro" and dro_criterion is not None:
        metrics["q_weights"] = [round(val, 4) for val in dro_criterion.q.detach().cpu().tolist()]
    return metrics


def evaluate(model, dataloader, device):
    model.eval()
    logger = GroupLogger()
    
    with torch.no_grad():
        for images, targets, groups in dataloader:
            images, targets, groups = images.to(device), targets.to(device), groups.to(device)
            outputs = model(images)
            logger.update(outputs, targets, groups)
            
    return logger.get_metrics()


def run_experiment(
    method="group_dro",
    data_dir="data/waterbird_complete95_forest2water2",
    batch_size=32,
    epochs=3,
    learning_rate=1e-4,
    weight_decay=1e-4,
    eta=0.01,
    adj_coeff=0.0,
    balanced_sampler=False,
    exp_name=None,
    data_bundle=None,
    device=None,
):
    """
    Directly callable Python function to run a single experiment without spawning a subprocess.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Only load datasets & group weights if not already passed in!
    if data_bundle is None:
        data_bundle = prepare_data_bundle(data_dir)

    train_dataset = data_bundle["train_dataset"]
    val_dataset = data_bundle["val_dataset"]
    test_dataset = data_bundle["test_dataset"]
    group_counts = data_bundle["group_counts"]
    group_weights = data_bundle["group_weights"]
    sample_weights = data_bundle["sample_weights"]

    if balanced_sampler:
        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(train_dataset), replacement=True)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=4, pin_memory=True)
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = get_waterbirds_model(num_classes=2, pretrained=True).to(device)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)

    dro_criterion = None
    if method == "group_dro":
        dro_criterion = GroupDROLoss(
            num_groups=4,
            eta=eta,
            group_counts=group_counts,
            adj_coeff=adj_coeff,
            device=device
        )

    history = []
    best_val_worst_acc = -1.0
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    tag = exp_name if exp_name else method
    ckpt_path = f"checkpoints/{tag}_model.pth"

    print(f"\n--- Running [{tag}] | Method: {method} | Balanced: {balanced_sampler} | lr={learning_rate}, wd={weight_decay}, eta={eta}, adj={adj_coeff} ---")
    for epoch in range(1, epochs + 1):
        train_metrics = train_epoch(
            model, train_loader, optimizer, method, device,
            group_weights=group_weights, dro_criterion=dro_criterion
        )
        val_metrics = evaluate(model, val_loader, device)

        train_metrics["epoch"] = epoch
        train_metrics["val_avg_acc"] = val_metrics["avg_acc"]
        train_metrics["val_worst_group_acc"] = val_metrics["worst_group_acc"]
        history.append(train_metrics)

        print(f"Epoch [{epoch}/{epochs}] Train Worst: {train_metrics['worst_group_acc']:.4f} | Val Worst: {val_metrics['worst_group_acc']:.4f} (Val Avg: {val_metrics['avg_acc']:.4f})")

        if val_metrics["worst_group_acc"] >= best_val_worst_acc:
            best_val_worst_acc = val_metrics["worst_group_acc"]
            torch.save(model.state_dict(), ckpt_path)

    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    test_metrics = evaluate(model, test_loader, device)
    test_metrics["hyperparameters"] = {
        "method": method, "batch_size": batch_size, "epochs": epochs,
        "learning_rate": learning_rate, "weight_decay": weight_decay, "eta": eta,
        "adj_coeff": adj_coeff, "balanced_sampler": balanced_sampler, "exp_name": tag
    }

    print(f"--> Test Result [{tag}]: Avg Acc = {test_metrics['avg_acc']*100:.2f}% | Worst-Group Acc = {test_metrics['worst_group_acc']*100:.2f}% (G{test_metrics['worst_group_id']})")

    with open(f"results/{tag}_results.json", "w") as f:
        json.dump(test_metrics, f, indent=4)

    with open(f"results/{tag}_history.json", "w") as f:
        json.dump(history, f, indent=4)

    return test_metrics


def main():
    parser = argparse.ArgumentParser(description="Unified Waterbirds Training CLI")
    parser.add_argument("--method", type=str, default="group_dro", choices=["erm", "reweight", "group_dro"])
    parser.add_argument("--data_dir", type=str, default="data/waterbird_complete95_forest2water2")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--eta", type=float, default=0.01)
    parser.add_argument("--adj_coeff", type=float, default=0.0)
    parser.add_argument("--balanced_sampler", action="store_true")
    parser.add_argument("--exp_name", type=str, default=None)
    args = parser.parse_args()

    run_experiment(**vars(args))


if __name__ == "__main__":
    main()