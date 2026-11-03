import os

# FORCE GPU 1 ASSIGNMENT BEFORE LOADING TORCH
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import sys
import json
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler

# Add root project directory to Python path so `src` modules load seamlessly
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.dataset import WaterbirdsDataset, get_waterbirds_transforms
from src.models import get_waterbirds_model
from src.utils import GroupLogger
from src.loss import GroupDROLoss

def compute_group_weights(dataset, num_groups=4):
    group_counts = torch.zeros(num_groups)
    for idx in range(len(dataset)):
        _, _, g = dataset[idx]
        group_counts[g] += 1
        
    total_samples = len(dataset)
    weights = total_samples / (num_groups * group_counts)
    return weights

def build_balanced_sampler(dataset, num_groups=4):
    group_counts = torch.zeros(num_groups)
    all_groups = torch.zeros(len(dataset), dtype=torch.long)
    for idx in range(len(dataset)):
        _, _, g = dataset[idx]
        group_counts[g] += 1
        all_groups[idx] = g

    weight_per_group = 1.0 / group_counts
    sample_weights = weight_per_group[all_groups]
    return WeightedRandomSampler(weights=sample_weights, num_samples=len(dataset), replacement=True)

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
        
        # Pass detached raw per-sample losses so logger doesn't recompute them
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

def main():
    parser = argparse.ArgumentParser(description="Unified Waterbirds Training CLI (ERM, Reweight, Group DRO)")
    parser.add_argument("--method", type=str, default="group_dro", choices=["erm", "reweight", "group_dro"])
    parser.add_argument("--data_dir", type=str, default="data/waterbird_complete95_forest2water2")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--eta", type=float, default=0.01, help="Group DRO exponentiated gradient step size")
    parser.add_argument("--balanced_sampler", action="store_true", help="Use WeightedRandomSampler for balanced batches")
    parser.add_argument("--exp_name", type=str, default=None, help="Custom suffix for saving results/checkpoints")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Method: {args.method} | Balanced Sampler: {args.balanced_sampler}")

    train_tf, eval_tf = get_waterbirds_transforms()
    metadata_path = os.path.join(args.data_dir, "metadata.csv")

    train_dataset = WaterbirdsDataset(metadata_path, args.data_dir, split="train", transform=train_tf)
    val_dataset = WaterbirdsDataset(metadata_path, args.data_dir, split="val", transform=eval_tf)
    test_dataset = WaterbirdsDataset(metadata_path, args.data_dir, split="test", transform=eval_tf)

    group_weights = compute_group_weights(train_dataset)

    if args.balanced_sampler:
        sampler = build_balanced_sampler(train_dataset)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler, num_workers=4, pin_memory=True)
    else:
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = get_waterbirds_model(num_classes=2, pretrained=True).to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)

    dro_criterion = None
    if args.method == "group_dro":
        dro_criterion = GroupDROLoss(
            num_groups=4,
            eta=args.eta,
            device=device
        )

    history = []
    best_val_worst_acc = -1.0
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    tag = args.exp_name if args.exp_name else args.method
    ckpt_path = f"checkpoints/{tag}_model.pth"

    print(f"\n--- Starting {args.method.upper()} Training ---")
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_epoch(
            model, train_loader, optimizer, args.method, device,
            group_weights=group_weights, dro_criterion=dro_criterion
        )
        val_metrics = evaluate(model, val_loader, device)

        train_metrics["epoch"] = epoch
        train_metrics["val_avg_acc"] = val_metrics["avg_acc"]
        train_metrics["val_worst_group_acc"] = val_metrics["worst_group_acc"]
        history.append(train_metrics)

        print(f"Epoch [{epoch}/{args.epochs}]")
        print(f"  Train -> Avg Acc: {train_metrics['avg_acc']:.4f} | Worst-Group Acc: {train_metrics['worst_group_acc']:.4f}")
        if "q_weights" in train_metrics:
            print(f"  DRO q -> {train_metrics['q_weights']}")
        print(f"  Val   -> Avg Acc: {val_metrics['avg_acc']:.4f} | Worst-Group Acc: {val_metrics['worst_group_acc']:.4f}")

        if val_metrics["worst_group_acc"] >= best_val_worst_acc:
            best_val_worst_acc = val_metrics["worst_group_acc"]
            torch.save(model.state_dict(), ckpt_path)

    print("\n--- Final Test Set Evaluation (Best Val Checkpoint) ---")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    test_metrics = evaluate(model, test_loader, device)
    test_metrics["hyperparameters"] = vars(args)

    print(f"Test Avg Accuracy:         {test_metrics['avg_acc']:.4f}")
    print(f"Test Worst-Group Accuracy: {test_metrics['worst_group_acc']:.4f} (Group {test_metrics['worst_group_id']})")
    for g in range(4):
        print(f"  - Group {g} Acc: {test_metrics[f'group_{g}_acc']:.4f} (Count: {test_metrics[f'group_{g}_count']})")

    with open(f"results/{tag}_results.json", "w") as f:
        json.dump(test_metrics, f, indent=4)

    with open(f"results/{tag}_history.json", "w") as f:
        json.dump(history, f, indent=4)

if __name__ == "__main__":
    main()