import os

# FORCE GPU 1 ASSIGNMENT BEFORE LOADING TORCH
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import json
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import WaterbirdsDataset, get_waterbirds_transforms
from models import get_waterbirds_model
from utils import GroupLogger

# Inside src/loss.py
import torch
import torch.nn as nn

class GroupDROLoss(nn.Module):
    """
    Group Distributionally Robust Optimization (Group DRO) Loss.
    Adapted from Sagawa et al. (2020), "Distributionally Robust Neural Networks for Group Shifts".
    q is the vector of group weights, updated dynamically during training.
    """
    def __init__(self, num_groups=4, eta=0.01, device="cuda"):
        super().__init__()
        self.num_groups = num_groups
        self.eta = eta
        self.criterion = nn.CrossEntropyLoss(reduction="none")

        # Initialize uniform group weights q = [0.25, 0.25, 0.25, 0.25]
        self.q = torch.ones(num_groups, device=device) / num_groups

    def forward(self, outputs, targets, groups):
        # 1. Per-sample cross-entropy loss (shape: [batch_size])
        raw_loss = self.criterion(outputs, targets)
        
        # 2. Compute mean loss for each group present in this batch
        group_losses = torch.zeros(self.num_groups, device=outputs.device)
        for g in range(self.num_groups):
            mask = (groups == g)
            if mask.sum() > 0:
                group_losses[g] = raw_loss[mask].mean()
                
        # 3. Update group weights q dynamically (detached from autograd)
        with torch.no_grad():
            self.q = self.q * torch.exp(self.eta * group_losses.detach())
            q_sum = self.q.sum()
            if q_sum > 0:
                self.q = self.q / q_sum
            else:
                print("Warning: Reset group weights to uniform due to zero sum.")
                self.q = torch.ones(self.num_groups, device=outputs.device) / self.num_groups
            
        # 4. Weighted sum of group losses
        dro_loss = torch.dot(self.q, group_losses)
        
        return dro_loss, raw_loss

def train_epoch(model, dataloader, optimizer, dro_criterion, device):
    model.train()
    logger = GroupLogger()
    
    for images, targets, groups in dataloader:
        images, targets, groups = images.to(device), targets.to(device), groups.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        
        # GroupDROLoss updates q and returns the weighted scalar loss + raw per-sample losses
        loss, raw_loss = dro_criterion(outputs, targets, groups)
        
        loss.backward()
        optimizer.step()
        
        logger.update(outputs, targets, groups, raw_losses=raw_loss)
        
    return logger.get_metrics()

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
    parser = argparse.ArgumentParser(description="Train Group-Reweighted Baseline on Waterbirds")
    parser.add_argument("--data_dir", type=str, default="data/waterbird_complete95_forest2water2")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Active Device: {device}")

    train_tf, eval_tf = get_waterbirds_transforms()
    metadata_path = os.path.join(args.data_dir, "metadata.csv")

    train_dataset = WaterbirdsDataset(metadata_path, args.data_dir, split="train", transform=train_tf)
    val_dataset = WaterbirdsDataset(metadata_path, args.data_dir, split="val", transform=eval_tf)
    test_dataset = WaterbirdsDataset(metadata_path, args.data_dir, split="test", transform=eval_tf)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)


    print("Initializing re-weighting factors...")
    dro_criterion = GroupDROLoss(num_groups=4)
    print(f"Assigned initial weights per group (0..3): {dro_criterion.q.tolist()}")

    model = get_waterbirds_model(num_classes=2, pretrained=True).to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)

    history = []

    print("\n--- Starting Group-Reweighted Training ---")
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_epoch(model, train_loader, optimizer, dro_criterion, device)
        val_metrics = evaluate(model, val_loader, device)
        
        train_metrics["epoch"] = epoch
        history.append(train_metrics)

        print(f"Epoch [{epoch}/{args.epochs}]")
        print(f"  Train -> Avg Acc: {train_metrics['avg_acc']:.4f} | Worst-Group Acc: {train_metrics['worst_group_acc']:.4f}")
        print(f"  Val   -> Avg Acc: {val_metrics['avg_acc']:.4f} | Worst-Group Acc: {val_metrics['worst_group_acc']:.4f}")

    print("\n--- Final Test Set Evaluation ---")
    test_metrics = evaluate(model, test_loader, device)
    print(f"Test Avg Accuracy:         {test_metrics['avg_acc']:.4f}")
    print(f"Test Worst-Group Accuracy: {test_metrics['worst_group_acc']:.4f} (Group {test_metrics['worst_group_id']})")

    os.makedirs("results", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    
    torch.save(model.state_dict(), "checkpoints/DROweighted_model.pth")
    
    with open("results/DROweighted_results.json", "w") as f:
        json.dump(test_metrics, f, indent=4)

    with open("results/DROweighted_history.json", "w") as f:
        json.dump(history, f, indent=4)
        
    print("\nTest metrics saved to results/DROweighted_results.json")
    print("Training history saved to results/DROweighted_history.json")

if __name__ == "__main__":
    main()