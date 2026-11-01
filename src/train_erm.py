import os
import json

# FORCE GPU 1 ASSIGNMENT BEFORE LOADING TORCH
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.models import resnet50, ResNet50_Weights

from dataset import WaterbirdsDataset, get_waterbirds_transforms
from utils import GroupLogger

def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    logger = GroupLogger()
    
    for images, targets, groups in dataloader:
        images, targets, groups = images.to(device), targets.to(device), groups.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        logger.update(outputs, targets, groups)
        
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
    parser = argparse.ArgumentParser(description="Train ERM Baseline on Waterbirds")
    parser.add_argument("--data_dir", type=str, default="data/waterbird_complete95_forest2water2", help="Path to extracted dataset")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device initialized: {device}")
    if torch.cuda.is_available():
        print(f"Active GPU Name: {torch.cuda.get_device_name(0)}")

    train_tf, eval_tf = get_waterbirds_transforms()
    metadata_path = os.path.join(args.data_dir, "metadata.csv")

    train_dataset = WaterbirdsDataset(metadata_path, args.data_dir, split="train", transform=train_tf)
    val_dataset = WaterbirdsDataset(metadata_path, args.data_dir, split="val", transform=eval_tf)
    test_dataset = WaterbirdsDataset(metadata_path, args.data_dir, split="test", transform=eval_tf)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Initialize Pretrained ResNet-50
    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)

    print("\n--- Starting ERM Training ---")
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate(model, val_loader, device)
        
        print(f"Epoch [{epoch}/{args.epochs}]")
        print(f"  Train -> Avg Acc: {train_metrics['avg_acc']:.4f} | Worst-Group Acc: {train_metrics['worst_group_acc']:.4f}")
        print(f"  Val   -> Avg Acc: {val_metrics['avg_acc']:.4f} | Worst-Group Acc: {val_metrics['worst_group_acc']:.4f}")

    print("\n--- Final Test Set Evaluation ---")
    test_metrics = evaluate(model, test_loader, device)
    print(f"Test Average Accuracy:     {test_metrics['avg_acc']:.4f}")
    print(f"Test Worst-Group Accuracy: {test_metrics['worst_group_acc']:.4f} (Group {test_metrics['worst_group_id']})")
    for g in range(4):
        print(f"  - Group {g} Acc: {test_metrics[f'group_{g}_acc']:.4f} (Count: {test_metrics[f'group_{g}_count']})")

    # Save test results to a JSON file
    os.makedirs("results", exist_ok=True)
    results_path = os.path.join("results", "erm_baseline_results.json")
    
    with open(results_path, "w") as f:
        json.dump(test_metrics, f, indent=4)
        
    print(f"\nTest metrics successfully saved to {results_path}")


if __name__ == "__main__":
    main()