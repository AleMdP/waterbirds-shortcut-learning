import os

# FORCE GPU 1 ASSIGNMENT BEFORE LOADING TORCH
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import sys
import json
import argparse
import torch
from torch.utils.data import DataLoader

# Add root project folder to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.dataset import WaterbirdsDataset, get_waterbirds_transforms
from src.models import get_waterbirds_model
from src.utils import GroupLogger

def main():
    parser = argparse.ArgumentParser(description="Standalone Evaluation Script for Waterbirds")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to saved model weights (.pth)")
    parser.add_argument("--data_dir", type=str, default="data/waterbird_complete95_forest2water2")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--output_file", type=str, default=None, help="Optional JSON file path to output metrics")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, eval_tf = get_waterbirds_transforms()
    metadata_path = os.path.join(args.data_dir, "metadata.csv")

    dataset = WaterbirdsDataset(metadata_path, args.data_dir, split=args.split, transform=eval_tf)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    model = get_waterbirds_model(num_classes=2, pretrained=False)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model = model.to(device)
    model.eval()

    logger = GroupLogger()
    with torch.no_grad():
        for images, targets, groups in dataloader:
            images, targets, groups = images.to(device), targets.to(device), groups.to(device)
            outputs = model(images)
            logger.update(outputs, targets, groups)

    metrics = logger.get_metrics()

    print(f"\n================ EVALUATION REPORT ({args.split.upper()}) ================")
    print(f"Checkpoint:           {args.checkpoint}")
    print(f"Average Accuracy:     {metrics['avg_acc']:.4f}")
    print(f"Worst-Group Accuracy: {metrics['worst_group_acc']:.4f} (Group {metrics['worst_group_id']})\n")
    for g in range(4):
        print(f"  - Group {g} Acc: {metrics[f'group_{g}_acc']:.4f} (Count: {metrics[f'group_{g}_count']})")
    print("=================================================================\n")

    if args.output_file:
        os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
        with open(args.output_file, "w") as f:
            json.dump(metrics, f, indent=4)
        print(f"Metrics saved to: {args.output_file}")

if __name__ == "__main__":
    main()