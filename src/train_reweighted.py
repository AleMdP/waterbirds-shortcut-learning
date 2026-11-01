import os

# FORZAR ASIGNACIÓN A GPU 1 ANTES DE CARGAR TORCH
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

def compute_group_weights(dataset, num_groups=4):
    """
    Calcula pesos inversamente proporcionales al tamaño de cada subgrupo:
    w_g = N_total / (num_groups * N_g)
    """
    group_counts = torch.zeros(num_groups)
    for idx in range(len(dataset)):
        _, _, g = dataset[idx]
        group_counts[g] += 1
        
    total_samples = len(dataset)
    weights = total_samples / (num_groups * group_counts)
    return weights

def train_epoch(model, dataloader, optimizer, group_weights, device):
    model.train()
    logger = GroupLogger()
    
    for images, targets, groups in dataloader:
        images, targets, groups = images.to(device), targets.to(device), groups.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        
        # Calcular CrossEntropyLoss sin reducir por batch
        raw_loss = nn.CrossEntropyLoss(reduction="none")(outputs, targets)
        
        # Aplicar el peso del subgrupo a cada muestra del batch
        sample_weights = group_weights[groups].to(device)
        loss = (raw_loss * sample_weights).mean()
        
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
    parser = argparse.ArgumentParser(description="Train Group-Reweighted Baseline on Waterbirds")
    parser.add_argument("--data_dir", type=str, default="data/waterbird_complete95_forest2water2")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo activo: {device}")

    train_tf, eval_tf = get_waterbirds_transforms()
    metadata_path = os.path.join(args.data_dir, "metadata.csv")

    train_dataset = WaterbirdsDataset(metadata_path, args.data_dir, split="train", transform=train_tf)
    val_dataset = WaterbirdsDataset(metadata_path, args.data_dir, split="val", transform=eval_tf)
    test_dataset = WaterbirdsDataset(metadata_path, args.data_dir, split="test", transform=eval_tf)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Calcular pesos por grupo
    print("Calculando pesos de re-ponderación por subgrupo...")
    group_weights = compute_group_weights(train_dataset)
    print(f"Pesos asignados por grupo (0..3): {group_weights.tolist()}")

    model = get_waterbirds_model(num_classes=2, pretrained=True).to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)

    print("\n--- Iniciando Entrenamiento Re-ponderado ---")
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_epoch(model, train_loader, optimizer, group_weights, device)
        val_metrics = evaluate(model, val_loader, device)
        
        print(f"Epoch [{epoch}/{args.epochs}]")
        print(f"  Train -> Avg Acc: {train_metrics['avg_acc']:.4f} | Worst-Group Acc: {train_metrics['worst_group_acc']:.4f}")
        print(f"  Val   -> Avg Acc: {val_metrics['avg_acc']:.4f} | Worst-Group Acc: {val_metrics['worst_group_acc']:.4f}")

    print("\n--- Evaluación Final en Test Set ---")
    test_metrics = evaluate(model, test_loader, device)
    print(f"Test Avg Accuracy:         {test_metrics['avg_acc']:.4f}")
    print(f"Test Worst-Group Accuracy: {test_metrics['worst_group_acc']:.4f} (Grupo {test_metrics['worst_group_id']})")

    # Guardar modelo e hiperparámetros/resultados
    os.makedirs("results", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    
    torch.save(model.state_dict(), "checkpoints/reweighted_model.pth")
    
    with open("results/reweighted_results.json", "w") as f:
        json.dump(test_metrics, f, indent=4)
        
    print("Resultados e historial guardados en results/reweighted_results.json")

if __name__ == "__main__":
    main()