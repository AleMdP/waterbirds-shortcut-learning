# waterbirds-shortcut-learning

waterbirds-shortcut-learning/  
├── data/                      # Excluded from Git (.gitignore)  
│   └── waterbird_complete95_forest222/  
├── src/  
│   ├── dataset.py             # Waterbirds PyTorch Dataset & DataLoader  
│   ├── models.py              # ResNet-50 & custom heads  
│   ├── train_erm.py           # Standard ERM training pipeline  
│   ├── train_group_dro.py     # Group DRO training pipeline  
│   └── utils.py               # Metrics, logging, worst-group accuracy  
├── scripts/  
│   ├── eda.py                # Exploratory Data Analysis  
│   ├── train.py              # Main training script  
│   └── eval.py               # Evaluation & metrics computation  
├── project_plan.md            # Complete project roadmap & task division  
├── notebooks/                # Visualization notebooks  
├── README.md                  # Instructions for reproducibility  
└── requirements.txt           # Python dependencies (torch, torchvision, pandas)  
