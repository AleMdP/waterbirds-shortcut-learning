# waterbirds-shortcut-learning

waterbirds-shortcut-learning/  
├── data/                      # Local/Gorina4 dataset (Ignored by Git)  
│   └── waterbird_complete95_forest2water2/  
│       ├── metadata.csv  
│       ├── 001.Black_footed_Albatross/  
│       ├── 002.Laysan_Albatross/  
│       └── ... (200 carpetas de especies)  
├── src/  
│   ├── dataset.py             # Waterbirds PyTorch Dataset & DataLoader  
│   ├── models.py              # ResNet-50 & custom heads  
│   ├── train_erm.py           # Standard ERM training pipeline  
│   ├── train_group_dro.py     # Group DRO training pipeline  
│   └── utils.py               # Metrics, logging, worst-group accuracy  
├── project_plan.md            # Complete project roadmap & task division  
├── README.md                  # Instructions for reproducibility  
└── requirements.txt           # Python dependencies (torch, torchvision, pandas)  
