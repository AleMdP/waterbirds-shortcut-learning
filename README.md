# waterbirds-shortcut-learning

waterbirds-shortcut-learning/  
├── checkpoints/                      # Saved locally on Gorina4 (.gitignore)  
│   └── reweighted_model.pth  
├── data/                             # Excluded from Git (.gitignore)  
│   └── waterbird_complete95_forest2water2/  
│       └── metadata.csv  
├── reports/  
│   └── figures/  
│       ├── eda_group_distribution.png  
│       ├── erm_vs_reweighted_comparison.png  
│       ├── erm_per_group_loss_curves.png  
│       └── reweighted_per_group_loss_curves.png  
├── results/  
│   ├── erm_baseline_results.json  
│   ├── erm_history.json  
│   ├── reweighted_results.json  
│   └── reweighted_history.json  
├── scripts/  
│   ├── eda.py  
│   ├── eval.py  
│   ├── plot_loss_curves.py  
│   └── plot_results.py  
├── src/  
│   ├── dataset.py  
│   ├── models.py  
│   ├── train_erm.py  
│   ├── train_reweighted.py  
│   └── utils.py  
├── project_plan.md            # Complete project roadmap & task division  
└── README.md                  # Instructions for reproducibility  
