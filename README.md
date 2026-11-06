# Waterbirds Detection: Classifying a Minority

Shortcut learning on the Waterbirds benchmark (Sagawa et al., 2020).
Goal: maximise **worst-group accuracy** (not only average accuracy) and report the trade-off between the two.

Authors: Lukáš Malík, Alejandro Macías del Pozo

We compare three objectives (ERM, group-reweighted loss, Group DRO), two data-loading schemes
(standard shuffling vs. group-balanced `WeightedRandomSampler`) and two pretrained backbones
(ResNet-50, ViT-B/16). A hyperparameter sweep is run for Group DRO on both backbones, and the
configuration is selected on the **validation** set.

---

## 1. Repository structure

```
waterbirds-shortcut-learning/
├── data/                         # NOT in Git. Dataset goes here (see Section 3)
│   └── waterbird_complete95_forest2water2/
│       ├── metadata.csv
│       └── <image folders>
├── checkpoints/                  # NOT in Git. Created while training (*.pth)
├── src/
│   ├── dataset.py                # WaterbirdsDataset + transforms
│   ├── models.py                 # ResNet-50 and ViT-B/16 with a 2-class head
│   ├── loss.py                   # GroupDROLoss
│   └── utils.py                  # GroupLogger (per-group acc/loss, worst-group acc)
├── scripts/
│   ├── eda.py                    # dataset statistics + figure
│   ├── train.py                  # unified training CLI (erm | reweight | group_dro)
│   ├── train_balanced_and_unbalanced.py   # 3 methods x 2 samplers (main experiment matrix)
│   ├── train_sweep_group_dro.py  # Group DRO hyperparameter sweep (14 runs)
│   ├── eval.py                   # re-evaluate a saved checkpoint
│   ├── result_utils.py           # helpers shared by the analysis scripts
│   ├── select_by_val.py          # model selection on validation + Markdown tables
│   ├── aggregate_seeds.py        # mean ± std over repeated runs
│   ├── plot_results.py           # bar plots (method x sampler, top-5 sweep configs)
│   ├── plot_tradeoff.py          # average vs worst-group scatter (all runs)
│   └── plot_loss_curves.py       # per-group training-loss curves
├── results/                      # JSON outputs (ResNet-50 in results/, ViT in results/vit/)
├── results_3_epochs/             # early preliminary runs (3 epochs, ResNet-50). Not used in the report
├── reports/                      # report PDF and figures
├── requirements.txt
└── README.md
```

## 2. Environment

Python 3.10+ is recommended.

```bash
git clone https://github.com/AleMdP/waterbirds-shortcut-learning.git
cd waterbirds-shortcut-learning
conda activate waterbirds (create it before)
pip install -r requirements.txt
```

`requirements.txt`: `torch`, `torchvision` (>= 0.13, needed for the `weights=` API), `pandas`, `numpy`, `pillow`, `matplotlib`, `seaborn`.
The first run downloads the ImageNet-pretrained weights of ResNet-50 and ViT-B/16 through torchvision, so internet access is needed once.

### GPU selection

The training and evaluation scripts (`train.py`, `train_balanced_and_unbalanced.py`, `train_sweep_group_dro.py`, `eval.py`)
set `os.environ["CUDA_VISIBLE_DEVICES"] = "1"`.  

If no GPU is visible the code prints a warning and runs on the CPU.  
The analysis scripts (`select_by_val.py`, `plot_*.py`, `aggregate_seeds.py`) do not need a GPU.

## 3. Data

Download the Waterbirds dataset released by the authors of Group DRO
(https://github.com/kohpangwei/group_DRO) and extract it into `data/` (we had some problems installing the dataset, so if you need it, tell us).  

**Groups.** `group = y * 2 + place`, where `y` = 0 landbird / 1 waterbird and `place` = 0 land / 1 water:

| Group | Bird / Background | Role | Train | Val | Test |
|---|---|---|---|---|---|
| G0 | Landbird / Land   | majority | 3498 | 467 | 2255 |
| G1 | Landbird / Water  | minority | 184  | 466 | 2255 |
| G2 | Waterbird / Land  | minority | 56   | 133 | 642  |
| G3 | Waterbird / Water | majority | 1057 | 133 | 642  |

(`scripts/eda.py` prints the exact counts.)

**Metrics.**
* *Worst-group accuracy*: minimum over G0..G3 of the test accuracy.
* *Average accuracy*: plain accuracy over the whole test set. The test set is group-balanced (see table), so this is **not** the accuracy under the training distribution.
* *Average accuracy (train-weighted)*: the four per-group test accuracies weighted by the group shares of the training set (3498/184/56/1057 of 4795). This is how Sagawa et al. (2020) report average accuracy; it is computed in `scripts/result_utils.py` from the saved per-group accuracies.

The checkpoint kept for the test evaluation is the epoch with the highest **validation** worst-group accuracy
(last epoch in case of a tie). Hyperparameter configurations are also ranked on validation (Section 4.5).

## 4. Reproducing the experiments

Run every command from the repository root. Output folders (`results/`, `results/vit/`, `checkpoints/`, `reports/figures/`, ...) are created automatically.
Training scripts take `--model resnet50` or `--model transformer` (ViT-B/16).

### 4.1 Exploratory data analysis

```bash
python scripts/eda.py
```
Output: group counts per split in the terminal and `reports/figures/eda_group_distribution.png`.

### 4.2 Main experiment matrix (3 methods x 2 samplers), 5 epochs

`train_balanced_and_unbalanced.py` trains ERM, loss re-weighting and Group DRO, each with
(a) standard shuffling and (b) group-balanced sampling.
Fixed hyperparameters: SGD, momentum 0.9, lr 1e-4, weight decay 1e-4, batch size 32, 5 epochs, Group DRO eta = 0.01.

```bash
python scripts/train_balanced_and_unbalanced.py --model resnet50      # -> results/
python scripts/train_balanced_and_unbalanced.py --model transformer   # -> results/vit/
```
Outputs per run: `<method>_<sampler>_results.json` (test metrics, plus `best_epoch` and the validation metrics of the selected epoch),
`<method>_<sampler>_history.json` (per-epoch train metrics, per-group losses, validation metrics), and
`balanced_and_unbalanced_summary.json`. The summary table is also printed to the terminal.

### 4.3 Group DRO hyperparameter sweep, 5 epochs

`train_sweep_group_dro.py` runs 14 Group DRO configurations (all with the balanced sampler):

* step size eta in {0.001, 0.01, 0.05, 0.1, 0.5, 1.0}
* weight decay in {1e-3, 1e-2, 1e-1} (eta = 0.01)
* generalisation adjustment C in {1, 2} (adjustment `C / sqrt(n_g)`) with weight decay in {1e-2, 1e-1}
* one run with lr = 1e-3, wd = 1e-3, C = 1

```bash
python scripts/train_sweep_group_dro.py --model resnet50         # -> results/group_dro_sweep/
python scripts/train_sweep_group_dro.py --model transformer      # -> results/vit/group_dro_sweep/
```
One `dro_*_results.json` / `dro_*_history.json` per run and `group_dro_results.json` (all runs). The table printed at the end is **ranked by validation worst-group accuracy**.

### 4.4 Single experiment

```bash
# ERM, standard shuffling
python scripts/train.py --method erm --epochs 5 --exp_name erm_unbalanced

# Loss re-weighting
python scripts/train.py --method reweight --epochs 5 --exp_name reweight_unbalanced

# Group DRO with the group-balanced sampler, ViT backbone, seed 1
python scripts/train.py --method group_dro --model transformer --balanced_sampler --eta 0.01 \
       --epochs 5 --seed 1 --exp_name vit/group_dro_balanced
```
Arguments: `--method {erm,reweight,group_dro}`, `--model {resnet50,transformer}`, `--balanced_sampler`, `--epochs`, `--learning_rate`,
`--weight_decay`, `--eta`, `--adj_coeff`, `--batch_size`, `--seed`, `--data_dir`, `--exp_name`.
Results: `results/<exp_name>_results.json` / `_history.json`; best-validation checkpoint: `checkpoints/<exp_name>_model.pth`.
With `--seed k` the name gets the suffix `_seed<k>` and Python, NumPy, PyTorch and the data loaders are seeded (cuDNN deterministic).
Without `--seed` nothing is seeded (this is how the runs reported in the paper were produced).

### 4.5 Model selection on the validation set

```bash
python scripts/select_by_val.py --model resnet50        # main 6 runs + sweep, ranked by validation worst-group acc
python scripts/select_by_val.py --model transformer
```
Prints Markdown tables with, for every run, the selected epoch, validation worst/avg accuracy and the test metrics.
Ties in validation worst-group accuracy are broken by validation average accuracy.

### 4.6 Re-evaluating a checkpoint

```bash
python scripts/eval.py --checkpoint checkpoints/erm_unbalanced_model.pth --model resnet50 --split test
python scripts/eval.py --checkpoint checkpoints/vit/erm_unbalanced_model.pth --model transformer --split test
```

### 4.7 Optional: repeated runs (mean ± std)

```bash
python scripts/train_balanced_and_unbalanced.py --model resnet50 --seeds 1 2
python scripts/aggregate_seeds.py --model resnet50                       # original run + seeds 1 and 2
python scripts/aggregate_seeds.py --model resnet50 --exclude_original    # only the seeded runs
```

### 4.8 Figures

```bash
python scripts/plot_tradeoff.py                                  # reports/figures/tradeoff_avg_vs_worst.png (both backbones)
python scripts/plot_results.py --model resnet50      
# <tag>_ablation_comparison.png + group_dro_sweep_top5_<tag>.png
python scripts/plot_results.py --model transformer
python scripts/plot_loss_curves.py --model resnet50  
# <tag>_<method>_<sampler>_loss_curves.png (6 figures per backbone)
python scripts/plot_loss_curves.py --model transformer
```
`<tag>` is `resnet50` or `vit`. Figures are saved in `reports/figures/`.
`plot_results.py` draws the 5 sweep configurations with the best validation worst-group accuracy.

## 5. Where each result of the report comes from

| Report item | Files |
|---|---|
| Table 1 (ResNet-50, 6 rows) | `results/{erm,reweight,group_dro}_{unbalanced,balanced}_results.json` |
| Table 1 (ViT-B/16, 6 rows) | `results/vit/{erm,reweight,group_dro}_{unbalanced,balanced}_results.json` |
| Table 2 (sweeps) | `results/group_dro_sweep/`, `results/vit/group_dro_sweep/` (validation columns: `python scripts/select_by_val.py`) |
| Trade-off figure | `python scripts/plot_tradeoff.py` |
| Per-group loss curves | `results/*_history.json`, field `group_<g>_loss` (`plot_loss_curves.py`) |
| Early 3-epoch runs (not in report) | `results_3_epochs/` |

## 6. Reference results (5 epochs, test set, single unseeded run)

| Backbone | Method | Sampler | Worst-group acc | Avg acc | Avg acc (train-weighted) |
|---|---|---|---|---|---|
| ResNet-50 | ERM | unbalanced | 8.88 | 73.33 | 93.27 |
| ResNet-50 | Reweight | unbalanced | 82.93 | 87.19 | 90.44 |
| ResNet-50 | Group DRO | unbalanced | 61.53 | 84.19 | 95.48 |
| ResNet-50 | ERM | balanced | 84.08 | 86.62 | 88.39 |
| ResNet-50 | Reweight | balanced | 74.90 | 85.14 | 77.33 |
| ResNet-50 | Group DRO | balanced | 86.74 | 88.92 | 90.74 |
| ViT-B/16 | ERM | unbalanced | 51.09 | 83.34 | 96.55 |
| ViT-B/16 | Reweight | unbalanced | 88.63 | 90.56 | 91.95 |
| ViT-B/16 | Group DRO | unbalanced | 82.35 | 89.44 | 95.67 |
| ViT-B/16 | ERM | balanced | 81.31 | 92.98 | 95.86 |
| ViT-B/16 | Reweight | balanced | 88.32 | 91.65 | 91.68 |
| ViT-B/16 | Group DRO | balanced | 84.42 | 90.99 | 94.99 |

**Reproducibility note.** The runs above were produced without fixing a seed (initialisation of the new classification head,
data order, augmentation and `WeightedRandomSampler` are random) and without deterministic cuDNN. Re-running will give numbers
close to, but not identical with, these. With 133 validation and 642 test images in the smallest groups, differences of a few
points in worst-group accuracy between two runs are expected (binomial standard error of about 1.4 points on 642 images for an accuracy near 85%,
not counting training randomness). Use `--seed` / `--seeds` (Sections 4.4 and 4.7) for controlled repetitions.

## 7. Implementation notes

* **ERM** - mean cross-entropy.
* **Reweight** - per-sample cross-entropy multiplied by `N / (4 * n_g)` (inverse group frequency, computed on the training set).
* **Group DRO** (`src/loss.py`) - per-group mean loss in the mini-batch, group weights `q` updated with
  `q_g <- q_g * exp(eta * (loss_g + C / sqrt(n_g)))`, renormalised, and the loss is `sum_g q_g * loss_g`
  (`q` is detached from the autograd graph). A group that is absent from a mini-batch contributes a loss of 0 to that update;
  with the unbalanced loader the smallest group (56 training images) is absent from about 69% of the batches of 32, which is why Group DRO is used with the balanced sampler.
* **Balanced sampler** - `WeightedRandomSampler` with weight `1 / n_g` per sample, sampling with replacement, one epoch = `len(train)` draws.
  Combined with *reweight* the inverse-frequency factor is applied twice (sampler and loss); `train.py` prints a note when this combination is used.
* **Data augmentation** - train: `RandomResizedCrop(224, scale=(0.7, 1))` + horizontal flip; eval: `Resize(256, 256)` + `CenterCrop(224)`; ImageNet mean/std normalisation.
* **Optimiser** - SGD with momentum 0.9 for all runs; all layers of the pretrained backbone are fine-tuned.

## 8. Sources and attribution

* Sagawa, Koh, Hashimoto, Liang. *Distributionally Robust Neural Networks for Group Shifts: On the Importance of Regularization for Worst-Case Generalization.* ICLR 2020. - Dataset, group definition, Group DRO objective, the exponentiated-gradient update of `q` and the `C / sqrt(n_g)` adjustment. Official code: https://github.com/kohpangwei/group_DRO. `src/loss.py` is our PyTorch re-implementation of that update rule; the group/split convention of `src/dataset.py` follows the dataset's `metadata.csv` as released by the authors.
* Geirhos et al. *Shortcut learning in deep neural networks.* Nature Machine Intelligence, 2020.
* Dataset sources: CUB-200-2011 (bird images) and Places (backgrounds), combined by Sagawa et al.
* Pretrained models and building blocks: `torchvision` (`resnet50`, `vit_b_16`), PyTorch, pandas, matplotlib, seaborn.
* Parts of the code were written with the assistance of generative AI tools (Gemini); all code has been reviewed and is understood by the authors.
