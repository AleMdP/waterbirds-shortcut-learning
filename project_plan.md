# Waterbirds Detection: Classifying a Minority

## Project Execution Plan & Task Division

### 1. Project Overview & Objectives

* **Core Problem:** Shortcut learning / Spurious correlation. Models trained with Empirical Risk Minimization (ERM) tend to rely on the background (water vs. land) rather than the bird species (waterbird vs. landbird) because background and bird type are strongly correlated in the training set.

* **Main Metric:** **Worst-Group Accuracy** (evaluating performance across the 4 distinct subgroups).

* **Secondary Metric:** **Average Accuracy** (evaluating overall performance).

* **Deliverable:** Analyze the trade-off between Average Accuracy and Worst-Group Accuracy using different methods.

#### The 4 Dataset Subgroups

1. **Landbird on Land** (*Majority group / Spurious correlation*)

2. **Waterbird on Water** (*Majority group / Spurious correlation*)

3. **Landbird on Water** (*Minority group / Out-of-distribution*)

4. **Waterbird on Land** (*Minority group / Out-of-distribution*)

### 2. Experimental Setup & Infrastructure

* **Server:** `Gorina4` Unix GPU server.

* **Target GPU:** NVIDIA Tesla P100 (16 GB) — `GPU ID: 1`.

* **CUDA Configuration:** Insert `os.environ["CUDA_VISIBLE_DEVICES"] = "1"` at the top of every entry script.

* **Dataset Location:** `~/waterbirds_project/data/waterbird_complete95_forest222/`

### 3. Work Division & Weekly Roadmap (50/50 Allocation)

#### **Roles Overview**

* **Partner A:** **Model Engineering & Training Pipeline Lead** (Focus: Dataset classes, loss functions, Group DRO, training scripts).

* **Partner B:** **Data Analysis, Baseline & Evaluation Lead** (Focus: Data exploration, resampling/re-weighting methods, visualization, metrics tracking).

#### **Week 1: Environment Setup, Data Pipeline & Baseline (ERM)**

* **Partner A:**

  * Initialize the GitHub repository structure (`src/`, `scripts/`, `notebooks/`, `.gitignore`).

  * Implement the PyTorch custom `WaterbirdsDataset` class parsing `metadata.csv` (handling `train`, `val`, `test` splits).

* **Partner B:**

  * Set up the python virtual environment on `Gorina4` (`torch`, `torchvision`, `pandas`, `matplotlib`, `seaborn`).

  * Create an exploratory data analysis script (`eda.py`) to report sample counts and proportions for each of the 4 groups.

* **Joint Deliverable:**

  * Train the baseline **ERM (Empirical Risk Minimization)** model using a pre-trained ResNet-50.

  * Compute and log initial test results for Average Accuracy vs. Worst-Group Accuracy.

#### **Week 2: Advanced Mitigations (Group DRO & Re-weighting)**

* **Partner A:**

  * Implement the **Group DRO (Distributionally Robust Optimization)** training logic and loss function (Sagawa et al., 2020).

  * Build a unified command-line interface (`train.py`) accepting parameters: `--method [erm|group_dro|reweight]`.

* **Partner B:**

  * Implement **Group-Balanced Sampling** / **Loss Re-weighting** in the data loading pipeline to serve as a strong secondary mitigation baseline.

  * Build the standardized evaluation pipeline (`eval.py`) to compute per-group accuracies automatically.

#### **Week 3: Hyperparameter Tuning, Experiments & Visualizations**

* **Partner A:**

  * Run hyperparameter sweeps (learning rate, weight decay, Group DRO adjustment step $\eta$).

  * Tabulate all experimental runs and results in clean Markdown/LaTeX format.

* **Partner B:**

  * Generate required plots and visualizations:

    * **Trade-off Plot:** Average Accuracy vs. Worst-Group Accuracy across all methods.

    * **Training Curves:** Convergence of per-group losses over epochs.

* **Joint Deliverable:**

  * Interpret results together. Analyze why improving worst-group performance may lead to a trade-off in overall average accuracy.

#### **Week 4: Final Report (IMRaD), GitHub & Oral Defense**

* **Partner A:**

  * Draft the **Methods & Materials** and **Results** sections of the 6-page report.

  * Finalize the `README.md` file ensuring 100% reproducibility.

* **Partner B:**

  * Draft the **Introduction** (literature background: Geirhos et al. & Sagawa et al.) and **Discussion** sections.

  * Create presentation slides for the mandatory oral presentation (Nov 11th).

* **Joint Deliverable:**

  * Conduct a thorough cross-review of the report (strict 6-page maximum excluding references).

  * Rehearse for the oral exam (Nov 24th) ensuring both partners can explain every line of code and design choice.

### 4. Report Structure (IMRaD Format - Max 6 Pages)

| IMRaD Section | Scope & Content | Lead Responsibility | 
| ----- | ----- | ----- | 
| **1. Introduction** | Shortcomings of ERM, definition of shortcut learning and background bias, formulation of worst-group accuracy objective. | Partner B | 
| **2. Materials & Methods** | Dataset details (4 subgroups), network architecture (ResNet-50), training details on Tesla P100, formal definitions of ERM, Re-weighting, and Group DRO. | Partner A | 
| **3. Results** | Tables showing per-group accuracies, worst-group accuracy, and average accuracy. Scatter/bar plots showing the accuracy trade-offs. | Partner A | 
| **4. Discussion** | Critical analysis of trade-offs, why spurious correlations occur, potential limitations, and broader implications for robust AI. | Partner B | 

### 5. Repository Structure

```
waterbirds_project/
├── data/                      # Excluded from Git (.gitignore)
│   └── waterbird_complete95_forest222/
├── src/
│   ├── dataset.py            # Waterbirds PyTorch Dataset
│   ├── models.py             # ResNet-50 backbone definition
│   └── loss.py               # Group DRO & Loss implementations
├── scripts/
│   ├── eda.py                # Exploratory Data Analysis
│   ├── train.py              # Main training script
│   └── eval.py               # Evaluation & metrics computation
├── notebooks/                # Visualization notebooks
├── README.md                 # Reproducibility guide
├── requirements.txt          # Python dependencies
└── .gitignore                # Git ignore rules

```