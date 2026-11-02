# IMRaD Report Draft: Partner B Sections

---

## Section 1: Introduction

### 1.1 Shortcomings of Empirical Risk Minimization (ERM)
Standard deep neural networks trained via Empirical Risk Minimization (ERM) minimize expected loss over the training distribution $P_{\text{train}}(X, Y)$:

$$\mathcal{L}_{\text{ERM}}(\theta) = \mathbb{E}_{(x,y) \sim P_{\text{train}}} [\ell(f_\theta(x), y)]$$

While ERM achieves state-of-the-art average accuracy on independent and identically distributed (i.i.d.) test sets, it suffers from a fundamental structural vulnerability: **shortcut learning**. When dataset collection introduces predictive spurious correlations between non-causal background features $A$ (e.g., land vs. water environments) and target class labels $Y$ (e.g., landbird vs. waterbird), gradient descent prioritizes extracting high-frequency or low-complexity background cues rather than learning invariant, semantic foreground features. Because ERM treats every training sample with equal weight, the loss gradient is overwhelmingly dominated by majority groups, allowing the model to minimize training risk without forming robust representations for rare sub-populations.

### 1.2 Literature Background: Geirhos et al. & Sagawa et al.
The phenomenon of shortcut learning in vision architectures was systematically formalized by **Geirhos et al. (2020)**, who demonstrated that deep learning models display strong "texture and background biases." Rather than learning compositional semantic representations, deep convolutional neural networks (CNNs) exploit opportunistic decision rules—such as classifying objects based on diagnostic background textures—because these features offer cheaper paths to loss minimization during backpropagation.

To benchmark and mitigate shortcut dependencies under distribution shifts, **Sagawa et al. (2020)** introduced the **Waterbirds dataset** and established the distributionally robust optimization framework. By combining bird foregorund images from the CUB-200-2011 dataset with background scenes from Places365, Sagawa et al. constructed an extreme confounder setup where Landbirds almost exclusively co-occur on Land backgrounds ($72.95\%$) and Waterbirds on Water backgrounds ($22.04\%$). Minority groups, such as Waterbirds on Land ($1.17\%$), comprise less than $2\%$ of training samples. When evaluated under standard ERM, ResNet architectures collapse on minority groups despite achieving high overall average accuracy.

### 1.3 Subgroup Formulation & Worst-Group Accuracy Objective
To formally analyze background bias, we partition the dataset into four mutually exclusive subgroups $g \in \mathcal{G} = \{0, 1, 2, 3\}$, defined by the tuple of class label $y \in \{0: \text{Landbird}, 1: \text{Waterbird}\}$ and background environment $a \in \{0: \text{Land}, 1: \text{Water}\}$ via $g = 2y + a$:

1. **Group 0 ($y=0, a=0$):** Landbird on Land (*Majority / Spurious*, $N_{\text{train}} = 3,498$)
2. **Group 1 ($y=0, a=1$):** Landbird on Water (*Minority / Out-of-Distribution*, $N_{\text{train}} = 184$)
3. **Group 2 ($y=1, a=0$):** Waterbird on Land (*Minority / Out-of-Distribution*, $N_{\text{train}} = 56$)
4. **Group 3 ($y=1, a=1$):** Waterbird on Water (*Majority / Spurious*, $N_{\text{train}} = 1,057$)

Because average accuracy masks catastrophic failures on minority sub-populations, the primary objective function evaluated in this work is **Worst-Group Accuracy ($\text{Acc}_{\text{worst}}$)**:

$$\text{Acc}_{\text{worst}}(f_\theta) = \min_{g \in \mathcal{G}} \mathbb{E}_{(x,y) \sim P_g} [\mathbb{I}(f_\theta(x) = y)]$$

Where $P_g$ denotes the conditional data distribution for subgroup $g$. The secondary metric remains **Average Accuracy ($\text{Acc}_{\text{avg}}$)** to evaluate general classification capabilities and measure accuracy trade-offs.

---

## Section 4: Discussion

### 4.1 Comparative Empirical Analysis: ERM vs. Group-Reweighted Baseline
Our empirical evaluations on the Waterbirds test set confirm the theoretical predictions of Geirhos et al. (2020) and Sagawa et al. (2020). Standard ERM optimization yields a high average accuracy of $76.51\%$, yet collapses on minority **Group 2 (Waterbird on Land)** with a worst-group accuracy of **$3.89\%$**. The model learns a spurious decision rule: *"If background == Land $\rightarrow$ Predict Landbird"*, leading to $96.11\%$ misclassification when a Waterbird appears on land.

Implementing **Group-Balanced Loss Re-weighting** dramatically mitigates this spurious dependence. By weighting sample loss inversely proportional to group counts ($w_g \propto \frac{1}{N_g}$), Group 2 samples receive a weight multiplier of **$21.41$** compared to **$0.34$** for Group 0. As shown in Table 1, Group-Reweighted training increases worst-group accuracy from **$3.89\%$ to $81.20\%$** ($+77.31\%$ absolute gain), while simultaneously raising overall average accuracy from **$76.51\%$ to $86.12\%$**.

#### Table 1: Performance Summary on Test Set (ResNet-50, 3 Epochs)
| Method | Worst-Group Accuracy ($\text{Acc}_{\text{worst}}$) | Average Accuracy ($\text{Acc}_{\text{avg}}$) | Group 0 (Land/Land) | Group 1 (Land/Water) | Group 2 (Water/Land) | Group 3 (Water/Water) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Standard ERM** | **3.89%** (Group 2) | 76.51% | 99.73% | 74.50% | **3.89%** | 74.61% |
| **Group-Reweighted** | **81.20%** (Group 1) | **86.12%** | 91.28% | 88.59% | **85.71%** | 87.51% |

### 4.2 Mechanism of Mitigation & Per-Group Loss Divergence
The per-group loss convergence curves recorded across training epochs provide direct insight into optimization dynamics under spurious correlations:

1. **Loss Divergence in Standard ERM:**
   Under ERM, cross-entropy loss for Group 0 decreases smoothly from $0.43$ to $0.15$. However, loss for Group 2 **diverges continuously**, climbing from $1.00$ in Epoch 1 to $1.46$ in Epoch 3. Because Group 0 contains $3,498$ samples and Group 2 contains only $56$ samples, standard backpropagation reduces overall risk by strengthening the background shortcut, actively worsening predictions on Group 2.

2. **Equalized Gradient Optimization in Group-Reweighted:**
   Applying group inverse-frequency weights normalizes total gradient contributions across subgroups ($\sum_{i \in g} w_g \nabla_\theta \ell_i \approx \text{constant}$). Consequently, losses across all four groups converge in parallel from $\sim 0.65-0.69$ down to $\sim 0.48-0.51$, forcing the feature extractor to abandon background shortcuts and learn genuine anatomical bird features.

### 4.3 Performance Trade-offs & Limitations
While loss re-weighting dramatically improves worst-group performance, it induces a slight reduction in Group 0 majority accuracy ($99.73\%$ in ERM vs. $91.28\%$ in Reweighted). This drop illustrates the fundamental trade-off in subpopulation robustness: ERM exploits background shortcuts to boost majority accuracy, whereas robust algorithms sacrifice shortcut reliance to ensure uniform performance across all environments.

**Limitations:**
1. **Requirement of Subgroup Annotations:** Group re-weighting requires explicit background and class annotations ($g = 2y + a$) during training, which may not be available in unannotated real-world datasets.
2. **Batch Variance:** In minority groups with extremely low counts ($N_2 = 56$), high sample weights ($21.41$) introduce variance into stochastic gradient estimates, requiring careful momentum tuning or gradient clipping.

### 4.4 Broader Implications for Robust AI
These results demonstrate that optimizing for average accuracy is insufficient for real-world deployment in high-stakes domains (e.g., medical imaging, autonomous driving, and algorithmic fair lending). Models that perform exceptionally well under standard validation splits can fail catastrophically under subtle environmental shifts. Methodologies that explicitly optimize for distributionally robust objectives or subgroup balance are critical steps toward building trustworthy, invariant artificial intelligence.