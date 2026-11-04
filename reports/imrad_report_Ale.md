# IMRaD Report Draft

---

## Section 1: Introduction

### 1.1 Shortcomings of Empirical Risk Minimization (ERM)
Standard deep neural networks trained via Empirical Risk Minimization (ERM) minimize expected loss over the training distribution $P_{\text{train}}(X, Y)$:

$$\mathcal{L}_{\text{ERM}}(\theta) = \mathbb{E}_{(x,y) \sim P_{\text{train}}} [\ell(f_\theta(x), y)]$$

While ERM achieves state-of-the-art average accuracy on independent and identically distributed (i.i.d.) test sets, it suffers from a fundamental structural vulnerability: **shortcut learning**. When dataset collection introduces predictive spurious correlations between non-causal background features $A$ (e.g., land vs. water environments) and target class labels $Y$ (e.g., landbird vs. waterbird), gradient descent prioritizes extracting high-frequency or low-complexity background cues rather than learning invariant, semantic foreground features. Because ERM treats every training sample with equal weight, the loss gradient is overwhelmingly dominated by majority groups, allowing the model to minimize training risk without forming robust representations for rare sub-populations.

### 1.2 Literature Background: Geirhos et al. & Sagawa et al.
The phenomenon of shortcut learning in vision architectures was systematically formalized by **Geirhos et al. (2020)**, who demonstrated that deep learning models display strong "texture and background biases." Rather than learning compositional semantic representations, deep convolutional neural networks (CNNs) exploit opportunistic decision rules—such as classifying objects based on diagnostic background textures—because these features offer cheaper paths to loss minimization during backpropagation.

To benchmark and mitigate shortcut dependencies under distribution shifts, **Sagawa et al. (2020)** introduced the **Waterbirds dataset** and established the distributionally robust optimization framework. By combining bird foreground images from the CUB-200-2011 dataset with background scenes from Places365, Sagawa et al. constructed an extreme confounder setup where Landbirds almost exclusively co-occur on Land backgrounds ($72.95\%$) and Waterbirds on Water backgrounds ($22.04\%$). Minority groups, such as Waterbirds on Land ($1.17\%$), comprise less than $2\%$ of training samples. When evaluated under standard ERM, ResNet architectures collapse on minority groups despite achieving high overall average accuracy.

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

### 4.1 Comparative Empirical Analysis: Architectural & Algorithmic Baselines
Our comprehensive evaluation across ResNet-50 and Vision Transformer (ViT-Base) backbones confirms the theoretical predictions of Geirhos et al. (2020) and Sagawa et al. (2020). As shown in Table 1, standard convolutional ERM on ResNet-50 achieves a high average accuracy of $76.96\%$, yet collapses on minority **Group 2 (Waterbird on Land)** with a worst-group accuracy of **$3.43\%$**. The model learns a spurious decision rule (*"If background == Land $\rightarrow$ Predict Landbird"*), classifying almost all landbirds correctly while failing on waterbirds in land environments.

#### Table 1: Comprehensive Benchmark Summary Across Backbones, Methods, and Samplers
| Architecture | Method | Sampler | Worst-Group Acc ($\text{Acc}_{\text{worst}}$) | Average Acc ($\text{Acc}_{\text{avg}}$) | Group 0 (L/L) | Group 1 (L/W) | Group 2 (W/L) | Group 3 (W/W) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **ResNet-50** | ERM Baseline | Unbalanced | **3.43%** (Group 2) | 76.96% | 99.73% | 77.25% | **3.43%** | 69.47% |
| **ResNet-50** | Loss Re-weighting | Unbalanced | **76.50%** (Group 1) | 83.38% | 86.74% | **76.50%** | 88.32% | 90.81% |
| **ResNet-50** | Group DRO ($\eta=0.01$) | Unbalanced | **49.84%** (Group 2) | 80.72% | 98.89% | 68.96% | **49.84%** | 89.10% |
| **ResNet-50** | Group DRO ($\eta=0.01$) | Balanced | **80.98%** (Group 1) | 85.93% | 89.67% | **80.98%** | 85.51% | 90.65% |
| **ViT-Base** | ERM Baseline | Unbalanced | **41.28%** (Group 2) | 83.66% | 99.73% | 77.69% | **41.28%** | 90.50% |
| **ViT-Base** | Loss Re-weighting | Unbalanced | **85.36%** (Group 2) | 92.56% | 96.19% | 91.80% | **85.36%** | 89.72% |
| **ViT-Base** | Group DRO ($\eta=0.01$) | Balanced | **86.76%** (Group 2) | 91.82% | 95.96% | 89.09% | **86.76%** | 91.90% |
| **ViT-Base** | Group DRO ($\text{Adj}=1.0$)* | Balanced | **86.29%** (Group 2) | **91.61%** | 95.12% | 89.89% | **86.29%** | 90.65% |

*\*Note: Optimal sweep configuration (`dro_adj_1_wd_1e-3_lr_1e-3`).*

#### Key Comparative Insights:
1. **Architectural Invariance (CNN vs. ViT):** Replacing the convolutional backbone with ViT-Base improves ERM worst-group accuracy from $3.43\%$ to **$41.28\%$**. The global self-attention mechanism in Transformers captures long-range spatial context, making the feature representation less reliant on local, diagnostic background textures. However, $41.28\%$ remains unacceptable for robust deployment, demonstrating that attention alone does not fully solve spurious bias.
2. **Loss Re-weighting Efficiency:** On ResNet-50, Group-Balanced Loss Re-weighting ($w_g \propto \frac{1}{N_g}$) boosts worst-group accuracy from **$3.43\%$ to $76.50\%$**. On ViT-Base, it reaches **$85.36\%$** worst-group accuracy while maintaining an overall average accuracy of **$92.56\%$**.
3. **Synergy Between Group DRO and Resampling:** Standard Group DRO without resampling (`group_dro_unbalanced`) yields only **$49.84\%$** worst-group accuracy on ResNet-50. Because Group 2 contains only $56$ training images ($1.17\%$), Group 2 samples rarely appear in standard mini-batches, preventing the exponentiated group weights $q$ from updating efficiently. Combining Group DRO with a `WeightedRandomSampler` (`group_dro_balanced`) resolves this bottleneck, elevating worst-group performance to **$80.98\%$** (ResNet-50) and **$86.76\%$** (ViT-Base).

---

### 4.2 Group DRO Sensitivity & Hyperparameter Sweeps
To characterize the optimization stability of Group DRO, we performed extensive sweeps across the step size parameter $\eta \in [0.001, 1.0]$, weight decay $WD \in [10^{-4}, 1.0]$, and generalization adjustment coefficients $\text{Adj} \in [0.0, 2.0]$.

1. **Step-Size ($\eta$) Instability:**
   Small step sizes ($\eta \in [0.001, 0.05]$) yield consistent, robust performance ($\sim 81.5\% - 83.8\%$ worst-group accuracy). However, setting $\eta = 0.5$ reduces worst-group accuracy to **$66.34\%$**, and $\eta = 1.0$ causes severe optimization collapse (**$37.83\%$** worst-group accuracy). High step sizes cause $q$-weights to fluctuate wildly between mini-batches rather than converging smoothly to the worst-performing subgroup.
2. **Generalization Adjustment Overshooting:**
   Adding a margins-based generalization penalty $\frac{C}{\sqrt{n_g}}$ with $\text{Adj}=1.0$, $WD=10^{-3}$, and $LR=10^{-3}$ yields the top overall result: **$91.61\%$ Average Accuracy** and **$86.29\%$ Worst-Group Accuracy**. However, setting $\text{Adj}=2.0$ without fine-tuned weight decay leads to training divergence ($0.00\%$ worst-group accuracy), confirming Sagawa et al.'s finding that over-penalizing rare subgroups causes gradient instability.

---

### 4.3 Mechanism of Mitigation & Per-Group Loss Convergence
Analyzing per-group loss curves across training epochs reveals the underlying optimization mechanics:

1. **Loss Divergence under ERM:**
   In standard ERM, loss for Group 0 decreases smoothly from $0.43$ to $0.15$. Conversely, loss for Group 2 **diverges continuously**, climbing from $1.00$ to $1.46$. Because Group 0 ($N_0=3,498$) overwhelmingly dominates the gradient sum relative to Group 2 ($N_2=56$), backpropagation continually minimizes global risk by exploiting the background shortcut at the direct expense of Group 2.
2. **Parallel Loss Minimization:**
   Under Loss Re-weighting and Group DRO with balanced sampling, sample gradient contributions are dynamically re-balanced. Consequently, losses across all four groups decrease in parallel (from $\sim 0.65-0.69$ down to $\sim 0.48-0.51$), forcing the network to unlearn shortcut reliance and extract invariant semantic features.

---

### 4.4 Performance Trade-offs & Broader Implications
Our findings highlight the inherent trade-off between standard average accuracy and worst-group robustness:

* **The Shortcut Tax:** Standard ERM achieves maximum accuracy on majority groups ($99.73\%$ on Group 0) by relying on background cues. Robust methods (Re-weighting and Group DRO) intentionally reduce majority group performance ($89.67\% - 96.19\%$) to guarantee equitable classification across out-of-distribution minority environments.
* **Requirements & Limitations:** Both Re-weighting and Group DRO depend on explicit subpopulation labels $g = 2y + a$ during training. In unannotated real-world datasets, obtaining these annotations can be costly or infeasible, necessitating future work into unsupervised spurious group discovery.

**Conclusion for Robust AI:**
Measuring model performance solely via average validation accuracy provides a false sense of reliability. Synthetic benchmarks like Waterbirds prove that high overall accuracy ($76.96\% - 83.66\%$) can hide near-complete failure on minority populations. Incorporating worst-group optimization objectives and group-aware data sampling is essential for deploying reliable, unbiased vision models in real-world environments.