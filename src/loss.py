""" 
PyTorch re-implementation of the group DRO update rule of Sagawa et al. (2020), Algorithm 1. Official code: github.com/kohpangwei/group_DRO.
"""

import torch
import torch.nn as nn

class GroupDROLoss(nn.Module):
    """
    Group Distributionally Robust Optimization (Group DRO) Loss.
    Adapted from Sagawa et al. (2020), "Distributionally Robust Neural Networks for Group Shifts".
    q is the vector of group weights, updated dynamically during training.
    """
    def __init__(self, num_groups=4, eta=0.01, group_counts=None, adj_coeff=0.0, device="cuda"):
        super().__init__()
        self.num_groups = num_groups
        self.eta = eta
        self.criterion = nn.CrossEntropyLoss(reduction="none")

        # Initialize uniform group weights q = [0.25, 0.25, 0.25, 0.25]
        self.q = torch.ones(num_groups, device=device) / num_groups

        # Optional generalization adjustment C / sqrt(n_g) from Sagawa et al. (Eq. 3)
        if group_counts is not None and adj_coeff > 0.0:
            counts = group_counts.to(device).float()
            self.adj = adj_coeff / torch.sqrt(counts)
        else:
            self.adj = torch.zeros(num_groups, device=device)

    def forward(self, outputs, targets, groups):
        # 1. Per-sample cross-entropy loss (shape: [batch_size])
        raw_loss = self.criterion(outputs, targets)
        
        # 2. Compute mean loss for each group present in this batch
        group_losses = torch.zeros(self.num_groups, device=outputs.device)
        for g in range(self.num_groups):
            mask = (groups == g)
            if mask.sum() > 0:
                group_losses[g] = raw_loss[mask].mean()
                
        # 3. Update group weights q dynamically (detached from autograd)
        with torch.no_grad():
            adjusted_losses = group_losses.detach() + self.adj
            self.q = self.q * torch.exp(self.eta * adjusted_losses)
            q_sum = self.q.sum()
            if q_sum > 0:
                self.q = self.q / q_sum
            else:
                print("Warning: Reset group weights to uniform due to zero sum.")
                self.q = torch.ones(self.num_groups, device=outputs.device) / self.num_groups
            
        # 4. Weighted sum of group losses
        dro_loss = torch.dot(self.q, group_losses)
        
        return dro_loss, raw_loss