import torch
import torch.nn as nn

class GroupDROLoss(nn.Module):
    """
    Group Distributionally Robust Optimization (Group DRO) Loss.
    Adapted from Sagawa et al. (2020), "Distributionally Robust Neural Networks for Group Shifts".
    q is the vector of group weights, updated dynamically during training.
    """
    def __init__(self, num_groups=4, eta=0.01, device="cuda"):
        super().__init__()
        self.num_groups = num_groups
        self.eta = eta
        self.criterion = nn.CrossEntropyLoss(reduction="none")

        # Initialize uniform group weights q = [0.25, 0.25, 0.25, 0.25]
        self.q = torch.ones(num_groups, device=device) / num_groups

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
            self.q = self.q * torch.exp(self.eta * group_losses.detach())
            q_sum = self.q.sum()
            if q_sum > 0:
                self.q = self.q / q_sum
            else:
                print("Warning: Reset group weights to uniform due to zero sum.")
                self.q = torch.ones(self.num_groups, device=outputs.device) / self.num_groups
            
        # 4. Weighted sum of group losses
        dro_loss = torch.dot(self.q, group_losses)
        
        return dro_loss, raw_loss