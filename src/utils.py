import torch
import torch.nn as nn

class GroupLogger:
    def __init__(self, num_groups=4):
        self.num_groups = num_groups
        self.reset()
        self.criterion = nn.CrossEntropyLoss(reduction="none")

    def reset(self):
        self.group_correct = torch.zeros(self.num_groups)
        self.group_total = torch.zeros(self.num_groups)
        self.group_loss_sum = torch.zeros(self.num_groups)

    def update(self, outputs, targets, groups, raw_losses=None):
        preds = outputs.argmax(dim=1)
        correct = (preds == targets).float()
        
        if raw_losses is None:
            raw_losses = self.criterion(outputs, targets)

        for g in range(self.num_groups):
            mask = (groups == g)
            count = mask.sum().item()
            if count > 0:
                self.group_total[g] += count
                self.group_correct[g] += correct[mask].sum().item()
                self.group_loss_sum[g] += raw_losses[mask].sum().item()

    def get_metrics(self):
        group_accs = {}
        group_losses = {}
        for g in range(self.num_groups):
            count = self.group_total[g].item()
            group_accs[f"group_{g}_acc"] = (self.group_correct[g] / self.group_total[g]).item() if count > 0 else 0.0
            group_losses[f"group_{g}_loss"] = (self.group_loss_sum[g] / self.group_total[g]).item() if count > 0 else 0.0
            group_accs[f"group_{g}_count"] = int(count)

        acc_tensor = torch.tensor([group_accs[f"group_{g}_acc"] for g in range(self.num_groups)])
        worst_group_id = torch.argmin(acc_tensor).item()
        
        total_samples = self.group_total.sum().item()
        avg_acc = (self.group_correct.sum() / total_samples).item() if total_samples > 0 else 0.0

        return {
            "avg_acc": avg_acc,
            "worst_group_acc": acc_tensor[worst_group_id].item(),
            "worst_group_id": worst_group_id,
            **group_accs,
            **group_losses
        }