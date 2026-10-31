import torch

class GroupLogger:
    """Tracks accuracy and loss per group to calculate Worst-Group Accuracy."""
    def __init__(self, num_groups=4):
        self.num_groups = num_groups
        self.reset()

    def reset(self):
        self.group_correct = torch.zeros(self.num_groups)
        self.group_total = torch.zeros(self.num_groups)
        self.group_loss = torch.zeros(self.num_groups)

    def update(self, outputs, targets, groups, loss_fn=None):
        preds = torch.argmax(outputs, dim=1)
        correct = (preds == targets).float()
        
        if loss_fn is not None:
            losses = loss_fn(outputs, targets)
        else:
            losses = torch.zeros_like(correct)

        for g in range(self.num_groups):
            mask = (groups == g)
            if mask.sum() > 0:
                self.group_correct[g] += correct[mask].sum().item()
                self.group_total[g] += mask.sum().item()
                self.group_loss[g] += losses[mask].sum().item()

    def get_metrics(self):
        # Calculate group accuracies (avoid division by zero)
        accs = torch.where(
            self.group_total > 0,
            self.group_correct / self.group_total,
            torch.tensor(0.0)
        )
        
        avg_acc = self.group_correct.sum() / self.group_total.sum()
        worst_group_acc = torch.min(accs).item()
        worst_group_id = torch.argmin(accs).item()

        group_acc_dict = {f"group_{g}_acc": accs[g].item() for g in range(self.num_groups)}
        group_counts_dict = {f"group_{g}_count": int(self.group_total[g].item()) for g in range(self.num_groups)}

        return {
            "avg_acc": avg_acc.item(),
            "worst_group_acc": worst_group_acc,
            "worst_group_id": worst_group_id,
            **group_acc_dict,
            **group_counts_dict
        }