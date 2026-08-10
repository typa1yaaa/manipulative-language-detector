import torch
from torch import nn
from transformers import Trainer


class FocalLoss(nn.Module):
    def __init__(self, pos_weight: torch.Tensor, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.pos_weight = pos_weight
        self.gamma = gamma
        self.reduction = reduction
 
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction="none", pos_weight=self.pos_weight.to(logits.device)
        )
        probs = torch.sigmoid(logits)
        pt = probs * targets + (1 - probs) * (1 - targets)
        loss = ((1 - pt) ** self.gamma) * bce
 
        if self.reduction == "mean":
            return loss.mean()
        return loss.sum()


class MultiLabelTrainer(Trainer):
    def __init__(self, *args, pos_weight: torch.Tensor, focal_gamma: float = 2.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_fct = FocalLoss(pos_weight=pos_weight, gamma=focal_gamma)
 
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
 
        mask = (labels != -100).any(dim=-1)
        logits = logits[mask]
        labels = labels[mask]
 
        loss = self.loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss