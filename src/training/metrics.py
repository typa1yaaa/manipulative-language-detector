import numpy as np
from sklearn.metrics import precision_recall_fscore_support
 
from . import config


def compute_metrics(eval_pred, threshold: float = config.EVAL_METRIC_THRESHOLD) -> dict:
    logits, labels = eval_pred
    probs = 1 / (1 + np.exp(-logits))
 
    valid_mask = labels[:, :, 0] != -100
    probs, labels = probs[valid_mask], labels[valid_mask]
 
    preds = (probs > threshold).astype(int)
 
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )
    p_micro, r_micro, f1_micro, _ = precision_recall_fscore_support(
        labels, preds, average="micro", zero_division=0
    )
 
    return {
        "precision_macro": p_macro, "recall_macro": r_macro, "f1_macro": f1_macro,
        "precision_micro": p_micro, "recall_micro": r_micro, "f1_micro": f1_micro,
    }