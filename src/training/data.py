from pathlib import Path
from typing import Tuple
import numpy as np
import torch
from datasets import DatasetDict, load_from_disk
 
from . import config


def load_dataset(dataset_path: str | Path | None = None) -> DatasetDict:
    path = dataset_path or config.DATASET_PATH
    ds = load_from_disk(path)
    ds.set_format(type=None)
    return ds


def get_num_classes(ds: DatasetDict) -> int:
    example_labels = ds["train"][0]["labels"]
    return int(np.array(example_labels).shape[1])


def compute_pos_weight(
    train_dataset,
    num_classes: int,
    clip: Tuple[float, float] = config.POS_WEIGHT_CLIP,
) -> torch.Tensor:
    counts = np.zeros(num_classes)
    total_tokens = 0
 
    for example in train_dataset:
        labels = np.array(example["labels"])
        valid_mask = ~(labels == -100).all(axis=1)
        labels = labels[valid_mask]
        total_tokens += labels.shape[0]
        counts += labels.sum(axis=0)
 
    pos = np.maximum(counts, 1)
    neg = np.maximum(total_tokens - pos, 1)
    pos_weight_np = np.clip(neg / pos, clip[0], clip[1])
    return torch.tensor(pos_weight_np, dtype=torch.float32)