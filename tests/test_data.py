import numpy as np
import pytest
import torch

from src.training.data import compute_pos_weight


def test_rarer_class_gets_higher_weight():
    train_dataset = [
        {"labels": [[1, 0], [1, 0]]},
        {"labels": [[1, 1], [0, 0]]},
    ]

    pos_weight = compute_pos_weight(train_dataset, num_classes=2)

    assert pos_weight[1] > pos_weight[0]


def test_pos_weight_is_clipped_to_config_range():
    train_dataset = [{"labels": [[1, 0]] * 60}]

    pos_weight = compute_pos_weight(train_dataset, num_classes=2)

    assert pos_weight[1] == pytest.approx(50.0)
    assert torch.all(pos_weight >= 1.0)
    assert torch.all(pos_weight <= 50.0)


def test_padding_tokens_do_not_affect_class_frequency():
    no_padding = [{"labels": [[1, 0], [0, 1]]}]
    with_padding = [{"labels": [[1, 0], [0, 1], [-100, -100]]}]

    pos_weight_no_pad = compute_pos_weight(no_padding, num_classes=2)
    pos_weight_with_pad = compute_pos_weight(with_padding, num_classes=2)

    assert torch.allclose(pos_weight_no_pad, pos_weight_with_pad)

