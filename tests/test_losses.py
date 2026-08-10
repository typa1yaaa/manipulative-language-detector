from types import SimpleNamespace
import pytest
import torch
import torch.nn.functional as F

from src.training.losses import FocalLoss, MultiLabelTrainer


def test_gamma_zero_reduces_to_plain_weighted_bce():
    logits = torch.tensor([2.0, -2.0, 0.5])
    targets = torch.tensor([1.0, 0.0, 1.0])
    pos_weight = torch.tensor([1.5, 1.5, 1.5])
 
    focal = FocalLoss(pos_weight=pos_weight, gamma=0.0)
    focal_loss_value = focal(logits, targets)
 
    reference_loss = F.binary_cross_entropy_with_logits(
        logits, targets, pos_weight=pos_weight, reduction="mean"
    )
 
    assert torch.isclose(focal_loss_value, reference_loss)

 
def test_worse_prediction_gives_higher_loss():
    pos_weight = torch.tensor([1.0])
    focal = FocalLoss(pos_weight=pos_weight, gamma=2.0)
 
    loss_good_prediction = focal(torch.tensor([2.0]), torch.tensor([1.0]))
    loss_bad_prediction = focal(torch.tensor([-2.0]), torch.tensor([1.0]))
 
    assert loss_bad_prediction > loss_good_prediction


def test_higher_pos_weight_increases_loss_on_missed_positive():
    logits = torch.tensor([-2.0])   # модель уверенно (и неправильно) говорит "негатив"
    targets = torch.tensor([1.0])   # а на самом деле это позитив
 
    loss_low_weight = FocalLoss(pos_weight=torch.tensor([1.0]), gamma=0.0)(logits, targets)
    loss_high_weight = FocalLoss(pos_weight=torch.tensor([10.0]), gamma=0.0)(logits, targets)
 
    assert loss_high_weight > loss_low_weight


class FakeModel:
    def __init__(self, logits: torch.Tensor):
        self.logits = logits
 
    def __call__(self, **kwargs):
        return SimpleNamespace(logits=self.logits)


@pytest.fixture
def trainer():
    t = MultiLabelTrainer.__new__(MultiLabelTrainer)
    t.loss_fct = FocalLoss(pos_weight=torch.tensor([1.0, 1.0]), gamma=2.0)
    return t


def test_padding_token_excluded_regardless_of_its_logit_values(trainer):
    labels = torch.tensor([[[1.0, 0.0], [-100.0, -100.0]]])  # токен1 = паддинг
 
    logits_mild_garbage = torch.tensor([[[2.0, -2.0], [0.0, 0.0]]])
    logits_extreme_garbage = torch.tensor([[[2.0, -2.0], [999.0, -999.0]]])
 
    loss_mild = trainer.compute_loss(FakeModel(logits_mild_garbage), {"labels": labels.clone()})
    loss_extreme = trainer.compute_loss(FakeModel(logits_extreme_garbage), {"labels": labels.clone()})
 
    assert torch.isclose(loss_mild, loss_extreme)


def test_loss_uses_only_non_padded_tokens(trainer):
    real_logits = torch.tensor([[2.0, -2.0]])
    real_labels = torch.tensor([[1.0, 0.0]])
 
    loss_without_padding = trainer.compute_loss(
        FakeModel(real_logits.unsqueeze(0)), {"labels": real_labels.unsqueeze(0)}
    )
 
    padded_logits = torch.tensor([[[2.0, -2.0], [123.0, -456.0]]])
    padded_labels = torch.tensor([[[1.0, 0.0], [-100.0, -100.0]]])
    loss_with_padding = trainer.compute_loss(FakeModel(padded_logits), {"labels": padded_labels})
 
    assert torch.isclose(loss_without_padding, loss_with_padding)