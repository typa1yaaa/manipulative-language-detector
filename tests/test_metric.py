import numpy as np
import pytest

from src.training.metrics import compute_metrics


def _logit(prob: float) -> float:
    prob = np.clip(prob, 1e-6, 1 - 1e-6)
    return float(np.log(prob / (1 - prob)))


def test_perfect_predictions_give_f1_of_one():
    logits = np.array([[[_logit(0.9), _logit(0.1)], [_logit(0.1), _logit(0.9)]]])
    labels = np.array([[[1.0, 0.0], [0.0, 1.0]]])
 
    metrics = compute_metrics((logits, labels), threshold=0.5)
 
    assert metrics["f1_macro"] == pytest.approx(1.0)
    assert metrics["f1_micro"] == pytest.approx(1.0)


def test_completely_wrong_predictions_give_f1_of_zero():
    logits = np.array([[[_logit(0.1), _logit(0.9)], [_logit(0.9), _logit(0.1)]]])
    labels = np.array([[[1.0, 0.0], [0.0, 1.0]]])
 
    metrics = compute_metrics((logits, labels), threshold=0.5)
 
    assert metrics["f1_macro"] == pytest.approx(0.0)


def test_completely_wrong_predictions_give_f1_of_zero():
    logits = np.array([[[_logit(0.1), _logit(0.9)], [_logit(0.9), _logit(0.1)]]])
    labels = np.array([[[1.0, 0.0], [0.0, 1.0]]])
 
    metrics = compute_metrics((logits, labels), threshold=0.5)
 
    assert metrics["f1_macro"] == pytest.approx(0.0)


def test_padding_tokens_are_excluded_from_metrics():
    logits_no_pad = np.array([[[_logit(0.9), _logit(0.1)]]])
    labels_no_pad = np.array([[[1.0, 0.0]]])
    metrics_no_pad = compute_metrics((logits_no_pad, labels_no_pad), threshold=0.5)
 
    # тот же самый пример + один паддинговый токен с меткой -100 в конце
    logits_with_pad = np.array([[[_logit(0.9), _logit(0.1)], [_logit(0.5), _logit(0.5)]]])
    labels_with_pad = np.array([[[1.0, 0.0], [-100.0, -100.0]]])
    metrics_with_pad = compute_metrics((logits_with_pad, labels_with_pad), threshold=0.5)
 
    assert metrics_no_pad == pytest.approx(metrics_with_pad)


 
@pytest.mark.parametrize(
    "threshold, expected_f1_macro",
    [
        (0.1, 2 / 3),  # низкий порог: у обоих классов появляются лишние ложные срабатывания
        (0.5, 1.0),    # порог ровно посередине между 0.4 и 0.6 -> предсказания точные
        (0.9, 0.0),    # высокий порог: обе настоящие метки "не проходят" порог уверенности
    ],
)
def test_threshold_changes_predictions(threshold, expected_f1_macro):
    logits = np.array([[[_logit(0.6), _logit(0.4)], [_logit(0.4), _logit(0.6)]]])
    labels = np.array([[[1.0, 0.0], [0.0, 1.0]]])
 
    metrics = compute_metrics((logits, labels), threshold=threshold)
 
    assert metrics["f1_macro"] == pytest.approx(expected_f1_macro)


def test_macro_f1_zero_division_quirk_documented():
    logits = np.array([[[_logit(0.9), _logit(0.1)]]])
    labels = np.array([[[1.0, 0.0]]])
 
    metrics = compute_metrics((logits, labels), threshold=0.5)
 
    assert metrics["f1_macro"] == pytest.approx(0.5)