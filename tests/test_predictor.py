import pytest
import torch

from src.inference.predictor import ManipulationDetector


@pytest.fixture
def detector():
    obj = ManipulationDetector.__new__(ManipulationDetector)
    obj.id2label = {0: "fear_uncertainty_pressure", 1: "authority_claim_pressure"}
    obj.threshold = 0.5
    return obj


def _probs(rows: list[list[float]]) -> torch.Tensor:
    return torch.tensor(rows, dtype=torch.float32)


def test_wordpiece_pieces_join_into_full_word_text(detector):
    tokens = ["[CLS]", "интер", "##нет", "[SEP]"]
    probs = _probs([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
 
    words, word_labels, word_probs = detector._words_from_tokens(tokens, probs)
 
    assert words == ["интернет"]


def test_label_on_continuation_piece_is_lost():
    detector = ManipulationDetector.__new__(ManipulationDetector)
    detector.id2label = {0: "fear_uncertainty_pressure"}
    detector.threshold = 0.5
 
    tokens = ["[CLS]", "интер", "##нет", "[SEP]"]
    probs = _probs([[0.0], [0.0], [0.9], [0.0]])
 
    words, word_labels, word_probs = detector._words_from_tokens(tokens, probs)
 
    assert words == ["интернет"]
    assert word_labels == [[]]


def test_label_on_first_piece_is_kept_for_whole_word():
    detector = ManipulationDetector.__new__(ManipulationDetector)
    detector.id2label = {0: "fear_uncertainty_pressure"}
    detector.threshold = 0.5
 
    tokens = ["[CLS]", "интер", "##нет", "[SEP]"]
    probs = _probs([[0.0], [0.9], [0.0], [0.0]])
 
    words, word_labels, word_probs = detector._words_from_tokens(tokens, probs)
 
    assert words == ["интернет"]
    assert word_labels == [["fear_uncertainty_pressure"]]


def test_special_tokens_are_excluded_from_words(detector):
    tokens = ["[CLS]", "привет", "[SEP]"]
    probs = _probs([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
 
    words, word_labels, word_probs = detector._words_from_tokens(tokens, probs)
 
    assert words == ["привет"]
    assert "[CLS]" not in words and "[SEP]" not in words


def test_no_active_labels_gives_empty_lists_not_missing_word(detector):
    tokens = ["[CLS]", "обычный", "текст", "[SEP]"]
    probs = _probs([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
 
    words, word_labels, word_probs = detector._words_from_tokens(tokens, probs)
 
    assert words == ["обычный", "текст"]
    assert word_labels == [[], []]


def test_multiple_labels_active_on_same_token(detector):
    tokens = ["[CLS]", "срочно", "[SEP]"]
    probs = _probs([[0.0, 0.0], [0.9, 0.8], [0.0, 0.0]])
 
    words, word_labels, word_probs = detector._words_from_tokens(tokens, probs)
 
    assert words == ["срочно"]
    assert set(word_labels[0]) == {"fear_uncertainty_pressure", "authority_claim_pressure"}