import pytest
import torch

from src.training.collator import MultiLabelTokenCollator


class FakeTokenizer:
    def pad(self, features, padding, return_tensors, pad_to_multiple_of=None):
        max_len = max(len(f["input_ids"]) for f in features)
        input_ids, attention_mask = [], []
        for f in features:
            pad_len = max_len - len(f["input_ids"])
            input_ids.append(f["input_ids"] + [0] * pad_len)
            attention_mask.append(f["attention_mask"] + [0] * pad_len)
        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention_mask),
        }


@pytest.fixture
def collator():
    return MultiLabelTokenCollator(tokenizer=FakeTokenizer())


def test_pads_labels_to_max_length_in_batch(collator):
    n_classes = 3
    features = [
        {
            "input_ids": [101, 202, 102],       
            "attention_mask": [1, 1, 1],
            "labels": [[0, 0, 1], [1, 0, 0], [0, 1, 0]],
        },
        {
            "input_ids": [101, 202, 303, 102],
            "attention_mask": [1, 1, 1, 1],
            "labels": [[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]],
        },
    ]
 
    batch = collator(features)

    assert batch["labels"].shape == (2, 4, n_classes)
 
    assert torch.equal(batch["labels"][0, 3], torch.full((n_classes,), -100.0))
 
    assert torch.equal(batch["labels"][1, 3], torch.tensor([0.0, 0.0, 0.0]))


def test_no_padding_needed_when_lengths_equal(collator):
    features = [
        {"input_ids": [101, 102], "attention_mask": [1, 1], "labels": [[1, 0], [0, 1]]},
        {"input_ids": [101, 102], "attention_mask": [1, 1], "labels": [[0, 1], [1, 0]]},
    ]
 
    batch = collator(features)
 
    assert batch["labels"].shape == (2, 2, 2)

    assert not torch.any(batch["labels"] == -100)


def test_single_example_batch(collator):
    features = [
        {"input_ids": [101, 202, 102], "attention_mask": [1, 1, 1], "labels": [[1, 0], [0, 1], [0, 0]]},
    ]
 
    batch = collator(features)
 
    assert batch["labels"].shape == (1, 3, 2)