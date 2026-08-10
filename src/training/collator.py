import torch


class MultiLabelTokenCollator:
    def __init__(self, tokenizer, pad_to_multiple_of: int | None = None):
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of


    def __call__(self, features: list[dict]) -> dict:
        labels = [f["labels"] for f in features]
        features = [{k: v for k, v in f.items() if k != "labels"} for f in features]
 
        batch = self.tokenizer.pad(
            features,
            padding=True,
            return_tensors="pt",
            pad_to_multiple_of=self.pad_to_multiple_of,
        )
 
        max_len = batch["input_ids"].shape[1]
        n_classes = len(labels[0][0])
 
        padded_labels = []
        for lab in labels:
            lab = torch.as_tensor(lab, dtype=torch.float32)
            pad_len = max_len - lab.shape[0]
            if pad_len > 0:
                pad = torch.full((pad_len, n_classes), -100.0)
                lab = torch.cat([lab, pad], dim=0)
            padded_labels.append(lab)
 
        batch["labels"] = torch.stack(padded_labels)
        return batch