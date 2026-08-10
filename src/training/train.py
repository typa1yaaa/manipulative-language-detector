# python -m src.training.train
# python -m src.training.train --epochs 5 --output-dir ./artifacts/my_run

import argparse
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    TrainingArguments,
)

from . import config
from .collator import MultiLabelTokenCollator
from .data import compute_pos_weight, get_num_classes, load_dataset
from .losses import MultiLabelTrainer
from .metrics import compute_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Обучение мультилейбл-детектора манипуляций")
    parser.add_argument("--dataset-path", default=str(config.DATASET_PATH))
    parser.add_argument("--output-dir", default=str(config.MODEL_OUTPUT_DIR))
    parser.add_argument("--checkpoints-dir", default=str(config.CHECKPOINTS_DIR))
    parser.add_argument("--epochs", type=int, default=config.TRAINING_HYPERPARAMS["num_train_epochs"])
    return parser.parse_args()


def train(args=None):
    args = args or parse_args()
 
    ds = load_dataset(args.dataset_path)
    num_classes = get_num_classes(ds)
 
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    model = AutoModelForTokenClassification.from_pretrained(
        config.MODEL_NAME,
        num_labels=num_classes,
        id2label=config.ID2LABEL,
        label2id=config.LABEL2ID,
    )
 
    data_collator = MultiLabelTokenCollator(tokenizer)
    pos_weight = compute_pos_weight(ds["train"], num_classes)
    print("Веса классов (pos_weight):", pos_weight)
 
    hyperparams = dict(config.TRAINING_HYPERPARAMS)
    hyperparams["num_train_epochs"] = args.epochs
 
    training_args = TrainingArguments(output_dir=args.checkpoints_dir, **hyperparams)
 
    trainer = MultiLabelTrainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        compute_metrics=compute_metrics,
        data_collator=data_collator,
        pos_weight=pos_weight,
        focal_gamma=config.FOCAL_LOSS_GAMMA,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=config.EARLY_STOPPING_PATIENCE)],
    )
 
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"\nМодель сохранена в {args.output_dir}")
 
    test_metrics = trainer.evaluate(ds["test"])
    print("\nМетрики на test:", test_metrics)
    return trainer, test_metrics
 
 
if __name__ == "__main__":
    train()
