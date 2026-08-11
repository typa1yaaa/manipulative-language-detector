from pathlib import Path
import os

MODEL_NAME = "DeepPavlov/rubert-base-cased"

LABEL_NAMES = [
    "emotional_triggering_language",
    "fear_uncertainty_pressure",
    "doubt_uncertainty_injection",
    "cognitive_closure_cliches",
    "social_proof_pressure",
    "gain_loss_exaggeration",
    "causal_fact_distortion",
    "authority_claim_pressure",
    "topic_shift_misrepresentation",
    "directive_action_pressure",
]

LABEL_RU = {
    "emotional_triggering_language": "эмоционально нагруженная лексика",
    "fear_uncertainty_pressure": "страх / давление неопределённостью",
    "doubt_uncertainty_injection": "вброс сомнений",
    "cognitive_closure_cliches": "клише / когнитивное закрытие",
    "social_proof_pressure": "социальное давление / большинство",
    "gain_loss_exaggeration": "преувеличение выигрыша/потерь",
    "causal_fact_distortion": "искажение причин/фактов",
    "authority_claim_pressure": "апелляция к авторитету",
    "topic_shift_misrepresentation": "подмена темы / искажение",
    "directive_action_pressure": "призыв к действию / давление",
}

ID2LABEL = {i: name for i, name in enumerate(LABEL_NAMES)}
LABEL2ID = {name: i for i, name in ID2LABEL.items()}
 
REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = REPO_ROOT / "data" / "ru_token_cls_dataset"
MODEL_OUTPUT_DIR = REPO_ROOT / "models" / "manipulation_detector_model"
CHECKPOINTS_DIR = REPO_ROOT / "models" / "rubert_multilabel_checkpoints"

HF_MODEL_REPO_ID = "ksruasdh/manipulation-detector-ru"

def get_model_source() -> str:
    if os.getenv("MODEL_SOURCE", "hub") == "local":
        return str(MODEL_OUTPUT_DIR)
    return HF_MODEL_REPO_ID

TRAINING_HYPERPARAMS = dict(
    num_train_epochs=15,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=2,
    warmup_ratio=0.1,
    weight_decay=0.01,
    learning_rate=3e-5,
    logging_steps=50,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",
    greater_is_better=True,
    lr_scheduler_type="cosine",
    max_grad_norm=1.0,
)

EARLY_STOPPING_PATIENCE = 3
FOCAL_LOSS_GAMMA = 2.0
POS_WEIGHT_CLIP = (1.0, 50.0)
PREDICT_THRESHOLD = 0.5
EVAL_METRIC_THRESHOLD = 0.3
MAX_SEQ_LENGTH = 512