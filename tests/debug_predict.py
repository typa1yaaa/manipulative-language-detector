from src.inference.predictor import ManipulationDetector
from src.training import config
 
detector = ManipulationDetector(config.MODEL_OUTPUT_DIR, threshold=0.0)  # threshold=0 -> видно вообще все ненулевые вероятности
 
text = (
    "Здравствуйте! Собрание кафедры переноситься на 5 вечера, не забудьте свои документы. До вечера, коллеги."
)
 
tokens, probs = detector._token_predictions(text)
 
print(f"{'токен':<20} | " + " | ".join(f"{name[:18]:<18}" for name in detector.id2label.values()))
print("-" * 160)
for tok, row in zip(tokens, probs):
    values = " | ".join(f"{v.item():<18.3f}" for v in row)
    print(f"{tok:<20} | {values}")
 
print("\nМаксимальная вероятность по каждому классу во всём тексте:")
max_per_class = probs.max(dim=0).values
for i, name in detector.id2label.items():
    print(f"  {name}: {max_per_class[i].item():.3f}")