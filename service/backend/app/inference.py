from functools import lru_cache

from src.inference.predictor import ManipulationDetector
from src.training import config


@lru_cache
def get_detector() -> ManipulationDetector:
    return ManipulationDetector(config.get_model_source())


def predict(text: str) -> dict:
    detector = get_detector()
    raw_spans = detector.predict(text)

    spans = [
        {
            "text": s["text"],
            "pattern_name": s["pattern_name"],
            "label_ru": config.LABEL_RU.get(s["pattern_name"], s["pattern_name"]),
            "confidence": s["confidence"],
        }
        for s in raw_spans
    ]

    return {"spans": spans, "has_manipulation": len(spans) > 0}