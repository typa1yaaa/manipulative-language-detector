from fastapi import FastAPI, HTTPException

from .inference import get_detector, predict
from .schemas import PredictRequest, PredictResponse

app = FastAPI(
    title="Manipulation Detector API",
    description="Детектирует манипулятивные языковые паттерны в русскоязычном тексте",
    version="0.1.0",
)


@app.on_event("startup")
def load_model_on_startup():
    get_detector()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict_endpoint(req: PredictRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Текст не должен быть пустым")
    return predict(req.text)
