from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10_000)


class Span(BaseModel):
    text: str
    pattern_name: str
    label_ru: str
    confidence: float


class PredictResponse(BaseModel):
    spans: list[Span]
    has_manipulation: bool
