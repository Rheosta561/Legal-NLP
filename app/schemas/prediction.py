from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    text: str = Field(min_length=1, max_length=100_000)


class PredictionResponse(BaseModel):
    predicted_class_id: int
    predicted_class: str
    confidence: float
    top_predictions: list[dict]
