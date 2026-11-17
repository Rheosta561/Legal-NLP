from fastapi import APIRouter, HTTPException, Request

from app.schemas.chat import ChatRequest, ChatResponse
from app.schemas.prediction import PredictionRequest, PredictionResponse

router = APIRouter()


@router.get("/health")
async def health():
    return {"status": "ok"}


@router.get("/ready")
async def ready(request: Request):
    state = request.app.state
    classifier = state.classifier
    if not classifier.ready:
        raise HTTPException(status_code=503, detail="Classifier artifacts are not ready")
    if not state.settings.ollama_base_url:
        raise HTTPException(status_code=503, detail="OLLAMA_BASE_URL is not configured")
    return {"status": "ready", "classifier_loaded": True, "scaler_loaded": True, "class_mapping_loaded": True}


@router.post("/predict", response_model=PredictionResponse)
async def predict(body: PredictionRequest, request: Request):
    try:
        return await request.app.state.inference.predict(body.text)
    except Exception as exc:
        raise HTTPException(status_code=502, detail="Legal inference failed; check embedding service availability") from exc


@router.post("/chat", response_model=ChatResponse)
async def chat(body: ChatRequest, request: Request):
    try:
        return await request.app.state.agent.invoke(body.message)
    except Exception as exc:
        if type(exc).__name__ == "GuardrailViolation":
            return {"response": "I cannot process that request under the active safety policy.", "tool_called": False, "classification": None}
        raise HTTPException(status_code=502, detail="Guardrailed chat request failed") from exc
