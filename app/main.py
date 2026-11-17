from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI

from app.agents.legal_agent import LegalAgent
from app.api.routes import router
from app.core.config import get_settings
from app.services.classifier_service import ClassifierService
from app.services.embedding_service import EmbeddingService
from app.services.legal_inference_service import LegalInferenceService


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    classifier = ClassifierService(settings.model_dir, settings.embedding_dimension, settings.num_classes)
    classifier.load()
    client = httpx.AsyncClient(timeout=settings.request_timeout)
    embeddings = EmbeddingService(settings.ollama_base_url, settings.ollama_model, settings.embedding_dimension, settings.request_timeout, client)
    inference = LegalInferenceService(embeddings, classifier)
    app.state.settings = settings
    app.state.classifier = classifier
    app.state.inference = inference
    app.state.agent = LegalAgent(settings, inference)
    app.state.ollama_client = client
    try:
        yield
    finally:
        await app.state.agent.close()
        await client.aclose()


app = FastAPI(title="Legal NLP Inference API", version="1.0.0", lifespan=lifespan)
app.include_router(router)
