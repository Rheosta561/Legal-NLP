import asyncio

from app.services.classifier_service import ClassifierService
from app.services.embedding_service import EmbeddingService


class LegalInferenceService:
    def __init__(self, embeddings: EmbeddingService, classifier: ClassifierService):
        self.embeddings = embeddings
        self.classifier = classifier

    async def predict(self, text: str) -> dict:
        vector = await self.embeddings.embed_query(text)
        return await asyncio.to_thread(self.classifier.predict_embedding, vector)
