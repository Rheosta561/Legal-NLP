"""Remote BGE-M3 embeddings for legal inference queries."""

from __future__ import annotations

from typing import Any

import httpx


class EmbeddingServiceError(RuntimeError):
    """Raised when the configured embedding API cannot return a valid vector."""


class EmbeddingService:
    """Calls an Ollama-compatible API and validates the BGE-M3 vector."""

    def __init__(
        self,
        base_url: str,
        model: str = "bge-m3",
        dimension: int = 1024,
        timeout: float = 120.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.dimension = dimension
        self.timeout = timeout
        self._client = client

    async def embed_query(self, text: str) -> list[float]:
        if not text.strip():
            raise ValueError("Query text must not be empty")
        if not self.base_url:
            raise EmbeddingServiceError("OLLAMA_BASE_URL is not configured")

        own_client = self._client is None
        client = self._client or httpx.AsyncClient(timeout=self.timeout)
        try:
            response = await client.post(
                f"{self.base_url}/api/embed",
                json={"model": self.model, "input": text},
            )
            response.raise_for_status()
            payload: Any = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise EmbeddingServiceError("BGE-M3 embedding API request failed") from exc
        finally:
            if own_client:
                await client.aclose()

        try:
            vectors = payload["embeddings"]
            vector = vectors[0] if vectors and isinstance(vectors[0], list) else vectors
            result = [float(value) for value in vector]
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise EmbeddingServiceError("Embedding API returned an invalid response") from exc
        if len(result) != self.dimension:
            raise EmbeddingServiceError(
                f"Expected {self.dimension}-dimensional BGE-M3 embedding; received {len(result)}"
            )
        return result
