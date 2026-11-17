from typing import Any

from langchain_core.tools import tool

from app.services.legal_inference_service import LegalInferenceService


def build_classification_tool(inference: LegalInferenceService):
    @tool("classify_legal_provision")
    async def classify_legal_provision(text: str) -> dict[str, Any]:
        """Classify a legal provision into one of the 100 LEDGAR categories using the trained BGE-M3 classifier."""
        return await inference.predict(text)

    return classify_legal_provision
