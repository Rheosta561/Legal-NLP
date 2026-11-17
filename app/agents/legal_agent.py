from __future__ import annotations

import os
from langchain.agents import create_agent
from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage

from app.core.config import Settings
from app.tools.classification_tool import build_classification_tool
from app.services.legal_inference_service import LegalInferenceService

SYSTEM_PROMPT = (
    "You are a legal NLP assistant. You can classify legal provisions using the "
    "classify_legal_provision tool. When the user requests classification, invoke "
    "the tool and ground your answer strictly in its output. Never invent classification "
    "labels, confidence scores, or predictions. Explain that the classifier provides "
    "an ML-based prediction and not professional legal advice."
)


class LegalAgent:
    def __init__(self, settings: Settings, inference: LegalInferenceService):
        self.settings = settings
        self.inference = inference
        self._agent = None

    def _build(self):
        model = ChatGroq(model=self.settings.groq_model, api_key=self.settings.groq_api_key, temperature=0)
        tool = build_classification_tool(self.inference)
        from nemoguardrails.integrations.langchain.middleware import GuardrailsMiddleware

        if self.settings.groq_api_key:
            os.environ["GROQ_API_KEY"] = self.settings.groq_api_key
        os.environ["MAIN_MODEL"] = self.settings.groq_model
        middleware = [GuardrailsMiddleware(config_path="config/guardrails")]
        return create_agent(model, tools=[tool], system_prompt=SYSTEM_PROMPT, middleware=middleware)

    async def invoke(self, message: str) -> dict:
        if not self.settings.groq_api_key:
            raise RuntimeError("GROQ_API_KEY is not configured")
        if self._agent is None:
            self._agent = self._build()
        result = await self._agent.ainvoke({"messages": [{"role": "user", "content": message}]})
        messages = result["messages"]
        tool_calls = [tc for item in messages for tc in getattr(item, "tool_calls", [])]
        tool_results = [item for item in messages if getattr(item, "type", "") == "tool"]
        classification = None
        if tool_results:
            raw = tool_results[-1].content
            if isinstance(raw, dict):
                classification = raw
            elif isinstance(raw, str):
                import json
                try:
                    classification = json.loads(raw)
                except json.JSONDecodeError:
                    classification = None
        final_message = next((item for item in reversed(messages) if isinstance(item, AIMessage)), messages[-1])
        return {
            "response": str(getattr(final_message, "content", "")),
            "tool_called": bool(tool_calls or tool_results),
            "classification": classification,
        }

    async def close(self) -> None:
        return None
