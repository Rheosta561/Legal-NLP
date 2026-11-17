from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=100_000)


class ChatResponse(BaseModel):
    response: str
    tool_called: bool
    classification: dict | None = None
