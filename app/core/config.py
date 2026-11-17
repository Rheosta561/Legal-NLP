from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    groq_api_key: str = ""
    groq_model: str = "qwen/qwen3.8-27b"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "bge-m3"
    embedding_dimension: int = 1024
    num_classes: int = 100
    request_timeout: float = 120.0
    model_dir: Path = Path("models/bge_m3")
    groq_timeout: float = 90.0

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    return Settings()
