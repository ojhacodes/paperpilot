import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

# Get the base directory
BASE_DIR = Path(__file__).resolve().parent

class Settings(BaseSettings):
    # API Keys
    GEMINI_API_KEY: Optional[str] = None
    OPENAI_API_KEY: Optional[str] = None
    ANTHROPIC_API_KEY: Optional[str] = None

    # LLM Settings
    DEFAULT_PROVIDER: str = "gemini"  # "gemini" | "openai" | "anthropic"
    GEMINI_MODEL: str = "gemini-2.5-flash"
    OPENAI_MODEL: str = "gpt-4o-mini"
    ANTHROPIC_MODEL: str = "claude-3-5-sonnet-20241022"

    # Vector DB and Files Settings
    CHROMA_DB_PATH: str = str(BASE_DIR / "data" / "chroma_db")
    PAPERS_DIR: str = str(BASE_DIR / "data" / "papers")

    # Chunking Defaults
    DEFAULT_CHUNK_SIZE: int = 512
    DEFAULT_CHUNK_OVERLAP: int = 50

    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR.parent / ".env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )

# Instantiate settings
settings = Settings()

# Ensure directories exist
os.makedirs(settings.CHROMA_DB_PATH, exist_ok=True)
os.makedirs(settings.PAPERS_DIR, exist_ok=True)
