"""
Configuration module for the Banking Support Chatbot.
Loads settings from environment variables with sensible defaults.
"""

import os
from pathlib import Path
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# Load .env file from project root
env_path = Path(__file__).resolve().parent.parent.parent / ".env"
load_dotenv(dotenv_path=env_path)


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # API Keys
    GOOGLE_API_KEY: str = ""

    # Application
    APP_ENV: str = "development"
    APP_HOST: str = "0.0.0.0"
    APP_PORT: int = 10000

    # RAG Pipeline
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50
    TOP_K_RESULTS: int = 5
    COLLECTION_NAME: str = "banking_knowledge"

    # Models
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    LLM_MODEL: str = "gemini-2.0-flash"

    # Paths
    DATA_DIR: str = str(Path(__file__).resolve().parent.parent / "data" / "banking_knowledge")
    CHROMA_PERSIST_DIR: str = str(Path(__file__).resolve().parent.parent / "chroma_db")

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
