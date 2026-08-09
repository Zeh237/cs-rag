import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings:
    ES_HOST: str = os.getenv("ES_HOST", "http://localhost:9200")
    ES_USER: str = os.getenv("ES_USER", "elastic")
    ES_PASSWORD: str = os.getenv("ES_PASSWORD", "password")
    INDEX_NAME: str = os.getenv("INDEX_NAME", "csc_notes")

    NOTES_DIR: str = os.getenv("NOTES_DIR", "./csc_notes")
    NOTES_MANIFEST: str = os.path.join(NOTES_DIR, "library_metadata.json")

    LLM_API_BASE: str = os.getenv("LLM_API_BASE", "")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")

    REDIS_HOST: str = os.getenv("REDIS_HOST", "redis://localhost:6379/")
    REDIS_PASSWORD: str = os.getenv("REDIS_PASSWORD", "")

    SECRET_KEY: str = os.getenv("SECRET_KEY", "change-me-in-production")
    ALGORITHM: str = "HS256"

    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///./rag.db")


settings = Settings()
