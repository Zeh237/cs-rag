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

    VISION_LLM_MODEL: str = os.getenv("VISION_LLM_MODEL", "")
    RESEARCH_MAX_PAGES: int = int(os.getenv("RESEARCH_MAX_PAGES", "12"))
    RESEARCH_DPI: int = int(os.getenv("RESEARCH_DPI", "110"))
    RESEARCH_SEED_WINDOW: int = int(os.getenv("RESEARCH_SEED_WINDOW", "1"))
    RESEARCH_TOP_K_DOCS: int = int(os.getenv("RESEARCH_TOP_K_DOCS", "1"))
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "google/embeddinggemma-2")

    REDIS_HOST: str = os.getenv("REDIS_HOST", "redis://localhost:6379/")
    REDIS_PASSWORD: str = os.getenv("REDIS_PASSWORD", "")

    SECRET_KEY: str = os.getenv("SECRET_KEY", "change-me-in-production")
    ALGORITHM: str = "HS256"

    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///./rag.db")


settings = Settings()
