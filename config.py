import os

class Settings:
    ES_HOST: str = os.getenv("ES_HOST", "http://localhost:9200")
    ES_USER: str = os.getenv("ES_USER", "elastic")
    ES_PASSWORD: str = os.getenv("ES_PASSWORD", "password")
    INDEX_NAME: str = os.getenv("INDEX_NAME", "research_papers")

    LIBRARY_DIR: str = os.getenv("LIBRARY_DIR", "./csc_library")
    MANIFEST_PATH: str = os.path.join(LIBRARY_DIR, "library_metadata.json")
    LLM_API_BASE: str = os.getenv("LLM_API_BASE")
    LLM_MODEL: str = os.getenv("LLM_MODEL")
    

settings = Settings()