import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api import auth, chats, ingest, rag, search
from app.db.session import init_db
from app.services.elasticsearch import es_service
from app.services.memory import memory_store

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    if not es_service.verify_connection():
        raise ConnectionError("Failed to connect to Elasticsearch")
    yield
    await memory_store.close()


def create_app() -> FastAPI:
    app = FastAPI(title="CSC RAG Engine", version="1.0.0", lifespan=lifespan)

    app.include_router(auth.router)
    app.include_router(chats.router)
    app.include_router(ingest.router)
    app.include_router(search.router)
    app.include_router(rag.router)

    @app.get("/health")
    def health_check():
        return {"status": "ok", "elasticsearch": es_service.verify_connection()}

    if os.path.isdir(STATIC_DIR):
        app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/")
    async def root():
        index_path = os.path.join(STATIC_DIR, "index.html")
        if os.path.isfile(index_path):
            return FileResponse(index_path)
        return {"message": "CSC RAG API", "docs": "/docs"}

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
