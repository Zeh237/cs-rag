from fastapi import FastAPI
from contextlib import asynccontextmanager
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from routers import ingest, search, rag
from services.elasticsearch_service import es_service
import os

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Testing Elasticsearch connection...")
    if es_service.verify_connection():
        print("Successfully connected to Elasticsearch!")
    else:
        raise ConnectionError("Failed to ping Elasticsearch. Verify host and auth configuration.")
    yield

app = FastAPI(
    title="CSC RAG Ingestion Engine",
    version="1.0.0",
    lifespan=lifespan
)

app.include_router(ingest.router)
app.include_router(search.router)
app.include_router(rag.router)

@app.get("/health")
def health_check():
    return {"status": "ok", "elasticsearch": es_service.verify_connection()}

# Serve static frontend
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/")
async def root():
    index_path = os.path.join(static_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "CSC RAG API", "docs": "/docs"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)