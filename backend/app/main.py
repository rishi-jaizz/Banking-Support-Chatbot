"""
FastAPI application entry point.
Configures the app, initializes the RAG pipeline on startup,
and serves the frontend as static files.
"""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from app.config import settings
from app.rag.ingestion import ingestion_pipeline
from app.rag.embeddings import embedding_service
from app.rag.generator import generator
from app.routes import chat, health, upload

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan handler.
    Initializes the RAG pipeline on startup with detailed logging.
    """
    logger.info("=" * 60)
    logger.info("BankAssist AI — Starting Up")
    logger.info(f"Provider: {settings.LLM_PROVIDER} | Model: {settings.LLM_MODEL}")
    logger.info(f"Embeddings: {settings.EMBEDDING_MODEL} | Chunk: {settings.CHUNK_SIZE}/{settings.CHUNK_OVERLAP}")
    logger.info("=" * 60)

    # Step 1: Load embedding model
    logger.info("[1/4] Loading embedding model...")
    embedding_service.load_model()

    # Step 2: Initialize ChromaDB
    logger.info("[2/4] Initializing ChromaDB...")
    ingestion_pipeline.initialize()

    # Step 3: Ingest documents
    logger.info("[3/4] Running document ingestion pipeline...")
    ingestion_pipeline.ingest_documents()

    # Step 4: Initialize LLM
    logger.info(f"[4/4] Initializing {settings.LLM_PROVIDER.upper()} LLM...")
    try:
        generator.initialize()
        logger.info(f"LLM initialized: {settings.LLM_PROVIDER}/{generator.model_name}")
    except ValueError as e:
        logger.warning(f"LLM initialization warning: {e}")
        logger.warning("Chat will not work without a valid API key")

    logger.info("=" * 60)
    logger.info(f"RAG Pipeline Ready | {ingestion_pipeline.documents_count} chunks indexed")
    logger.info("=" * 60)

    yield  # Application runs

    logger.info("BankAssist AI — Shutting Down")


# Create FastAPI app
app = FastAPI(
    title="BankAssist AI — Banking Support Chatbot API",
    description="AI-powered banking support chatbot using Retrieval-Augmented Generation (RAG) with ChromaDB vector database and multi-provider LLM support (Gemini/Groq/OpenAI).",
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware
origins = [org.strip() for org in settings.CORS_ORIGINS.split(",") if org.strip()]
allow_credentials = True
if "*" in origins:
    allow_credentials = False

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(chat.router)
app.include_router(health.router)
app.include_router(upload.router)

# Serve frontend static files
frontend_dir = Path(__file__).resolve().parent.parent.parent / "frontend"
if frontend_dir.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")


@app.get("/")
async def serve_frontend():
    """Serve the frontend index.html."""
    index_path = frontend_dir / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return {"message": "BankAssist AI API", "docs": "/docs"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.APP_HOST,
        port=settings.APP_PORT,
        reload=settings.APP_ENV == "development"
    )
