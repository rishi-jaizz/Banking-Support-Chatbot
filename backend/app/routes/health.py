"""
Health check API route.
Provides detailed system status including RAG pipeline health,
LLM provider info, and configuration details.
"""

from fastapi import APIRouter
from app.models.schemas import HealthResponse
from app.rag.ingestion import ingestion_pipeline
from app.rag.embeddings import embedding_service
from app.config import settings

router = APIRouter(prefix="/api", tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint returning comprehensive system status.
    Used for monitoring, deployment health checks, and debugging.
    """
    rag_status = "ready" if ingestion_pipeline.documents_count > 0 else "not_initialized"

    provider = settings.LLM_PROVIDER.lower()
    if provider == "gemini":
        llm_key_configured = bool(settings.GOOGLE_API_KEY)
    elif provider == "groq":
        llm_key_configured = bool(settings.GROQ_API_KEY)
    elif provider == "openai":
        llm_key_configured = bool(settings.OPENAI_API_KEY)
    else:
        llm_key_configured = False

    embedding_status = "active" if embedding_service.model is not None else "not_loaded"
    vector_db_status = "connected" if ingestion_pipeline.collection is not None else "disconnected"

    return HealthResponse(
        status="healthy",
        rag_status=rag_status,
        documents_indexed=ingestion_pipeline.documents_count,
        vector_db="ChromaDB (persistent, cosine similarity)",
        llm_provider=settings.LLM_PROVIDER,
        llm_model=settings.LLM_MODEL,
        embedding_model=settings.EMBEDDING_MODEL,
        chunk_size=settings.CHUNK_SIZE,
        top_k=settings.TOP_K_RESULTS,
        llm_key_configured=llm_key_configured,
        embedding_status=embedding_status,
        vector_db_status=vector_db_status,
    )
