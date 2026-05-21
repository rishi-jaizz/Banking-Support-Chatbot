"""
Health check API route.
Provides system status including RAG pipeline health.
"""

from fastapi import APIRouter
from app.models.schemas import HealthResponse
from app.rag.ingestion import ingestion_pipeline
from app.config import settings

router = APIRouter(prefix="/api", tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint returning system status.
    Useful for monitoring and deployment health checks.
    """
    rag_status = "ready" if ingestion_pipeline.documents_count > 0 else "not_initialized"

    return HealthResponse(
        status="healthy",
        rag_status=rag_status,
        documents_indexed=ingestion_pipeline.documents_count,
        vector_db="ChromaDB",
        llm_model=settings.LLM_MODEL,
        embedding_model=settings.EMBEDDING_MODEL
    )
