"""
Pydantic models for API request/response schemas.
Production-ready models with comprehensive validation.
"""

from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


class ChatRequest(BaseModel):
    """Request schema for the chat endpoint."""
    message: str = Field(
        ..., min_length=1, max_length=2000,
        description="User's question or message"
    )
    session_id: Optional[str] = Field(
        None, description="Session ID for conversation continuity"
    )
    stream: bool = Field(
        False, description="Whether to stream response tokens (SSE)"
    )
    regenerate: bool = Field(
        False, description="Whether to regenerate the last assistant response"
    )


class SourceChunk(BaseModel):
    """A retrieved document chunk used as context."""
    content: str = Field(..., description="Text content of the chunk")
    source: str = Field(..., description="Source document filename")
    relevance_score: float = Field(..., description="Similarity score (0-1)")


class ChatResponse(BaseModel):
    """Response schema for the chat endpoint."""
    response: str = Field(..., description="AI-generated response")
    session_id: str = Field(..., description="Session ID for conversation continuity")
    sources: list[SourceChunk] = Field(default_factory=list, description="Retrieved source chunks")
    confidence: float = Field(default=0.0, description="Average retrieval confidence (0-1)")
    suggested_questions: list[str] = Field(default_factory=list, description="Suggested follow-up questions")
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class ChatMessage(BaseModel):
    """A single message in the conversation history."""
    role: str = Field(..., description="Message role: 'user' or 'assistant'")
    content: str = Field(..., description="Message content")
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    sources: list[SourceChunk] = Field(default_factory=list)
    suggested_questions: list[str] = Field(default_factory=list, description="Suggested follow-up questions")


class ConversationHistory(BaseModel):
    """Full conversation history for a session."""
    session_id: str
    messages: list[ChatMessage] = Field(default_factory=list)
    message_count: int = Field(default=0, description="Total messages in session")


class UploadResponse(BaseModel):
    """Response schema for the document upload endpoint."""
    status: str = Field(..., description="Upload status")
    message: str = Field(..., description="Human-readable result message")
    filename: str = Field(..., description="Uploaded filename")
    chunks_added: int = Field(..., description="Number of chunks indexed")
    total_indexed_chunks: int = Field(..., description="Total chunks in knowledge base")


class HealthResponse(BaseModel):
    """Health check response with detailed system status."""
    status: str
    rag_status: str
    documents_indexed: int
    vector_db: str
    llm_provider: str
    llm_model: str
    embedding_model: str
    chunk_size: int
    top_k: int
    llm_key_configured: Optional[bool] = None
    embedding_status: Optional[str] = None
    vector_db_status: Optional[str] = None
