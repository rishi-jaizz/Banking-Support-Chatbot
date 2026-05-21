"""
Pydantic models for API request/response schemas.
"""

from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


class ChatRequest(BaseModel):
    """Request schema for the chat endpoint."""
    message: str = Field(..., min_length=1, max_length=2000, description="User's question or message")
    session_id: Optional[str] = Field(None, description="Session ID for conversation continuity")


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
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class ChatMessage(BaseModel):
    """A single message in the conversation history."""
    role: str = Field(..., description="Message role: 'user' or 'assistant'")
    content: str = Field(..., description="Message content")
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    sources: list[SourceChunk] = Field(default_factory=list)


class ConversationHistory(BaseModel):
    """Full conversation history for a session."""
    session_id: str
    messages: list[ChatMessage] = Field(default_factory=list)


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    rag_status: str
    documents_indexed: int
    vector_db: str
    llm_model: str
    embedding_model: str
