"""
Chat API routes.
Handles conversation, session management, and chat history.
"""

import uuid
import logging
from datetime import datetime
from collections import defaultdict

from fastapi import APIRouter, HTTPException

from app.models.schemas import (
    ChatRequest,
    ChatResponse,
    ChatMessage,
    ConversationHistory,
    SourceChunk,
)
from app.rag.generator import generator

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["chat"])

# In-memory session store (use Redis/DB in production)
sessions: dict[str, list[dict]] = defaultdict(list)


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Process a chat message and return an AI-generated response.

    The endpoint:
    1. Validates the input
    2. Retrieves relevant banking knowledge via semantic search
    3. Generates a context-aware response using Gemini LLM
    4. Stores the conversation in session history
    """
    # Generate or reuse session ID
    session_id = request.session_id or str(uuid.uuid4())

    # Get conversation history for this session
    conversation_history = sessions[session_id]

    try:
        # Generate response using RAG pipeline
        response_text, source_chunks = await generator.generate_response(
            query=request.message,
            conversation_history=conversation_history
        )

        # Store user message in history
        sessions[session_id].append({
            "role": "user",
            "content": request.message,
        })

        # Store assistant response in history
        sessions[session_id].append({
            "role": "assistant",
            "content": response_text,
        })

        # Keep only last 20 messages per session to manage memory
        if len(sessions[session_id]) > 20:
            sessions[session_id] = sessions[session_id][-20:]

        return ChatResponse(
            response=response_text,
            session_id=session_id,
            sources=source_chunks,
            timestamp=datetime.utcnow().isoformat()
        )

    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(
            status_code=500,
            detail="An error occurred while processing your request. Please try again."
        )


@router.get("/sessions/{session_id}/history", response_model=ConversationHistory)
async def get_history(session_id: str):
    """Get the conversation history for a session."""
    if session_id not in sessions:
        return ConversationHistory(session_id=session_id, messages=[])

    messages = []
    for msg in sessions[session_id]:
        messages.append(ChatMessage(
            role=msg["role"],
            content=msg["content"],
            sources=msg.get("sources", [])
        ))

    return ConversationHistory(session_id=session_id, messages=messages)


@router.delete("/sessions/{session_id}")
async def clear_session(session_id: str):
    """Clear a conversation session."""
    if session_id in sessions:
        del sessions[session_id]
    return {"message": "Session cleared", "session_id": session_id}
