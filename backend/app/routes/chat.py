"""
Chat API routes.
Handles conversation, session management, and chat history
with production-grade logging, error handling, and context management.
"""

import uuid
import logging
import json
import os
from pathlib import Path
from datetime import datetime
from fastapi import APIRouter, HTTPException, Depends, Request
from fastapi.responses import StreamingResponse

from app.models.schemas import (
    ChatRequest,
    ChatResponse,
    ChatMessage,
    ConversationHistory,
    SourceChunk,
)
from app.config import settings
from app.rag.generator import generator
from app.utils.rate_limit import check_chat_rate_limit

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["chat"])

MAX_HISTORY_LENGTH = 20


class FileSessionStore:
    """Disk-persisted session store for conversation history."""
    def __init__(self):
        self.sessions_dir = Path(settings.SESSIONS_DIR)
        os.makedirs(self.sessions_dir, exist_ok=True)

    def _get_path(self, session_id: str) -> Path:
        # Sanitize session_id to prevent path traversal
        clean_id = "".join([c for c in session_id if c.isalnum() or c in ("-", "_")])
        return self.sessions_dir / f"{clean_id}.json"

    def get(self, session_id: str) -> list[dict]:
        path = self._get_path(session_id)
        if not path.exists():
            return []
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data.get("messages", [])
        except Exception as e:
            logger.error(f"Error loading session {session_id} from disk: {e}")
            return []

    def save(self, session_id: str, messages: list[dict]):
        path = self._get_path(session_id)
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"session_id": session_id, "messages": messages}, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error saving session {session_id} to disk: {e}")

    def delete(self, session_id: str):
        path = self._get_path(session_id)
        if path.exists():
            try:
                path.unlink()
            except Exception as e:
                logger.error(f"Error deleting session file {session_id}: {e}")


session_store = FileSessionStore()


@router.post("/chat", dependencies=[Depends(check_chat_rate_limit)])
async def chat(request: ChatRequest):
    """
    Process a chat message and return an AI-generated response.
    Supports streaming response tokens (SSE) if request.stream is True.
    Supports regenerating the last assistant response if request.regenerate is True.
    """
    session_id = request.session_id or f"sess_{uuid.uuid4().hex[:12]}"
    query = request.message.strip()

    # Handle regeneration logic
    if request.regenerate:
        history = session_store.get(session_id)
        if not history:
            raise HTTPException(status_code=400, detail="No conversation history to regenerate.")
        
        # Pop the last assistant message and get the preceding user query
        last_msg = history[-1]
        if last_msg["role"] == "assistant":
            history.pop()  # remove assistant message
            if history and history[-1]["role"] == "user":
                last_user_msg = history.pop()  # remove user message to recreate it
                query = last_user_msg["content"]
            else:
                if query == "":
                    raise HTTPException(status_code=400, detail="No user message found to regenerate.")
        else:
            if last_msg["role"] == "user":
                history.pop()
                query = last_msg["content"]
            else:
                if query == "":
                    raise HTTPException(status_code=400, detail="No user message found to regenerate.")
        
        # Save clean history prior to this turn
        session_store.save(session_id, history)
        logger.info(f"[CHAT REGENERATE] session={session_id} | re-running query='{query[:50]}...'")

    logger.info(f"[CHAT] session={session_id} | query='{query[:80]}...' | stream={request.stream}")

    # Load conversation history for prompt injection
    conversation_history = session_store.get(session_id)

    # 1. Streaming response pathway (SSE)
    if request.stream:
        async def event_generator():
            accumulated_response = ""
            sources = []
            confidence = 0.0
            suggested_questions = []
            
            try:
                async for chunk in generator.generate_response_stream(query, conversation_history):
                    event = chunk.get("event")
                    if event == "metadata":
                        sources = chunk.get("sources", [])
                        confidence = chunk.get("confidence", 0.0)
                        yield f"data: {json.dumps(chunk)}\n\n"
                    elif event == "token":
                        text = chunk.get("text", "")
                        accumulated_response += text
                        yield f"data: {json.dumps(chunk)}\n\n"
                    elif event == "done":
                        suggested_questions = chunk.get("suggested_questions", [])
                        yield f"data: {json.dumps(chunk)}\n\n"
                
                # Append to persistent history
                hist = session_store.get(session_id)
                hist.append({
                    "role": "user",
                    "content": query,
                    "timestamp": datetime.utcnow().isoformat()
                })
                hist.append({
                    "role": "assistant",
                    "content": accumulated_response,
                    "sources": sources,
                    "suggested_questions": suggested_questions,
                    "timestamp": datetime.utcnow().isoformat()
                })
                if len(hist) > MAX_HISTORY_LENGTH:
                    hist = hist[-MAX_HISTORY_LENGTH:]
                session_store.save(session_id, hist)
                
            except Exception as e:
                logger.error(f"Error in streaming event generator: {e}", exc_info=True)
                error_token = "I apologize, but I am currently experiencing connectivity issues with the AI model. Please try sending your message again in a moment."
                yield f"data: {json.dumps({'event': 'token', 'text': error_token})}\n\n"
                yield f"data: {json.dumps({'event': 'done', 'text': error_token, 'suggested_questions': []})}\n\n"

        return StreamingResponse(event_generator(), media_type="text/event-stream")

    # 2. Standard JSON response pathway
    try:
        response_text, source_chunks, confidence, suggested_questions = await generator.generate_response(
            query=query,
            conversation_history=conversation_history
        )

        # Append messages to session history
        history = session_store.get(session_id)
        history.append({
            "role": "user",
            "content": query,
            "timestamp": datetime.utcnow().isoformat()
        })
        history.append({
            "role": "assistant",
            "content": response_text,
            "sources": [c.dict() for c in source_chunks],
            "suggested_questions": suggested_questions,
            "timestamp": datetime.utcnow().isoformat()
        })

        if len(history) > MAX_HISTORY_LENGTH:
            history = history[-MAX_HISTORY_LENGTH:]

        session_store.save(session_id, history)

        logger.info(f"[CHAT] session={session_id} | response_len={len(response_text)} | sources={len(source_chunks)} | confidence={confidence}")

        return ChatResponse(
            response=response_text,
            session_id=session_id,
            sources=source_chunks,
            confidence=confidence,
            suggested_questions=suggested_questions,
            timestamp=datetime.utcnow().isoformat()
        )

    except Exception as e:
        logger.error(f"[CHAT ERROR] session={session_id} | error={e}", exc_info=True)
        error_msg = "I apologize, but I am currently experiencing connectivity issues with the AI model. Please try again in a moment."
        return ChatResponse(
            response=error_msg,
            session_id=session_id,
            sources=[],
            confidence=0.0,
            suggested_questions=[],
            timestamp=datetime.utcnow().isoformat()
        )


@router.get("/sessions/{session_id}/history", response_model=ConversationHistory)
async def get_history(session_id: str):
    """Get the conversation history for a session."""
    history = session_store.get(session_id)
    if not history:
        return ConversationHistory(session_id=session_id, messages=[], message_count=0)

    messages = []
    for msg in history:
        sources_list = []
        for s in msg.get("sources", []):
            if isinstance(s, dict):
                sources_list.append(SourceChunk(**s))
            else:
                sources_list.append(s)

        messages.append(ChatMessage(
            role=msg["role"],
            content=msg["content"],
            timestamp=msg.get("timestamp", datetime.utcnow().isoformat()),
            sources=sources_list,
            suggested_questions=msg.get("suggested_questions", [])
        ))

    return ConversationHistory(
        session_id=session_id,
        messages=messages,
        message_count=len(messages)
    )


@router.delete("/sessions/{session_id}")
async def clear_session(session_id: str):
    """Clear a conversation session."""
    session_store.delete(session_id)
    logger.info(f"[SESSION] Cleared session: {session_id}")
    return {"message": "Session cleared", "session_id": session_id}
