import time
import logging
from collections import defaultdict
from fastapi import Request, HTTPException
from app.config import settings

logger = logging.getLogger(__name__)

class RateLimiter:
    """Simple in-memory IP-based rate limiter."""
    def __init__(self, limit: int, period: int = 60):
        self.limit = limit
        self.period = period
        # Maps IP -> list of timestamps
        self.requests: dict[str, list[float]] = defaultdict(list)

    def is_allowed(self, ip: str) -> bool:
        now = time.time()
        # Filter out timestamps older than the period
        self.requests[ip] = [t for t in self.requests[ip] if now - t < self.period]
        
        if len(self.requests[ip]) >= self.limit:
            return False
        
        self.requests[ip].append(now)
        return True

chat_limiter = RateLimiter(settings.RATE_LIMIT_CHAT)
upload_limiter = RateLimiter(settings.RATE_LIMIT_UPLOAD)

async def check_chat_rate_limit(request: Request):
    """Dependency to check rate limits for chat requests."""
    ip = request.client.host if request.client else "unknown"
    if not chat_limiter.is_allowed(ip):
        logger.warning(f"Rate limit exceeded for chat endpoint by IP: {ip}")
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Please wait a minute before trying again."
        )

async def check_upload_rate_limit(request: Request):
    """Dependency to check rate limits for upload requests."""
    ip = request.client.host if request.client else "unknown"
    if not upload_limiter.is_allowed(ip):
        logger.warning(f"Rate limit exceeded for upload endpoint by IP: {ip}")
        raise HTTPException(
            status_code=429,
            detail="Upload rate limit exceeded. Please wait a minute before trying again."
        )
