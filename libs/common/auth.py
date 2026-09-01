"""Authentication and authorization utilities."""

import hashlib
import secrets
import time
from datetime import datetime, timedelta, timezone
from functools import wraps
from typing import Any

import jwt
from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

from .schemas import FraudVector


class TokenPayload(BaseModel):
    """JWT token payload."""
    sub: str  # subject (service name or user ID)
    iss: str  # issuer
    aud: list[str]  # audience
    exp: int  # expiration timestamp
    iat: int  # issued at timestamp
    scopes: list[str] = []
    metadata: dict[str, Any] = {}


class APIKey(BaseModel):
    """API key model."""
    key_id: str
    key_hash: str
    name: str
    scopes: list[str]
    merchant_ids: list[str]  # Empty = all merchants
    vectors: list[FraudVector]  # Empty = all vectors
    rate_limit: int = 1000  # requests per minute
    created_at: datetime
    expires_at: datetime | None = None
    is_active: bool = True
    last_used: datetime | None = None


class AuthConfig:
    """Authentication configuration."""
    JWT_SECRET: str = "dev-secret-change-in-production"
    JWT_ALGORITHM: str = "HS256"
    JWT_ISSUER: str = "ai-risk-manager"
    JWT_AUDIENCE: list[str] = ["ai-risk-manager-api"]
    JWT_EXPIRY_MINUTES: int = 60
    API_KEY_PREFIX: str = "ak_"
    API_KEY_LENGTH: int = 32


def create_jwt_token(
    subject: str,
    scopes: list[str] = None,
    metadata: dict[str, Any] = None,
    expiry_minutes: int = None,
    config: AuthConfig = None,
) -> str:
    """Create a JWT token."""
    config = config or AuthConfig()
    now = datetime.now(timezone.utc)
    expiry = now + timedelta(minutes=expiry_minutes or config.JWT_EXPIRY_MINUTES)

    payload = TokenPayload(
        sub=subject,
        iss=config.JWT_ISSUER,
        aud=config.JWT_AUDIENCE,
        exp=int(expiry.timestamp()),
        iat=int(now.timestamp()),
        scopes=scopes or [],
        metadata=metadata or {},
    )

    return jwt.encode(
        payload.model_dump(),
        config.JWT_SECRET,
        algorithm=config.JWT_ALGORITHM,
    )


def decode_jwt_token(token: str, config: AuthConfig = None) -> TokenPayload:
    """Decode and validate a JWT token."""
    config = config or AuthConfig()

    try:
        payload = jwt.decode(
            token,
            config.JWT_SECRET,
            algorithms=[config.JWT_ALGORITHM],
            audience=config.JWT_AUDIENCE,
            issuer=config.JWT_ISSUER,
        )
        return TokenPayload(**payload)
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError as e:
        raise HTTPException(status_code=401, detail=f"Invalid token: {e!s}")


def generate_api_key(config: AuthConfig = None) -> tuple[str, str]:
    """Generate a new API key. Returns (key_id, full_key)."""
    config = config or AuthConfig()
    key_id = secrets.token_urlsafe(16)
    key_secret = secrets.token_urlsafe(config.API_KEY_LENGTH)
    full_key = f"{config.API_KEY_PREFIX}{key_id}.{key_secret}"
    key_hash = hashlib.sha256(full_key.encode()).hexdigest()
    return key_id, full_key


def hash_api_key(key: str) -> str:
    """Hash an API key for storage."""
    return hashlib.sha256(key.encode()).hexdigest()


def verify_api_key(key: str, stored_hash: str) -> bool:
    """Verify an API key against stored hash."""
    return hash_api_key(key) == stored_hash


# FastAPI dependencies
security = HTTPBearer(auto_error=False)


async def get_current_token(
    credentials: HTTPAuthorizationCredentials | None = Depends(security),
    config: AuthConfig = None,
) -> TokenPayload | None:
    """Extract and validate JWT token from Authorization header."""
    if not credentials:
        return None

    if credentials.scheme.lower() != "bearer":
        raise HTTPException(status_code=401, detail="Invalid authentication scheme")

    return decode_jwt_token(credentials.credentials, config)


async def require_scopes(
    required_scopes: list[str],
    token: TokenPayload = Depends(get_current_token),
) -> TokenPayload:
    """Require specific scopes in JWT token."""
    if not token:
        raise HTTPException(status_code=401, detail="Authentication required")

    if not all(scope in token.scopes for scope in required_scopes):
        raise HTTPException(
            status_code=403,
            detail=f"Insufficient scopes. Required: {required_scopes}, Have: {token.scopes}"
        )

    return token


async def require_merchant_access(
    merchant_id: str,
    token: TokenPayload = Depends(get_current_token),
) -> TokenPayload:
    """Require access to specific merchant."""
    if not token:
        raise HTTPException(status_code=401, detail="Authentication required")

    # Check if token has merchant access (via metadata or scopes)
    allowed_merchants = token.metadata.get("merchant_ids", [])
    if allowed_merchants and merchant_id not in allowed_merchants:
        raise HTTPException(
            status_code=403,
            detail=f"Access denied to merchant {merchant_id}"
        )

    return token


async def require_vector_access(
    vector: FraudVector,
    token: TokenPayload = Depends(get_current_token),
) -> TokenPayload:
    """Require access to specific fraud vector."""
    if not token:
        raise HTTPException(status_code=401, detail="Authentication required")

    allowed_vectors = token.metadata.get("vectors", [])
    if allowed_vectors and vector.value not in allowed_vectors:
        raise HTTPException(
            status_code=403,
            detail=f"Access denied to vector {vector.value}"
        )

    return token


# Service-to-service authentication
def create_service_token(
    service_name: str,
    vectors: list[FraudVector] = None,
    expiry_hours: int = 24,
    config: AuthConfig = None,
) -> str:
    """Create a service-to-service token."""
    return create_jwt_token(
        subject=service_name,
        scopes=["service:read", "service:write"],
        metadata={
            "type": "service",
            "vectors": [v.value for v in vectors] if vectors else [],
        },
        expiry_minutes=expiry_hours * 60,
        config=config,
    )


def validate_service_token(token: str, expected_service: str = None) -> TokenPayload:
    """Validate a service-to-service token."""
    payload = decode_jwt_token(token)

    if payload.metadata.get("type") != "service":
        raise HTTPException(status_code=403, detail="Not a service token")

    if expected_service and payload.sub != expected_service:
        raise HTTPException(status_code=403, detail="Invalid service")

    return payload


# Rate limiting
class RateLimiter:
    """Simple in-memory rate limiter."""

    def __init__(self):
        self._requests: dict[str, list[float]] = {}

    def is_allowed(self, key: str, limit: int, window_seconds: int = 60) -> bool:
        """Check if request is allowed under rate limit."""
        now = time.time()
        if key not in self._requests:
            self._requests[key] = []

        # Remove old requests outside window
        self._requests[key] = [
            ts for ts in self._requests[key]
            if now - ts < window_seconds
        ]

        if len(self._requests[key]) >= limit:
            return False

        self._requests[key].append(now)
        return True

    def get_remaining(self, key: str, limit: int, window_seconds: int = 60) -> int:
        """Get remaining requests in current window."""
        now = time.time()
        if key not in self._requests:
            return limit

        current = len([
            ts for ts in self._requests[key]
            if now - ts < window_seconds
        ])
        return max(0, limit - current)


# Global rate limiter instance
rate_limiter = RateLimiter()


def rate_limit(limit: int, window_seconds: int = 60, key_func: callable = None):
    """Decorator for rate limiting endpoints."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract request from args/kwargs
            request = None
            for arg in args:
                if isinstance(arg, Request):
                    request = arg
                    break
            if not request:
                for v in kwargs.values():
                    if isinstance(v, Request):
                        request = v
                        break

            if request:
                key = key_func(request) if key_func else request.client.host
                if not rate_limiter.is_allowed(key, limit, window_seconds):
                    remaining = rate_limiter.get_remaining(key, limit, window_seconds)
                    raise HTTPException(
                        status_code=429,
                        detail="Rate limit exceeded. Try again later.",
                        headers={"X-RateLimit-Remaining": str(remaining)}
                    )

            return await func(*args, **kwargs)
        return wrapper
    return decorator
