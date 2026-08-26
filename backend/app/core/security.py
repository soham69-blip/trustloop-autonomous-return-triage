"""
TrustLoop Security & Access Control Module.

Provides role-based API key authentication, administrative gating, and path sanitization.
"""

from fastapi import Header, HTTPException, status
from typing import Optional
from pathlib import Path
import secrets

from backend.app.core.config import settings


def verify_api_key(
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
) -> str:
    """
    Validate standard operational API key.
    When AUTH_ENABLED is True, strictly requires a valid key.
    """
    if not settings.AUTH_ENABLED:
        return x_api_key or "anonymous"

    if not x_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required: missing 'X-API-Key' header",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    valid_keys = [settings.API_KEY, settings.ADMIN_API_KEY]
    is_valid = any(secrets.compare_digest(x_api_key, vk) for vk in valid_keys)

    if not is_valid:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key provided",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    return x_api_key


def verify_admin_key(
    x_admin_key: Optional[str] = Header(None, alias="X-Admin-API-Key"),
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
) -> str:
    """
    Validate administrative access key for destructive/lifecycle endpoints:
    - /models/rollback
    - /models/promote
    - /shadow/toggle
    - /snapshots/create
    """
    provided_key = x_admin_key or x_api_key

    if not settings.AUTH_ENABLED:
        return provided_key or "admin-dev"

    if not provided_key:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Administrative authorization required: missing 'X-Admin-API-Key' header",
        )

    is_admin = secrets.compare_digest(provided_key, settings.ADMIN_API_KEY)
    if not is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Forbidden: insufficient administrative permissions",
        )

    return provided_key


def sanitize_secure_path(base_directory: Path, untrusted_filename: str) -> Path:
    """
    Path traversal protection.
    Ensures that resolved path stays strictly within the intended base directory.
    """
    base_resolved = base_directory.resolve()
    target_path = (base_directory / untrusted_filename).resolve()

    try:
        target_path.relative_to(base_resolved)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Security violation: path traversal detected for filename '{untrusted_filename}'",
        )

    if target_path == base_resolved:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Security violation: invalid target filename '{untrusted_filename}'",
        )

    return target_path
