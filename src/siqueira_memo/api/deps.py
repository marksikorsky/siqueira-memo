"""FastAPI dependencies: DB session, auth, profile resolution."""

from __future__ import annotations

import base64
import hashlib
import hmac
import secrets
import time
from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import Depends, Header, HTTPException, Request, status
from sqlalchemy.ext.asyncio import AsyncSession

from siqueira_memo.config import Settings, get_settings
from siqueira_memo.db import get_session_factory

ADMIN_SESSION_COOKIE = "siqueira_admin_session"
_ADMIN_SESSION_PREFIXES = ("/v1/admin", "/v1/memory", "/v1/recall")


def _secret_value(secret: object | None) -> str:
    if secret is None:
        return ""
    if hasattr(secret, "get_secret_value"):
        return str(secret.get_secret_value())
    return str(secret)


def admin_auth_enabled(settings: Settings) -> bool:
    """Return whether app-level admin password auth is configured."""
    return bool(_secret_value(settings.admin_password))


def verify_admin_password(settings: Settings, password: str) -> bool:
    """Constant-time admin password verification."""
    expected = _secret_value(settings.admin_password)
    if not expected:
        return False
    return secrets.compare_digest(password, expected)


def _admin_session_secret(settings: Settings) -> str:
    configured = _secret_value(settings.admin_session_secret)
    if configured:
        return configured
    return settings.api_token.get_secret_value()


def create_admin_session_token(settings: Settings, *, now: int | None = None) -> str:
    """Create a signed, timestamped admin session token for an HttpOnly cookie."""
    issued_at = int(time.time() if now is None else now)
    nonce = secrets.token_urlsafe(18)
    payload = f"{issued_at}:{nonce}"
    encoded_payload = base64.urlsafe_b64encode(payload.encode("utf-8")).decode("ascii").rstrip("=")
    signature = hmac.new(
        _admin_session_secret(settings).encode("utf-8"),
        encoded_payload.encode("ascii"),
        hashlib.sha256,
    ).hexdigest()
    return f"{encoded_payload}.{signature}"


def validate_admin_session_token(
    settings: Settings, token: str | None, *, now: int | None = None
) -> bool:
    """Validate the signed admin session cookie without server-side state."""
    if not admin_auth_enabled(settings) or not token or "." not in token:
        return False
    encoded_payload, signature = token.rsplit(".", 1)
    expected_signature = hmac.new(
        _admin_session_secret(settings).encode("utf-8"),
        encoded_payload.encode("ascii"),
        hashlib.sha256,
    ).hexdigest()
    if not secrets.compare_digest(signature, expected_signature):
        return False
    try:
        padded_payload = encoded_payload + "=" * (-len(encoded_payload) % 4)
        payload = base64.urlsafe_b64decode(padded_payload.encode("ascii")).decode("utf-8")
        issued_at_text, _nonce = payload.split(":", 1)
        issued_at = int(issued_at_text)
    except (ValueError, UnicodeDecodeError):
        return False
    current_time = int(time.time() if now is None else now)
    return 0 <= current_time - issued_at <= settings.admin_session_ttl_seconds


def request_has_admin_session(request: Request) -> bool:
    """Return whether the request carries a valid browser admin session."""
    settings: Settings = request.app.state.settings
    return validate_admin_session_token(settings, request.cookies.get(ADMIN_SESSION_COOKIE))


async def get_db_session(request: Request) -> AsyncIterator[AsyncSession]:
    settings: Settings = request.app.state.settings
    factory = get_session_factory(settings)
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except BaseException:
            await session.rollback()
            raise


async def require_api_token(
    request: Request,
    authorization: Annotated[str | None, Header(alias="Authorization")] = None,
) -> str:
    settings: Settings = request.app.state.settings
    expected = settings.api_token.get_secret_value()
    if authorization is not None:
        if not authorization.lower().startswith("bearer "):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid bearer token")
        token = authorization[7:].strip()
        if not secrets.compare_digest(token, expected):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid bearer token")
        return token
    if expected and request.url.path.startswith(_ADMIN_SESSION_PREFIXES) and request_has_admin_session(request):
        return "admin-session"
    if not expected:
        # Service is unauthenticated. Allowed in dev but log a warning per request.
        return "anonymous"
    raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="missing bearer token")


def get_profile_id(
    request: Request,
    x_profile_id: Annotated[str | None, Header(alias="X-Profile-Id")] = None,
) -> str:
    if x_profile_id:
        return x_profile_id
    settings: Settings = request.app.state.settings
    return settings.derive_profile_id()


SessionDep = Annotated[AsyncSession, Depends(get_db_session)]
AuthDep = Annotated[str, Depends(require_api_token)]
ProfileDep = Annotated[str, Depends(get_profile_id)]
SettingsDep = Annotated[Settings, Depends(get_settings)]
