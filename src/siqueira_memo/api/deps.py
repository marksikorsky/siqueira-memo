"""FastAPI dependencies: DB session, auth, profile resolution."""

from __future__ import annotations

import secrets
from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import Depends, Header, HTTPException, Request, status
from sqlalchemy.ext.asyncio import AsyncSession

from siqueira_memo.config import Settings, get_settings
from siqueira_memo.db import get_session_factory


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
    if not expected:
        # Service is unauthenticated. Allowed in dev but log a warning per request.
        return "anonymous"
    if authorization is None or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="missing bearer token")
    token = authorization[7:].strip()
    if not secrets.compare_digest(token, expected):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid bearer token")
    return token


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
