"""Recall endpoint. Plan §4.3."""

from __future__ import annotations

from fastapi import APIRouter

from siqueira_memo.api.deps import AuthDep, ProfileDep, SessionDep
from siqueira_memo.schemas.recall import RecallRequest, RecallResponse
from siqueira_memo.services.embedding_service import build_embedding_provider
from siqueira_memo.services.retrieval_service import RetrievalService

router = APIRouter()


@router.post("/v1/recall", response_model=RecallResponse)
async def recall(
    payload: RecallRequest,
    session: SessionDep,
    profile_id: ProfileDep,
    _token: AuthDep,
) -> RecallResponse:
    payload.profile_id = payload.profile_id or profile_id
    embedding = build_embedding_provider()
    svc = RetrievalService(
        profile_id=payload.profile_id or profile_id,
        embedding_provider=embedding,
    )
    result = await svc.recall(session, payload)
    return RecallResponse(
        context_pack=result.context_pack,
        query=payload.query,
        mode=payload.mode,
        latency_ms=result.latency_ms,
    )
