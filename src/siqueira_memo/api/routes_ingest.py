"""Ingest endpoints. Plan §4.2."""

from __future__ import annotations

from fastapi import APIRouter

from siqueira_memo.api.deps import AuthDep, ProfileDep, SessionDep
from siqueira_memo.schemas import (
    ArtifactIngestIn,
    ArtifactIngestOut,
    BuiltinMemoryMirrorIn,
    DelegationObservationIn,
    GenericEventIn,
    GenericEventOut,
    HermesAuxCompactionIn,
    MessageIngestIn,
    MessageIngestOut,
    ToolEventIngestIn,
    ToolEventIngestOut,
)
from siqueira_memo.services.ingest_service import IngestService

router = APIRouter(prefix="/v1/ingest")


def _service(profile_id: str) -> IngestService:
    return IngestService(profile_id=profile_id)


@router.post("/message", response_model=MessageIngestOut)
async def ingest_message(
    payload: MessageIngestIn,
    session: SessionDep,
    profile_id: ProfileDep,
    _token: AuthDep,
) -> MessageIngestOut:
    payload.profile_id = payload.profile_id or profile_id
    return await _service(profile_id).ingest_message(session, payload)


@router.post("/tool-event", response_model=ToolEventIngestOut)
async def ingest_tool_event(
    payload: ToolEventIngestIn,
    session: SessionDep,
    profile_id: ProfileDep,
    _token: AuthDep,
) -> ToolEventIngestOut:
    payload.profile_id = payload.profile_id or profile_id
    return await _service(profile_id).ingest_tool_event(session, payload)


@router.post("/artifact", response_model=ArtifactIngestOut)
async def ingest_artifact(
    payload: ArtifactIngestIn,
    session: SessionDep,
    profile_id: ProfileDep,
    _token: AuthDep,
) -> ArtifactIngestOut:
    payload.profile_id = payload.profile_id or profile_id
    return await _service(profile_id).ingest_artifact(session, payload)


@router.post("/event", response_model=GenericEventOut)
async def ingest_event(
    payload: GenericEventIn,
    session: SessionDep,
    profile_id: ProfileDep,
    _token: AuthDep,
) -> GenericEventOut:
    payload.profile_id = payload.profile_id or profile_id
    return await _service(profile_id).ingest_event(session, payload)


@router.post("/delegation", response_model=GenericEventOut)
async def ingest_delegation(
    payload: DelegationObservationIn,
    session: SessionDep,
    profile_id: ProfileDep,
    _token: AuthDep,
) -> GenericEventOut:
    payload.profile_id = payload.profile_id or profile_id
    return await _service(profile_id).ingest_delegation(session, payload)


@router.post("/hermes-compaction", response_model=GenericEventOut)
async def ingest_hermes_compaction(
    payload: HermesAuxCompactionIn,
    session: SessionDep,
    profile_id: ProfileDep,
    _token: AuthDep,
) -> GenericEventOut:
    payload.profile_id = payload.profile_id or profile_id
    return await _service(profile_id).ingest_hermes_aux_compaction(session, payload)


@router.post("/builtin-memory-mirror", response_model=GenericEventOut)
async def ingest_builtin_memory_mirror(
    payload: BuiltinMemoryMirrorIn,
    session: SessionDep,
    profile_id: ProfileDep,
    _token: AuthDep,
) -> GenericEventOut:
    payload.profile_id = payload.profile_id or profile_id
    return await _service(profile_id).ingest_builtin_memory_mirror(session, payload)
