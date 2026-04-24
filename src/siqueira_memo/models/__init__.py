"""ORM model package.

Importing this package registers every ``siqueira_memo`` table on the shared
declarative ``Base.metadata`` so Alembic autogenerate and
``Base.metadata.create_all`` see them.
"""

from __future__ import annotations

from siqueira_memo.models.artifacts import Artifact
from siqueira_memo.models.base import Base
from siqueira_memo.models.chunks import (
    EMBEDDING_TABLE_BY_MODEL,
    Chunk,
    ChunkEmbeddingBGEM3,
    ChunkEmbeddingMock,
    ChunkEmbeddingOpenAITEL3,
    EmbeddingIndex,
)
from siqueira_memo.models.decisions import Decision, DecisionSource
from siqueira_memo.models.entities import Entity, EntityAlias, EntityRelationship
from siqueira_memo.models.evals import EvalRun
from siqueira_memo.models.events import MemoryEvent
from siqueira_memo.models.facts import Fact, FactSource
from siqueira_memo.models.messages import Message
from siqueira_memo.models.retrieval import MemoryConflict, PromptVersion, RetrievalLog
from siqueira_memo.models.summaries import ProjectState, SessionSummary, TopicSummary
from siqueira_memo.models.tools import ToolEvent

__all__ = [
    "Artifact",
    "Base",
    "Chunk",
    "ChunkEmbeddingBGEM3",
    "ChunkEmbeddingMock",
    "ChunkEmbeddingOpenAITEL3",
    "Decision",
    "DecisionSource",
    "EmbeddingIndex",
    "EMBEDDING_TABLE_BY_MODEL",
    "Entity",
    "EntityAlias",
    "EntityRelationship",
    "EvalRun",
    "Fact",
    "FactSource",
    "MemoryConflict",
    "MemoryEvent",
    "Message",
    "ProjectState",
    "PromptVersion",
    "RetrievalLog",
    "SessionSummary",
    "TopicSummary",
    "ToolEvent",
]
