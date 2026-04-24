"""Eval run records."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import DateTime, Float, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from siqueira_memo.models.base import Base, UUIDPrimaryKey
from siqueira_memo.models.types import JSONB


class EvalRun(UUIDPrimaryKey, Base):
    __tablename__ = "eval_runs"

    profile_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    suite: Mapped[str] = mapped_column(String(64), nullable=False)
    question: Mapped[str] = mapped_column(Text(), nullable=False)
    passed: Mapped[bool] = mapped_column(nullable=False, default=False)
    score: Mapped[float] = mapped_column(Float(), nullable=False, default=0.0)
    expected_contains: Mapped[list[str]] = mapped_column(JSONB(), nullable=False, default=list)
    missing_terms: Mapped[list[str]] = mapped_column(JSONB(), nullable=False, default=list)
    candidates_returned: Mapped[int] = mapped_column(Integer(), nullable=False, default=0)
    latency_ms: Mapped[int] = mapped_column(Integer(), nullable=False, default=0)
    extra_metadata: Mapped[dict[str, Any]] = mapped_column(JSONB(), nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
