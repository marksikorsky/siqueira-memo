"""Structured JSON logging for Siqueira Memo.

Logs never contain raw secrets. See plan §10.1: redaction counts are logged but
not the redacted content itself. Callers should use :func:`get_logger` and
never call ``logging.getLogger`` directly for service code.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from typing import Any


class _JSONFormatter(logging.Formatter):
    """Minimal JSON formatter that tolerates extra structured fields."""

    _reserved = {
        "name",
        "msg",
        "args",
        "levelname",
        "levelno",
        "pathname",
        "filename",
        "module",
        "exc_info",
        "exc_text",
        "stack_info",
        "lineno",
        "funcName",
        "created",
        "msecs",
        "relativeCreated",
        "thread",
        "threadName",
        "processName",
        "process",
        "message",
        "taskName",
    }

    def format(self, record: logging.LogRecord) -> str:  # noqa: D401
        payload: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created))
            + f".{int(record.msecs):03d}Z",
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key in self._reserved or key.startswith("_"):
                continue
            payload[key] = _safe(value)
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


def _safe(value: Any) -> Any:
    try:
        json.dumps(value, ensure_ascii=False, default=str)
        return value
    except (TypeError, ValueError):
        return repr(value)


_configured = False


def configure_logging(level: str = "INFO") -> None:
    """Install a single JSON handler on the root logger.

    Idempotent: repeated calls reconfigure the level but do not duplicate
    handlers. Tests can freely call it.
    """
    global _configured
    root = logging.getLogger()
    root.setLevel(level.upper())
    if _configured:
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(_JSONFormatter())
    # Remove default handlers first so we do not double-log in test runners.
    for h in list(root.handlers):
        root.removeHandler(h)
    root.addHandler(handler)
    _configured = True


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
