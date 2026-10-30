"""Structured logging: JSON lines to a rotating file, human-readable console output.

Secrets are redacted by key name before anything is rendered. WARNING and above are
also kept in a bounded in-memory buffer so the engine can surface them on the dashboard.
"""

from __future__ import annotations

import collections
import logging
import logging.handlers
import sys
import threading
from pathlib import Path
from typing import Any

import structlog

_SENSITIVE_KEY_PARTS = (
    "signature",
    "private_key",
    "api_key",
    "secret",
    "password",
    "authorization",
    "kalshi-access",
)

_recent_lock = threading.Lock()
_recent_events: collections.deque[dict[str, Any]] = collections.deque(maxlen=500)


def _redact(value: Any, depth: int = 0) -> Any:
    if depth > 4:
        return value
    if isinstance(value, dict):
        return {
            k: ("***REDACTED***" if _is_sensitive(k) else _redact(v, depth + 1))
            for k, v in value.items()
        }
    if isinstance(value, list | tuple):
        return type(value)(_redact(v, depth + 1) for v in value)
    return value


def _is_sensitive(key: Any) -> bool:
    lowered = str(key).lower()
    return any(part in lowered for part in _SENSITIVE_KEY_PARTS)


def redact_processor(_logger: Any, _name: str, event_dict: dict[str, Any]) -> dict[str, Any]:
    return _redact(event_dict)


def capture_warnings_processor(
    _logger: Any, _name: str, event_dict: dict[str, Any]
) -> dict[str, Any]:
    level = str(event_dict.get("level", "")).lower()
    if level in ("warning", "error", "critical", "exception"):
        with _recent_lock:
            _recent_events.append(
                {
                    "ts": event_dict.get("timestamp"),
                    "level": level,
                    "component": event_dict.get("component") or event_dict.get("logger"),
                    "message": str(event_dict.get("event")),
                    "details": {
                        k: str(v)[:500]
                        for k, v in event_dict.items()
                        if k not in ("timestamp", "level", "event", "logger", "component")
                    },
                }
            )
    return event_dict


def drain_recent_events() -> list[dict[str, Any]]:
    """Return and clear warnings/errors captured since the last call."""
    with _recent_lock:
        items = list(_recent_events)
        _recent_events.clear()
    return items


def configure_logging(
    level: str = "INFO", log_dir: Path | None = None, console: bool = True
) -> None:
    log_level = getattr(logging, level.upper(), logging.INFO)

    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        redact_processor,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    structlog.configure(
        # Capture runs once per structlog call (not per handler) to avoid duplicates.
        processors=[
            *shared_processors,
            capture_warnings_processor,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
    root.setLevel(log_level)

    if console:
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setFormatter(
            structlog.stdlib.ProcessorFormatter(
                foreign_pre_chain=shared_processors,
                processors=[
                    structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                    structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty()),
                ],
            )
        )
        root.addHandler(console_handler)

    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            log_dir / "kalshi-trader.jsonl",
            maxBytes=20 * 1024 * 1024,
            backupCount=10,
            encoding="utf-8",
        )
        file_handler.setFormatter(
            structlog.stdlib.ProcessorFormatter(
                foreign_pre_chain=shared_processors,
                processors=[
                    structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                    structlog.processors.JSONRenderer(default=str),
                ],
            )
        )
        root.addHandler(file_handler)

    # Third-party libraries are noisy at INFO (one line per HTTP request).
    for noisy in ("httpx", "httpcore", "websockets", "uvicorn.access"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(component: str) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(component).bind(component=component)
