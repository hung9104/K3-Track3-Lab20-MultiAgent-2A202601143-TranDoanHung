"""Logging configuration with best-effort secret redaction."""

import logging
import re
from typing import Any

SENSITIVE_KEY = re.compile(r"(?i)(api[_-]?key|authorization|password|secret|token)")
SECRET_VALUE = re.compile(r"(?i)\b(sk-[a-z0-9_-]{8,}|bearer\s+[a-z0-9._-]{8,})\b")
REDACTED = "[REDACTED]"


def redact(value: Any, key: str | None = None) -> Any:
    """Recursively remove common credentials before logging or exporting traces."""

    if key and SENSITIVE_KEY.search(key):
        return REDACTED
    if isinstance(value, dict):
        return {str(item_key): redact(item, str(item_key)) for item_key, item in value.items()}
    if isinstance(value, list):
        return [redact(item) for item in value]
    if isinstance(value, tuple):
        return tuple(redact(item) for item in value)
    if isinstance(value, str):
        return SECRET_VALUE.sub(REDACTED, value)
    return value


class RedactingFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = redact(record.msg)
        if record.args:
            record.args = redact(record.args)
        return True


def configure_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler()
    handler.addFilter(RedactingFilter())
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s - %(message)s")
    )
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        handlers=[handler],
        force=True,
    )
