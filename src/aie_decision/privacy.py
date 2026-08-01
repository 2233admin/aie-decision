"""Privacy and secret guards for logs, exports, and provider boundaries."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping


_SECRET_KEYS = re.compile(r"(?:api[_-]?key|authorization|password|passwd|secret|token|cookie)", re.IGNORECASE)
_SECRET_TEXT = (
    re.compile(r"(?i)(bearer\s+)[A-Za-z0-9._~+/=-]+"),
    re.compile(r"(?i)((?:api[_-]?key|password|secret|token)\s*[:=]\s*)[^\s,;]+"),
)


def redact_text(text: str) -> str:
    result = text
    for pattern in _SECRET_TEXT:
        result = pattern.sub(r"\1[REDACTED]", result)
    return result


def sanitize(value: Any) -> Any:
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, Mapping):
        return {
            str(key): "[REDACTED]" if _SECRET_KEYS.search(str(key)) else sanitize(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [sanitize(item) for item in value]
    if isinstance(value, tuple):
        return tuple(sanitize(item) for item in value)
    return value


@dataclass(frozen=True, slots=True)
class ProviderBoundary:
    allowed_fields: frozenset[str]

    def prepare(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        disallowed = sorted(set(payload) - self.allowed_fields)
        if disallowed:
            raise ValueError(f"provider payload contains disallowed fields: {', '.join(disallowed)}")
        cleaned = sanitize(payload)
        assert isinstance(cleaned, dict)
        return cleaned
