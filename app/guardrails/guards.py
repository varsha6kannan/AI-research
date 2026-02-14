"""Build Guardrails AI Guards for RAG input and output validation."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from app.config import settings

if TYPE_CHECKING:
    from guardrails import Guard

logger = logging.getLogger(__name__)

# Module-level cache so we build each Guard once per process.
_output_guard: "Guard | None" = None
_input_guard: "Guard | None" = None


def get_output_guard() -> "Guard | None":
    """
    Build a Guard for validating the LLM response string (the 'response' field content).
    Uses DetectPII to block or redact PII in answers.
    Returns None if guardrails_output_enabled is False or validators are not available.
    """
    global _output_guard
    if not settings.guardrails_output_enabled:
        return None
    if _output_guard is not None:
        return _output_guard
    try:
        from guardrails import Guard
        from guardrails.hub import DetectPII

        # Detect PII in response; on_fail=exception so pipeline catches and returns fallback.
        _output_guard = Guard().use(
            DetectPII(pii_entities="pii", on_fail="exception"),
        )
        return _output_guard
    except ImportError as e:
        logger.debug(
            "Guardrails output guard not built (missing DetectPII?): %s. "
            "Run: guardrails hub install hub://guardrails/detect_pii",
            e,
        )
        return None


def get_input_guard() -> "Guard | None":
    """
    Build a Guard for validating the user question before retrieval.
    Uses DetectPII to block questions that contain PII.
    Returns None if guardrails_input_enabled is False or validators not available.
    """
    global _input_guard
    if not settings.guardrails_input_enabled:
        return None
    if _input_guard is not None:
        return _input_guard
    try:
        from guardrails import Guard
        from guardrails.hub import DetectPII

        # Block questions containing PII.
        _input_guard = Guard().use(
            DetectPII(pii_entities="pii", on_fail="exception"),
        )
        return _input_guard
    except ImportError as e:
        logger.debug(
            "Guardrails input guard not built (missing DetectPII?): %s. "
            "Run: guardrails hub install hub://guardrails/detect_pii",
            e,
        )
        return None
