"""Tests for the restricted-profile closed-context instruction."""
from __future__ import annotations

from src.access import AccessPolicy
from src.agy_runner import AgyResult
from src.closed_context import (
    CLOSED_CONTEXT_REPLY,
    add_closed_context_instruction,
    is_closed_context_denial,
)


def test_restricted_profile_prefixes_closed_context_instruction() -> None:
    prompt = add_closed_context_instruction("Explain sockets.", AccessPolicy())

    assert prompt.endswith("# User task\nExplain sockets.")
    assert "# Closed Context Mode" in prompt


def test_host_profile_does_not_add_closed_context_instruction() -> None:
    policy = AccessPolicy(profile="host", execution_mode="code", label="Host")

    assert add_closed_context_instruction("Inspect the host.", policy) == "Inspect the host."


def test_denied_restricted_turn_maps_to_fixed_reply() -> None:
    result = AgyResult("", 1, "agy result contains denied actions", "DENIED")

    assert is_closed_context_denial(result, AccessPolicy())
    assert CLOSED_CONTEXT_REPLY == "I can't help with questions about the execution environment."
