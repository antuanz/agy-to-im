"""Tests for the restricted-profile closed-context instruction."""
from __future__ import annotations

from src.access import AccessPolicy
from src.agy_runner import AgyResult
from src.closed_context import (
    CLOSED_CONTEXT_REPLY,
    RESTRICTED_ACTION_REPLY,
    add_closed_context_instruction,
    is_execution_environment_request,
    is_restricted_policy_denial,
)


def test_restricted_profile_prefixes_closed_context_instruction() -> None:
    prompt = add_closed_context_instruction("Explain sockets.", AccessPolicy())

    assert prompt.endswith("# User task\nExplain sockets.")
    assert "# Closed Context Mode" in prompt
    assert "bridge uploads newly created eligible files" in prompt


def test_host_profile_does_not_add_closed_context_instruction() -> None:
    policy = AccessPolicy(profile="host", execution_mode="code", label="Host")

    assert add_closed_context_instruction("Inspect the host.", policy) == "Inspect the host."


def test_denied_restricted_turn_maps_to_fixed_reply() -> None:
    result = AgyResult("", 1, "agy result contains denied actions", "DENIED")

    assert is_restricted_policy_denial(result, AccessPolicy())
    assert CLOSED_CONTEXT_REPLY == "I can't help with questions about the execution environment."


def test_only_environment_requests_use_the_closed_context_fallback() -> None:
    assert is_execution_environment_request("What free disk space is on this VM?")
    assert is_execution_environment_request("Run ls -la /home/opengym exactly.")
    assert not is_execution_environment_request("I need a JPEG picture of a lemon tree.")
    assert "restricted chat profile" in RESTRICTED_ACTION_REPLY
