"""Bounded, non-sensitive diagnostics for failed agy subprocesses."""
from __future__ import annotations

from src.agy_runner import AgyResult


SAFE_FAILURE_REASONS = frozenset({
    "authentication unavailable",
    "selected model is unavailable",
    "agy launch configuration rejected",
    "worker runtime configuration unavailable",
    "worker permission policy rejected the turn",
    "model service rejected the turn",
    "isolated runtime storage unavailable",
    "worker filesystem policy blocked agy",
    "worker network access failed",
    "agy returned invalid output",
    "agy timed out",
    "agy execution failed",
})


def safe_failure_reason(result: AgyResult) -> str:
    """Classify a failure without returning any CLI diagnostic text.

    CLI stderr and terminal errors can include credentials, prompts, and local
    paths. Consumers can log or transfer only the fixed values in
    ``SAFE_FAILURE_REASONS``.
    """
    detail = result.stderr.lower()
    if result.exit_code == 124 or result.status == "TIMEOUT":
        return "agy timed out"
    if result.status == "DENIED" or "denied actions" in detail:
        return "worker permission policy rejected the turn"
    if any(marker in detail for marker in (
        "oauth", "authentication", "unauthenticated", "not logged in",
        "credential", "login required",
    )):
        return "authentication unavailable"
    if any(marker in detail for marker in (
        "invalid model selection", "model is not recognized", "unknown model",
    )):
        return "selected model is unavailable"
    if any(marker in detail for marker in (
        "unknown flag", "unknown option", "flag provided but not defined",
        "invalid argument", "cannot be used", "cannot be combined",
        "input format", "output format",
    )):
        return "agy launch configuration rejected"
    if any(marker in detail for marker in (
        "settings", "configuration", "agent definition", "workspace", "project",
    )):
        return "worker runtime configuration unavailable"
    if any(marker in detail for marker in (
        "read-only file system", "permission denied", "eacces", "enoent",
        "no such file", "cannot create", "mkdir",
    )):
        return "isolated runtime storage unavailable"
    if any(marker in detail for marker in (
        "operation not permitted", "not permitted", "seccomp", "sandbox",
    )):
        return "worker filesystem policy blocked agy"
    if any(marker in detail for marker in (
        "network is unreachable", "connection refused", "getaddrinfo",
        "could not resolve", "dns", "tls", "certificate",
    )):
        return "worker network access failed"
    if any(marker in detail for marker in (
        "permission", "approval", "not allowed", "not authorized",
    )):
        return "worker permission policy rejected the turn"
    if any(marker in detail for marker in (
        "quota", "rate limit", "resource exhausted", "capacity", "overloaded",
    )):
        return "model service rejected the turn"
    if result.status == "INVALID":
        return "agy returned invalid output"
    return "agy execution failed"
