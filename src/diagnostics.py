"""Bounded, non-sensitive diagnostics for failed agy subprocesses."""
from __future__ import annotations

from src.agy_runner import AgyResult


SAFE_FAILURE_REASONS = frozenset({
    "authentication unavailable",
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
    if any(marker in detail for marker in (
        "oauth", "authentication", "unauthenticated", "not logged in",
        "credential", "login required",
    )):
        return "authentication unavailable"
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
    if result.status == "INVALID":
        return "agy returned invalid output"
    return "agy execution failed"
