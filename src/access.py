"""Server-owned execution policy for the bridge worker.

The Telegram configuration and chat messages never select a profile. The
root-owned service configuration provides these values to the worker process,
and they are deliberately omitted from the environment inherited by ``agy``.
"""
from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path


_PROFILES = frozenset({"sandbox", "workspace", "host"})
_EXECUTION_MODES = frozenset({"plan", "code"})
_POLICY_PATH = Path("/etc/agy-telegram-bridge/access.env")


class AccessPolicyError(ValueError):
    """Raised when the server service provides an unsafe access policy."""


@dataclass(frozen=True)
class AccessPolicy:
    """An access profile selected only by the server service manager."""

    profile: str = "sandbox"
    execution_mode: str = "plan"
    label: str = "Sandbox"
    allowed_execution_modes: frozenset[str] = frozenset({"plan"})

    @property
    def permits_unattended_tools(self) -> bool:
        """Whether this server-approved turn may auto-approve sandboxed tools."""
        return self.execution_mode == "code" and "code" in self.allowed_execution_modes

    def for_execution_mode(self, requested: str) -> "AccessPolicy":
        """Return an allowed per-chat execution choice without changing profile."""
        if requested not in self.allowed_execution_modes:
            raise AccessPolicyError("requested execution mode is not allowed by the server")
        return replace(self, execution_mode=requested)


def load_access_policy(environ: Mapping[str, str] | None = None) -> AccessPolicy:
    """Load systemd policy, defaulting safely to sandboxed planning.

    This is deliberately separate from ``config.json``: that file is readable
    by the Telegram controller and must never affect worker privileges.
    """
    values = dict(os.environ if environ is None else environ)
    if values.get("AGY_BRIDGE_ACCESS_POLICY_PATH"):
        values.update(_read_server_policy(values["AGY_BRIDGE_ACCESS_POLICY_PATH"]))
    profile = values.get("AGY_BRIDGE_ACCESS_PROFILE", "sandbox").strip().lower()
    if profile not in _PROFILES:
        raise AccessPolicyError(
            "AGY_BRIDGE_ACCESS_PROFILE must be sandbox, workspace, or host"
        )

    configured_mode = values.get("AGY_BRIDGE_EXECUTION_MODE", "plan").strip().lower()
    allowed_raw = values.get("AGY_BRIDGE_ALLOWED_EXECUTION_MODES", configured_mode)
    allowed_execution_modes = frozenset(
        value.strip().lower() for value in allowed_raw.split(",") if value.strip()
    )
    if configured_mode not in _EXECUTION_MODES or not allowed_execution_modes <= _EXECUTION_MODES:
        raise AccessPolicyError("execution modes must be plan or code")
    if not allowed_execution_modes or configured_mode not in allowed_execution_modes:
        raise AccessPolicyError("default execution mode must be allowed by the server")
    # Workspace and Host access are deliberately never adjustable from Telegram.
    # Sandbox alone can opt in, in its root-owned policy, to per-chat code mode.
    if profile == "workspace" and (
        configured_mode != "plan" or allowed_execution_modes != {"plan"}
    ):
        raise AccessPolicyError("workspace profile must use server-enforced plan execution")
    if profile == "host" and allowed_execution_modes != {configured_mode}:
        raise AccessPolicyError("host execution mode must be server-enforced")

    label = values.get("AGY_BRIDGE_ACCESS_LABEL", profile.title()).strip()
    if not label or len(label) > 80 or any(ord(char) < 32 for char in label):
        raise AccessPolicyError(
            "AGY_BRIDGE_ACCESS_LABEL must be printable and at most 80 characters"
        )
    return AccessPolicy(
        profile=profile, execution_mode=configured_mode, label=label,
        allowed_execution_modes=allowed_execution_modes,
    )


def _read_server_policy(raw_path: str) -> dict[str, str]:
    """Read a root-owned, non-secret policy file without shell evaluation."""
    path = Path(raw_path)
    if path != _POLICY_PATH:
        raise AccessPolicyError("access policy must use the fixed server path")
    try:
        meta = path.stat()
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise AccessPolicyError("server access policy is unavailable") from exc
    if meta.st_uid != 0 or meta.st_mode & 0o022:
        raise AccessPolicyError("server access policy must be root-owned and non-writable")
    parsed: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, sep, value = line.partition("=")
        if not sep or key not in {
            "AGY_BRIDGE_ACCESS_PROFILE", "AGY_BRIDGE_EXECUTION_MODE",
            "AGY_BRIDGE_ALLOWED_EXECUTION_MODES", "AGY_BRIDGE_ACCESS_LABEL",
        }:
            raise AccessPolicyError("server access policy contains an invalid entry")
        parsed[key] = value.strip()
    return parsed
