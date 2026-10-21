"""Server-owned execution policy for the bridge worker.

The Telegram configuration and chat messages never select a profile. The
root-owned service configuration provides these values to the worker process,
and they are deliberately omitted from the environment inherited by ``agy``.
"""
from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path


_PROFILES = frozenset({"sandbox", "workspace", "host"})
_HOST_EXECUTION_MODES = frozenset({"plan", "code"})
_POLICY_PATH = Path("/etc/agy-telegram-bridge/access.env")


class AccessPolicyError(ValueError):
    """Raised when the server service provides an unsafe access policy."""


@dataclass(frozen=True)
class AccessPolicy:
    """An access profile selected only by the server service manager."""

    profile: str = "sandbox"
    execution_mode: str = "plan"
    label: str = "Sandbox"

    @property
    def permits_unattended_tools(self) -> bool:
        """Only the administrator-selected Host profile may auto-approve tools."""
        return self.profile == "host" and self.execution_mode == "code"


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
    if profile in {"sandbox", "workspace"}:
        if configured_mode != "plan":
            raise AccessPolicyError(
                f"{profile} profile must use server-enforced plan execution"
            )
        execution_mode = "plan"
    else:
        if configured_mode not in _HOST_EXECUTION_MODES:
            raise AccessPolicyError("host execution mode must be plan or code")
        execution_mode = configured_mode

    label = values.get("AGY_BRIDGE_ACCESS_LABEL", profile.title()).strip()
    if not label or len(label) > 80 or any(ord(char) < 32 for char in label):
        raise AccessPolicyError(
            "AGY_BRIDGE_ACCESS_LABEL must be printable and at most 80 characters"
        )
    return AccessPolicy(profile=profile, execution_mode=execution_mode, label=label)


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
            "AGY_BRIDGE_ACCESS_LABEL",
        }:
            raise AccessPolicyError("server access policy contains an invalid entry")
        parsed[key] = value.strip()
    return parsed
