"""Server-managed access-policy tests."""
from __future__ import annotations

import pytest

from src.access import AccessPolicy, AccessPolicyError, load_access_policy
from src.agy_runner import _build_args, _child_environment


def test_default_policy_is_sandboxed_plan() -> None:
    policy = load_access_policy({})
    assert policy == AccessPolicy()
    assert not policy.permits_unattended_tools


def test_workspace_cannot_select_code_execution() -> None:
    with pytest.raises(AccessPolicyError, match="workspace profile"):
        load_access_policy({
            "AGY_BRIDGE_ACCESS_PROFILE": "workspace",
            "AGY_BRIDGE_EXECUTION_MODE": "code",
        })


def test_sandbox_policy_can_allow_code_without_changing_profile() -> None:
    policy = load_access_policy({
        "AGY_BRIDGE_ACCESS_PROFILE": "sandbox",
        "AGY_BRIDGE_EXECUTION_MODE": "code",
        "AGY_BRIDGE_ALLOWED_EXECUTION_MODES": "plan,code",
        "AGY_BRIDGE_ACCESS_LABEL": "Sandbox execution",
    })
    assert policy.profile == "sandbox"
    assert policy.for_execution_mode("plan").execution_mode == "plan"
    args = _build_args(
        agy_path="/usr/bin/agy", prompt="x", conversation_id="", model="",
        mode="code", print_timeout="15m", access=policy,
    )
    assert "--dangerously-skip-permissions" in args
    assert args[args.index("--mode") + 1] == "accept-edits"


def test_server_locked_policy_rejects_unapproved_chat_mode() -> None:
    policy = load_access_policy({})
    with pytest.raises(AccessPolicyError, match="not allowed"):
        policy.for_execution_mode("code")


def test_host_execution_is_not_switchable_from_telegram() -> None:
    with pytest.raises(AccessPolicyError, match="server-enforced"):
        load_access_policy({
            "AGY_BRIDGE_ACCESS_PROFILE": "host",
            "AGY_BRIDGE_EXECUTION_MODE": "code",
            "AGY_BRIDGE_ALLOWED_EXECUTION_MODES": "plan,code",
        })


def test_child_environment_excludes_controller_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CREDENTIALS_DIRECTORY", "/run/credentials/controller")
    monkeypatch.setenv("AGY_TELEGRAM_BOT_TOKEN", "not-for-child")
    monkeypatch.setenv("AGY_BRIDGE_WORKER_SOCKET", "/run/worker.sock")
    child = _child_environment()
    assert "CREDENTIALS_DIRECTORY" not in child
    assert "AGY_TELEGRAM_BOT_TOKEN" not in child
    assert "AGY_BRIDGE_WORKER_SOCKET" not in child
