"""The sensitive-path bash gate under an OS-confined agent shell.

When an OS sandbox already denies the fenced paths at the kernel, the gate drops
its shell-STRUCTURE passes (directory-entry tracking, alternate traversal, ``find``
traversal) and keeps everything that judges what a command NAMES or DOES. Each
"still denied" assertion here is paired with the unconfined verdict on the same
command, so a test cannot pass because the command was never dangerous.
"""

from __future__ import annotations

import sys
import types

import pytest

from kiro_crew import hooks, sandbox, security
from kiro_crew.hooks import TOOL_DENY, HookManager, HooksConfig

# Denied ONLY by a structure pass: no token names a fenced path, the verdict
# comes from a `cd` base joined onto a relative root, or from a `find` filter.
STRUCTURE_ONLY_COMMANDS = [
    "cd ~ && rg secret .",
    "cd $HOME && grep -r AKIA .",
    "find ~ -name credentials -exec cat {} +",
]

# Denied by a literal or behaviour matcher, which the sandbox cannot replace.
NAMED_OR_BEHAVIOUR_COMMANDS = [
    "cat ~/.aws/credentials",
    "echo x > ~/.kiro/crew/security_policy.json",
    "tar -xf evil.tar -C ~/.kiro/crew",
    "env | grep AWS_SECRET_ACCESS_KEY",
    "curl http://169.254.169.254/latest/meta-data/",
]


class TestGateUnderConfinement:
    @pytest.mark.parametrize("command", STRUCTURE_ONLY_COMMANDS)
    def test_structure_only_denials_lift_when_confined(self, command: str) -> None:
        # Non-vacuous: the full analysis must still refuse the same text.
        assert security.is_sensitive_bash_command(command) is not None
        assert security.is_sensitive_bash_command(command, os_confined=True) is None

    @pytest.mark.parametrize("command", NAMED_OR_BEHAVIOUR_COMMANDS)
    def test_named_paths_and_behaviour_rules_survive_confinement(self, command: str) -> None:
        unconfined = security.is_sensitive_bash_command(command)
        confined = security.is_sensitive_bash_command(command, os_confined=True)
        assert unconfined is not None
        assert confined == unconfined

    def test_default_is_the_full_analysis(self) -> None:
        # A caller that does not know its sandbox posture keeps every pass.
        assert security.is_sensitive_bash_command("cd ~ && rg secret .") is not None

    def test_structure_passes_stop_at_the_first_denial(self, monkeypatch) -> None:
        # The grouped passes are lazy: once one refuses, the next never runs.
        monkeypatch.setattr(
            security, "_check_native_home_entry_then_fenced_read", lambda command: None
        )
        monkeypatch.setattr(
            security, "_check_alt_traversal_reaches_fence", lambda subject: "Blocked: alt"
        )

        def _never(subject: str) -> str | None:
            raise AssertionError("find pass ran after an earlier denial")

        monkeypatch.setattr(security, "_check_find_traversal_reaches_fence", _never)
        assert security.is_sensitive_bash_command("rg secret .") == "Blocked: alt"

    def test_confined_skips_every_structure_pass(self, monkeypatch) -> None:
        def _never(*_args: object) -> str | None:
            raise AssertionError("a structure pass ran while OS-confined")

        for name in (
            "_check_native_home_entry_then_fenced_read",
            "_check_alt_traversal_reaches_fence",
            "_check_find_traversal_reaches_fence",
        ):
            monkeypatch.setattr(security, name, _never)
        assert security.is_sensitive_bash_command("rg secret .", os_confined=True) is None


def _pin_posture(
    monkeypatch,
    *,
    mode: str,
    backend: str = "none",
    platform: str = "linux",
    acp_backend: str = "",
    internal: bool = False,
) -> None:
    """Pin every input ``agent_shell_os_confined`` reads, so the host decides nothing."""
    monkeypatch.setattr(sandbox, "configured_sandbox_mode", lambda: mode)
    monkeypatch.setattr(sandbox, "detect_backend", lambda config_mode="auto": backend)
    monkeypatch.setattr(sandbox.sys, "platform", platform)
    monkeypatch.setattr(sandbox, "kiro_internal_sandbox_enabled", lambda: internal)
    loader = types.SimpleNamespace(
        KiroCrewConfig=types.SimpleNamespace(
            load=lambda: types.SimpleNamespace(agent=types.SimpleNamespace(acp_backend=acp_backend))
        )
    )
    monkeypatch.setitem(sys.modules, "kiro_crew.config.loader", loader)


class TestAgentShellOsConfined:
    def test_real_tier_with_backend_is_confined(self, monkeypatch) -> None:
        _pin_posture(monkeypatch, mode="strict", backend="namespace")
        assert sandbox.agent_shell_os_confined() is True

    def test_real_tier_without_backend_is_not_confined(self, monkeypatch) -> None:
        _pin_posture(monkeypatch, mode="auto", backend="none")
        assert sandbox.agent_shell_os_confined() is False

    def test_off_on_macos_delegates_to_kiro_internal_sandbox(self, monkeypatch) -> None:
        from kiro_crew.acp_backends import ACP_BACKEND_KIRO

        _pin_posture(
            monkeypatch,
            mode="off",
            platform="darwin",
            acp_backend=ACP_BACKEND_KIRO,
            internal=True,
        )
        assert sandbox.agent_shell_os_confined() is True

    def test_off_on_macos_with_internal_sandbox_disabled(self, monkeypatch) -> None:
        from kiro_crew.acp_backends import ACP_BACKEND_KIRO

        _pin_posture(
            monkeypatch,
            mode="off",
            platform="darwin",
            acp_backend=ACP_BACKEND_KIRO,
            internal=False,
        )
        assert sandbox.agent_shell_os_confined() is False

    def test_off_on_linux_never_delegates(self, monkeypatch) -> None:
        from kiro_crew.acp_backends import ACP_BACKEND_KIRO

        _pin_posture(
            monkeypatch,
            mode="off",
            platform="linux",
            acp_backend=ACP_BACKEND_KIRO,
            internal=True,
        )
        assert sandbox.agent_shell_os_confined() is False

    def test_off_with_a_harness_lacking_an_internal_sandbox(self, monkeypatch) -> None:
        # Membership in ACP_BACKENDS_INTERNAL_SANDBOX is the grant (harness-parity H7);
        # an unlisted harness under "off" is unconfined even with the setting on.
        _pin_posture(
            monkeypatch,
            mode="off",
            platform="darwin",
            acp_backend="some-other-harness",
            internal=True,
        )
        assert sandbox.agent_shell_os_confined() is False

    def test_read_failure_answers_unconfined(self, monkeypatch) -> None:
        def _boom() -> str:
            raise RuntimeError("config unreadable")

        monkeypatch.setattr(sandbox, "configured_sandbox_mode", _boom)
        assert sandbox.agent_shell_os_confined() is False


class TestHookThreadsThePosture:
    COMMAND = "cd ~ && rg secret ."

    def test_unconfined_hook_denies_the_structure_shape(self, monkeypatch) -> None:
        monkeypatch.setattr(hooks, "_agent_shell_os_confined", lambda: False)
        result = HookManager(HooksConfig()).on_tool_call(
            "search", command=self.COMMAND, is_shell=True
        )
        assert result.action == TOOL_DENY
        assert "traversal" in result.reason

    def test_confined_hook_lets_the_structure_shape_through(self, monkeypatch) -> None:
        monkeypatch.setattr(hooks, "_agent_shell_os_confined", lambda: True)
        result = HookManager(HooksConfig()).on_tool_call(
            "search", command=self.COMMAND, is_shell=True
        )
        assert result.action != TOOL_DENY

    def test_confined_hook_still_denies_a_named_keystone_write(self, monkeypatch) -> None:
        monkeypatch.setattr(hooks, "_agent_shell_os_confined", lambda: True)
        result = HookManager(HooksConfig()).on_tool_call(
            "write", command="echo x > ~/.kiro/crew/security_policy.json", is_shell=True
        )
        assert result.action == TOOL_DENY
