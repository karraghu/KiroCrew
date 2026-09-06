"""Enforcement of the ``approval_modes`` policy at every ``yolo`` arming surface.

``test_approval_modes_governance.py`` pins the *predicate*
(``approval_mode_permitted`` reading the ceiling on the active context). This module
pins the places that must CONSULT it, because a mode the policy denies is only
actually off if every path that can arm it -- and every path that HONOURS an existing
grant -- refuses:

* ``POST /api/chat/mode`` -- the explicit session-mode switch (403, no mutation).
* ``safety_override`` arming -- session-wide and scoped.
* ``is_active`` / ``is_scope_active`` / ``renew_scoped`` -- the consult points that
  honour a LIVE grant.

It also pins WHEN the verdict is computed, which is the design property the rest of
the module rests on: ``approval_mode_permitted("yolo")`` is resolved ONCE per ceiling,
pushed by a hook on ``platform.context._install``, and every consumer then reads a
module-level flag with no filesystem access, no TTL and no third "not resolved yet"
state. A deny arriving that way destroys live grants at the moment it is installed
rather than when something next asks.

The scope governs ``yolo`` only. ``trust`` / ``trust_reads`` are non-deniable (a
policy naming them is refused at parse time), because their consumption predicates
are not gated -- so there is deliberately nothing here asserting Trust enforcement.

Every refusal must also be SEL-audited: a governance denial that leaves no trace is
indistinguishable from the request never having been made, which is exactly the
record an operator needs after an attempted escalation.
"""

from __future__ import annotations

import asyncio
import dataclasses
import threading
from unittest.mock import AsyncMock, MagicMock

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from kiro_crew import safety_override as so_mod
from kiro_crew.dashboard.state import DashboardState
from kiro_crew.history import ConversationLog
from kiro_crew.platform import context as ctx_mod
from kiro_crew.platform import governance_profiles as gp
from kiro_crew.platform.bootstrap import build_default_context
from kiro_crew.platform.governance import parse_policy


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    d = tmp_path / "profiles"
    d.mkdir()
    monkeypatch.setattr(gp, "_PROFILES_DIR", d)
    # Reset on BOTH sides, and reset the context too. Every piece of state these
    # cases touch is module-level: the profile store, the active context, and the
    # pushed YOLO verdict. Resetting only on teardown leaves each case at the mercy
    # of whatever ran before it in the same worker, which is an order-dependent
    # result that says nothing about the code under test. Cleaning up on entry is
    # what makes a case mean the same thing alone and inside the file.
    gp.reset_store()
    ctx_mod.reset_context()
    so_mod.reset_yolo_policy_state()
    so_mod.reset_singleton()
    yield
    so_mod.reset_yolo_policy_state()
    so_mod.reset_singleton()
    gp.reset_store()
    ctx_mod.reset_context()


def _install_no_policy() -> None:
    """Install a context with NO governance ceiling: every mode permitted."""
    from kiro_crew.config.loader import KiroCrewConfig

    ctx_mod.set_context(
        dataclasses.replace(build_default_context(KiroCrewConfig.load()), governance=None)
    )


def _deny(*modes: str) -> None:
    """Install a ceiling denying ``modes``.

    This is the push: ``set_context`` resolves the verdict and revokes live grants
    before it returns, so nothing in a case needs to nudge a cache afterwards.
    """
    from kiro_crew.config.loader import KiroCrewConfig

    base = build_default_context(KiroCrewConfig.load())
    ceiling = parse_policy(
        {
            "version": 1,
            "boot": {"fail_closed": True},
            "approval_modes": {"mode": "deny", "deny": list(modes)},
        }
    )
    ctx_mod.set_context(dataclasses.replace(base, governance=ceiling))


def _install_unrelated_ceiling() -> None:
    """Install a DIFFERENT ceiling that still permits ``yolo``.

    Names a real scope rather than another mode: ``approval_modes`` may only deny
    ``yolo`` (the other three are non-deniable and a policy naming one is refused at
    parse time), so an install that changes something without touching this verdict
    has to change a different scope.
    """
    from kiro_crew.config.loader import KiroCrewConfig

    base = build_default_context(KiroCrewConfig.load())
    ceiling = parse_policy(
        {
            "version": 1,
            "boot": {"fail_closed": True},
            "yolo_duration": {"mode": "deny", "deny": ["permanent"]},
        }
    )
    ctx_mod.set_context(dataclasses.replace(base, governance=ceiling))


def _make_state(tmp_path) -> DashboardState:
    sessions = MagicMock(count=0)
    sessions.get_pid = MagicMock(return_value=None)
    sessions.remove = AsyncMock()
    return DashboardState(
        sessions=sessions,
        crons=MagicMock(list_jobs=MagicMock(return_value=[]), status=MagicMock(return_value={})),
        lessons=MagicMock(load_all=MagicMock(return_value=[])),
        start_time=0.0,
        conversation_log=ConversationLog(base_dir=tmp_path),
    )


def _make_app(state: DashboardState) -> web.Application:
    from kiro_crew.dashboard.chat import api_chat_mode, api_chat_slot_approve

    @web.middleware
    async def _test_auth(request: web.Request, handler):
        if "app" not in request:
            request["app"] = ""
        if "user" not in request:
            request["user"] = "local-app"
        return await handler(request)

    app = web.Application(middlewares=[_test_auth])
    app["state"] = state
    app.router.add_post("/api/chat/slots/{slot}/approve", api_chat_slot_approve)
    app.router.add_post("/api/chat/mode", api_chat_mode)
    return app


class TestChatModeEndpointRefusalIsAudited:
    @pytest.mark.asyncio
    async def test_mode_switch_refusal_is_sel_audited(self, tmp_path, monkeypatch):
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        _deny("yolo")
        recorded: list[dict] = []
        fake_sel = MagicMock()
        fake_sel.log_api_access = lambda **kw: recorded.append(kw)
        monkeypatch.setattr("kiro_crew.dashboard.chat_handlers.sel", lambda: fake_sel)
        state = _make_state(tmp_path)
        state.get_or_create_slot("s1")

        async with TestClient(TestServer(_make_app(state))) as client:
            resp = await client.post("/api/chat/mode", json={"mode": "yolo", "slot": "s1"})
            data = await resp.json()

        assert resp.status == 403
        assert data["code"] == "mode_disabled_by_policy"
        assert recorded[-1]["outcome"] == "approval_mode_denied_by_policy"

    @pytest.mark.asyncio
    async def test_a_governance_error_is_also_a_refusal(self, tmp_path, monkeypatch):
        """An unreadable policy denies. There is no third answer to report.

        While the verdict was polled behind a TTL it could also be "not resolved
        under the installed ceiling yet", which this endpoint reported as a
        transient 503 so an operator was not sent hunting a policy that may not
        exist. A pushed verdict has no such state: the resolve happens at install
        time and an error there fails CLOSED, so the honest answer at request time
        is the same refusal a real deny gets.
        """
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)

        def _boom(mode: str) -> bool:
            raise RuntimeError("governance unavailable")

        monkeypatch.setattr(so_mod, "approval_mode_permitted", _boom)
        _install_no_policy()  # a permissive ceiling; the RESOLVE is what fails
        state = _make_state(tmp_path)
        state.get_or_create_slot("s1")

        async with TestClient(TestServer(_make_app(state))) as client:
            resp = await client.post("/api/chat/mode", json={"mode": "yolo", "slot": "s1"})
            data = await resp.json()

        assert resp.status == 403, "a policy nobody can read must not select yolo"
        assert data["code"] == "mode_disabled_by_policy", data


class TestSafetyOverrideRefusalIsAudited:
    def test_denied_yolo_arming_is_audited_and_refused(self, monkeypatch):
        from kiro_crew import safety_override as so

        _deny("yolo")
        so.reset_singleton()
        recorded: list[dict] = []
        fake_sel = MagicMock()
        fake_sel.log_api_access = lambda **kw: recorded.append(kw)
        monkeypatch.setattr(so, "sel", lambda: fake_sel)

        result = so.safety_override().activate("dashboard")

        assert result.active is False
        assert recorded, "a refused arming must leave a security-event trace"
        assert recorded[-1]["outcome"] == "denied"
        assert "approval_modes" in recorded[-1]["resources"]

    def test_scoped_arming_is_refused_and_audited_too(self, monkeypatch):
        """A narrow scoped grant is still an auto-approve grant."""
        from kiro_crew import safety_override as so

        _deny("yolo")
        so.reset_singleton()
        recorded: list[dict] = []
        fake_sel = MagicMock()
        fake_sel.log_api_access = lambda **kw: recorded.append(kw)
        monkeypatch.setattr(so, "sel", lambda: fake_sel)

        result = so.safety_override().activate_scoped("session:abc", "dashboard")

        assert result.active is False
        assert recorded[-1]["outcome"] == "denied"
        assert "session:abc" in recorded[-1]["resources"]


class TestTheVerdictIsResolvedAtCeilingInstall:
    """The push, stated directly: an install is what computes the verdict.

    Every consumer then reads a flag. That ordering is what removes the windows a
    polled cache had -- a freshness key, an "unknown" third state, and a per-caller
    rule for collapsing it -- because a flag is either the answer for the ceiling in
    force or there is no ceiling installed at all.
    """

    def test_a_deny_install_revokes_a_live_grant_and_fires_the_callback_once(self, monkeypatch):
        from kiro_crew import safety_override as so

        _install_no_policy()
        so.reset_singleton()
        monkeypatch.setattr(so, "sel", lambda: MagicMock())
        override = so.safety_override()
        assert override.activate("dashboard").active is True
        seen: list[str] = []
        override.on_expired = seen.append

        _deny("yolo")

        assert seen == ["policy"], (
            "the revocation must run when the ceiling is installed, not when "
            "something next consults the predicate"
        )
        assert override.has_grant() is False, "a deny DESTROYS the grant, it does not mask it"
        # The consult points agree, and consulting them repeatedly cannot re-fire a
        # teardown that broadcasts and rewrites every slot's approval policy.
        for _ in range(5):
            assert override.is_active() is False
        assert seen == ["policy"], f"expected exactly one teardown, got {seen}"

    def test_a_permitting_install_leaves_a_live_grant_alone(self, monkeypatch):
        """A ceiling install is not itself a revocation.

        Central distribution re-installs a ceiling on every refresh, most of which
        change nothing. Tearing a grant down on any install would make an unattended
        run lose its authority on an unrelated poll.
        """
        from kiro_crew import safety_override as so

        _install_no_policy()
        so.reset_singleton()
        monkeypatch.setattr(so, "sel", lambda: MagicMock())
        override = so.safety_override()
        override.activate("dashboard")
        assert override.activate_scoped("run:abc", "dashboard").active is True
        seen: list[str] = []
        override.on_expired = seen.append

        _install_unrelated_ceiling()  # a new ceiling that says nothing about yolo

        assert seen == [], "an install that still permits yolo must revoke nothing"
        assert override.is_active() is True
        assert override.is_scope_active("run:abc") is True

    def test_a_raising_governance_layer_resolves_to_denied(self, monkeypatch):
        """Fail CLOSED. The alternative is auto-approving against unreadable policy."""
        from kiro_crew import safety_override as so

        def _boom(mode: str) -> bool:
            raise RuntimeError("governance unavailable")

        monkeypatch.setattr(so, "approval_mode_permitted", _boom)

        _install_no_policy()

        assert so.yolo_policy_permits() is False
        assert so.cached_disabled_approval_modes() == ["yolo"], (
            "reporting nothing-denied on a resolve error would unhide a mode the "
            "gate is refusing"
        )

    def test_a_raising_governance_layer_still_revokes_a_live_grant(self, monkeypatch):
        from kiro_crew import safety_override as so

        _install_no_policy()
        so.reset_singleton()
        monkeypatch.setattr(so, "sel", lambda: MagicMock())
        override = so.safety_override()
        override.activate("dashboard")
        seen: list[str] = []
        override.on_expired = seen.append

        def _boom(mode: str) -> bool:
            raise RuntimeError("governance unavailable")

        monkeypatch.setattr(so, "approval_mode_permitted", _boom)
        _install_no_policy()

        assert override.has_grant() is False
        assert seen == ["policy"]

    def test_the_first_read_without_an_install_does_not_leave_yolo_dead(self, monkeypatch):
        """A process that never booted the platform must still be able to arm.

        The flag starts DENIED so a failed bootstrap cannot hand out auto-approve,
        which makes the bootstrap itself load-bearing: reading the verdict with no
        ceiling installed has to resolve one (``current_context`` composes and
        installs the standalone default) rather than serve that initial value.
        """
        from kiro_crew import safety_override as so

        ctx_mod.reset_context()
        so.reset_yolo_policy_state()
        assert ctx_mod.installed_context() is None

        assert so.yolo_policy_permits() is True, (
            "an ungoverned host has no ceiling denying yolo; leaving the initial "
            "fail-closed value in place would disable the feature outright"
        )
        assert ctx_mod.installed_context() is not None, "the read resolved a ceiling"

    def test_the_silent_lazy_install_still_yields_the_real_ceilings_verdict(self, monkeypatch):
        """The lazy default is the one install that does NOT notify, and that is safe.

        It is reached from inside a governance read (the profile store resolves the
        active context for its freshness key), so a hook that reads governance would
        re-enter a store that is mid-load and get its fail-closed "not loaded" answer
        -- a verdict about nothing, cached as the truth.

        Skipping the notification is not a hole, and this is the case that proves it:
        the lazy default composes ``load_security_policy()``, so it CAN carry a deny,
        and the first read must still see it. It does, because the read's own bootstrap
        pushes after composing. Nothing can hold a grant earlier than that, since
        arming reads the same verdict.
        """
        from kiro_crew import safety_override as so
        from kiro_crew.platform import bootstrap as bs

        denying = parse_policy(
            {
                "version": 1,
                "boot": {"fail_closed": True},
                "approval_modes": {"mode": "deny", "deny": ["yolo"]},
            }
        )
        monkeypatch.setattr(bs, "load_security_policy", lambda: denying)
        ctx_mod.reset_context()
        so.reset_yolo_policy_state()
        so.reset_singleton()
        monkeypatch.setattr(so, "sel", lambda: MagicMock())

        assert so.yolo_policy_permits() is False
        assert so.safety_override().activate("dashboard").active is False

    def test_a_ceiling_installed_before_registration_is_still_resolved(self, monkeypatch):
        """The late-registration ordering.

        Registration deliberately does not replay an install that already happened --
        resolving governance from inside a module import is how cycles are born -- so
        the first read has to cope with a context that is already in place and has no
        pending install to announce.
        """
        from kiro_crew import safety_override as so

        _deny("yolo")
        # Exactly the state a late import leaves behind: the ceiling is installed,
        # but nothing has been pushed into the flag.
        so.reset_yolo_policy_state()

        assert so.yolo_policy_permits() is False

    def test_a_context_reset_forgets_the_verdict_rather_than_keeping_it(self):
        """``reset_context`` means there is no ceiling to derive an answer from.

        Keeping the last answer would let a deny outlive the ceiling that issued it,
        which in this suite reads as yolo being refused in a file that never
        configured a policy.
        """
        from kiro_crew import safety_override as so

        _deny("yolo")
        assert so.yolo_policy_permits() is False

        ctx_mod.reset_context()
        assert so._yolo_policy_resolved is False, "the pushed verdict is dropped"
        assert so.yolo_policy_permits() is True, "the re-resolve sees no denying ceiling"


class TestADenyLandingMidArmDoesNotLeaveAGrantBehind:
    """The gate is a check and the commit is the act, with I/O in between.

    Arming reads the verdict, then writes a fail-closed SEL event -- a synchronous
    filesystem write -- and only then records the grant. A denying ceiling installed
    in that gap has already run ``revoke_for_policy`` past an entry that does not
    exist yet, so committing anyway leaves a grant nothing will ever tear down. The
    dashboard caller then writes ``approval_policy="auto"`` onto its slots, and
    ``admission.parent_trusted`` reads that policy directly.

    Driven by installing the ceiling from inside the audit write, which is the real
    ordering rather than a simulation of it: that is exactly where the gap is.
    """

    def _deny_during_the_audit(self, monkeypatch, so):
        """Make the fail-closed audit write install a denying ceiling as it runs."""
        real = so.SafetyOverride._log_sel
        fired: list[str] = []

        def _log_sel(self, **kw):
            if kw.get("critical") and not fired:
                fired.append(kw.get("operation", ""))
                _deny("yolo")
            return real(self, **kw)

        monkeypatch.setattr(so.SafetyOverride, "_log_sel", _log_sel)
        return fired

    def test_a_session_wide_arm_is_refused(self, monkeypatch):
        from kiro_crew import safety_override as so

        _install_no_policy()
        so.reset_singleton()
        monkeypatch.setattr(so, "sel", lambda: MagicMock())
        override = so.safety_override()
        fired = self._deny_during_the_audit(monkeypatch, so)

        result = override.activate("dashboard")

        assert fired, "the ceiling must have been installed during the audit"
        assert result.active is False, (
            "an arm whose policy was withdrawn mid-flight must be reported as refused, "
            "or the dashboard writes approval_policy='auto' onto its slots"
        )
        assert override.has_grant() is False, (
            "the revocation already ran past this grant, so committing it would leave "
            "one that nothing tears down"
        )

    def test_a_grant_revoked_after_the_commit_is_not_reported_as_active(self, monkeypatch):
        """The caller acts on the RESULT, not on the grant, so the result must be true.

        A deny landing after the commit revokes the grant through
        ``revoke_for_policy``. If ``activate`` still answered ``active=True``, the
        dashboard would write ``approval_policy="auto"`` onto its slots on the
        strength of that answer, and ``admission.parent_trusted`` reads that policy
        directly -- so the inherited trust the revocation had just cleared would come
        straight back, with nothing left to clear it again.

        Driven through ``_sync_breadcrumb``, which runs in the real gap between the
        commit and the return.
        """
        from kiro_crew import safety_override as so

        _install_no_policy()
        so.reset_singleton()
        monkeypatch.setattr(so, "sel", lambda: MagicMock())
        override = so.safety_override()

        real = so.SafetyOverride._sync_breadcrumb
        fired: list[int] = []

        def _sync(self):
            if not fired:
                fired.append(1)
                _deny("yolo")
            return real(self)

        monkeypatch.setattr(so.SafetyOverride, "_sync_breadcrumb", _sync)

        result = override.activate("dashboard")

        assert fired, "the deny must have landed inside the commit-to-return gap"
        assert override.has_grant() is False, "the revocation tore the grant down"
        assert result.active is False, (
            "reporting a revoked grant as active is what puts approval_policy='auto' "
            "back onto the slots after the deny cleared it"
        )

    def test_the_survival_check_reads_both_kinds_of_grant(self, monkeypatch):
        """Both arming paths report from live state through one shared check.

        ``activate_scoped`` has no I/O between its commit and its return, so the
        scoped branch cannot be driven by a mid-flight deny the way the session-wide
        one can. Its logic is asserted directly instead -- the branch, not the race.
        """
        from kiro_crew import safety_override as so

        _install_no_policy()
        so.reset_singleton()
        monkeypatch.setattr(so, "sel", lambda: MagicMock())
        override = so.safety_override()
        override.activate("dashboard")
        override.activate_scoped("run:abc", "dashboard")

        assert override._committed_grant_survives() is True
        assert override._committed_grant_survives("run:abc") is True
        assert override._committed_grant_survives("run:missing") is False

        assert override.revoke_for_policy() is True

        assert override._committed_grant_survives() is False
        assert override._committed_grant_survives("run:abc") is False, (
            "a scoped grant revoked by policy must not be reported back to its caller "
            "-- taskrunner persists run.auto_approve straight from that result"
        )

    def test_the_verdict_is_written_before_the_revocation_runs(self, monkeypatch):
        """The ordering the two commit re-checks rely on, pinned directly.

        Under real concurrency the two checks above are only sound because the push
        writes the denied verdict BEFORE taking ``_lock`` to revoke. If it revoked
        first, a commit already holding the lock would read a stale permit, the revoke
        would then run past an entry that did not exist yet, and the grant would
        survive -- the same orphan, reached by the interleaving instead of by the gap.
        No single-threaded case can observe that, so the invariant is asserted here.
        """
        from kiro_crew import safety_override as so

        _install_no_policy()
        so.reset_singleton()
        monkeypatch.setattr(so, "sel", lambda: MagicMock())
        override = so.safety_override()
        override.activate("dashboard")

        seen: list[bool] = []
        real = so.SafetyOverride.revoke_for_policy

        def _spy(self):
            seen.append(so._yolo_policy_permitted)
            return real(self)

        monkeypatch.setattr(so.SafetyOverride, "revoke_for_policy", _spy)

        _deny("yolo")

        assert seen == [False], (
            "the denied verdict must already be visible when the revocation runs, or a "
            "concurrent commit holding the lock reads a stale permit and its grant "
            "outlives the deny"
        )

    def test_a_scoped_arm_is_refused(self, monkeypatch):
        from kiro_crew import safety_override as so

        _install_no_policy()
        so.reset_singleton()
        monkeypatch.setattr(so, "sel", lambda: MagicMock())
        override = so.safety_override()
        fired = self._deny_during_the_audit(monkeypatch, so)

        result = override.activate_scoped("run:abc", "dashboard")

        assert fired
        assert result.active is False
        assert override.is_scope_active("run:abc") is False
        with override._lock:
            assert "run:abc" not in override._scoped, "no orphaned scoped grant"


class TestTheVerdictFailsClosedWhileANewCeilingIsBeingResolved:
    """Publishing the ceiling before re-resolving is a bypass, so it does not.

    ``_install`` makes the new ceiling visible and only then runs the hook that
    re-resolves the verdict, and that resolve is an ``iterdir`` plus a per-file
    ``stat``. Ordered that way, the denying ceiling is in force for the whole width of
    a governance read while ``is_active()`` still hands out the permit resolved under
    the ceiling that was just retired -- and a tool call landing there is
    auto-approved against a policy that denies it, with nothing to undo afterwards
    because the tool has already run.

    So invalidation is the FIRST phase of an install, not the second: the verdict is
    masked before the assignment and the true answer is written after it.
    """

    def test_a_reader_inside_the_resolve_window_is_denied(self, monkeypatch):
        from kiro_crew import safety_override as so

        _install_no_policy()
        so.reset_singleton()
        monkeypatch.setattr(so, "sel", lambda: MagicMock())
        override = so.safety_override()
        override.activate("dashboard")
        assert override.is_active() is True

        # Sampled from INSIDE the post-publication resolve, which is exactly the
        # window: the new ceiling is already visible and the verdict has not been
        # rewritten yet.
        real = so.approval_mode_permitted
        seen: list[bool] = []

        def _sampling(mode: str) -> bool:
            seen.append(override.is_active())
            return real(mode)

        monkeypatch.setattr(so, "approval_mode_permitted", _sampling)

        _deny("yolo")

        assert seen, "the resolve must have run"
        assert seen == [False], (
            "a reader inside the resolve window must fail closed -- the new ceiling "
            "is already in force there, so serving the previous permit auto-approves "
            "against a policy that denies"
        )

    def test_the_mask_runs_before_the_new_ceiling_is_published(self, monkeypatch):
        """The ordering itself, not just its effect.

        Masking during the resolve is not enough: the window that has to be closed
        opens the instant the new ceiling becomes visible. So the invalidation runs
        while ``installed_context()`` still answers the OUTGOING ceiling -- which is
        what leaves no moment where the incoming one is in force and the previous
        permit is still being served.
        """
        from kiro_crew import safety_override as so

        _install_no_policy()
        before = ctx_mod.installed_context()
        seen: list[object] = []

        def _probe() -> None:
            seen.append(ctx_mod.installed_context())

        ctx_mod.register_ceiling_invalidate_hook(_probe)
        try:
            _deny("yolo")
        finally:
            ctx_mod._CEILING_INVALIDATE_HOOKS.remove(_probe)

        assert seen, "the invalidate hook must run on a declared install"
        assert seen[0] is before, (
            "invalidation must happen BEFORE the new ceiling is published, or the "
            "denying ceiling is live while the stale permit is still served"
        )
        assert so.yolo_policy_permits() is False

    def test_a_permitting_install_restores_the_verdict_it_masked(self, monkeypatch):
        """The mask is transient, and must not outlive the resolve.

        Masking on EVERY install is what makes the window safe, so a permitting
        install masks too. If the install hook then failed to write the true answer
        back, auto-approve would be off for the life of the process.
        """
        from kiro_crew import safety_override as so

        _install_no_policy()
        so.reset_singleton()
        monkeypatch.setattr(so, "sel", lambda: MagicMock())
        override = so.safety_override()
        override.activate("dashboard")

        _install_unrelated_ceiling()

        assert so.yolo_policy_permits() is True
        assert override.is_active() is True, "the grant survives a permitting install"

    def test_the_mask_does_not_revoke(self, monkeypatch):
        """Masking is not a teardown: a permitting install must keep the grant.

        Revoking on the mask would destroy a live grant on every unrelated ceiling
        refresh -- central distribution re-installs constantly.
        """
        from kiro_crew import safety_override as so

        _install_no_policy()
        so.reset_singleton()
        monkeypatch.setattr(so, "sel", lambda: MagicMock())
        override = so.safety_override()
        override.activate("dashboard")
        seen: list[str] = []
        override.on_expired = seen.append

        real = so.approval_mode_permitted
        during: list[bool] = []

        def _sampling(mode: str) -> bool:
            during.append(override.has_grant())
            return real(mode)

        monkeypatch.setattr(so, "approval_mode_permitted", _sampling)
        _install_unrelated_ceiling()

        assert during == [True], "the grant must still EXIST while the mask is on"
        assert seen == [], "a mask fires no expiry teardown"


class TestTheRevocationNoticeReachesTheLoopItWasInstalledOn:
    """``apply_ceiling`` can install a ceiling from a worker thread.

    ``_on_expired`` is loop-affine: it schedules the Slack expiry DM and the
    unattended-run notice with ``loop.create_task`` behind a ``get_running_loop()``
    probe, so called off-loop it silently posts nothing -- and silence about a
    security grant being revoked is what that notice exists to prevent. So the
    callback is scheduled onto the loop it was installed from, while the grant
    teardown itself (thread-safe, and the part that must be true immediately) runs
    inline on the installing thread.
    """

    @pytest.mark.asyncio
    async def test_the_inherited_trust_teardown_is_not_deferred(self, monkeypatch):
        """Only the NOTIFICATION may wait for the loop; the state may not.

        ``admission.parent_trusted`` reads a session's ``approval_policy == "auto"``
        directly, consulting no flag here, so deferring that teardown leaves a
        loop-turn window in which a spawn is auto-approved against the denying ceiling
        -- and the subagent it launched is not un-spawned by the cleanup that arrives
        afterwards. So the inherited half runs inline, on the installing thread, before
        anything is scheduled.
        """
        from kiro_crew import safety_override as so

        _install_no_policy()
        so.reset_singleton()
        monkeypatch.setattr(so, "sel", lambda: MagicMock())
        override = so.safety_override()
        override.activate("dashboard")

        loop = asyncio.get_running_loop()
        notified = asyncio.Event()
        order: list[str] = []
        threads: dict[str, str] = {}

        def _clear_state(source: str) -> None:
            order.append(f"state:{source}")
            threads["state"] = threading.current_thread().name

        def _notify(source: str) -> None:
            order.append(f"notify:{source}")
            threads["notify"] = threading.current_thread().name
            loop.call_soon_threadsafe(notified.set)

        override.on_policy_revoked = _clear_state
        override.on_expired = _notify

        loop_thread = threading.current_thread().name
        install_thread: list[str] = []

        def _install() -> None:
            install_thread.append(threading.current_thread().name)
            _deny("yolo")

        await asyncio.to_thread(_install)
        await asyncio.wait_for(notified.wait(), timeout=5)

        # Which THREAD each half ran on is the timing-independent form of "one is
        # inline and the other is deferred": sampling the ordering mid-flight would
        # race the loop, which is free to run the scheduled half the moment the
        # installing thread yields.
        assert threads["state"] == install_thread[0], (
            "the inherited-trust teardown must run INLINE on the thread that installed "
            "the ceiling -- deferring it is the window a spawn is auto-approved in"
        )
        assert threads["notify"] == loop_thread, (
            "the notification is loop-affine and must be scheduled, not run off-loop "
            "where it silently posts nothing"
        )
        # Deterministic regardless of scheduling: the inline call precedes the
        # ``call_soon_threadsafe`` that queues the other.
        assert order == ["state:policy", "notify:policy"]

    def test_the_teardown_iterates_a_snapshot_of_the_slots(self):
        """A concurrent slot change must not abort the revocation partway.

        The teardown runs on whichever thread installed the denying ceiling, while the
        loop keeps creating and removing slots. Iterating the live dict raises
        "dictionary changed size during iteration", which aborts the loop and leaves
        every slot it had not reached yet at ``approval_policy="auto"`` -- a partial
        revocation, with nothing scheduled to come back for the rest.

        Asserted against the source for the same reason as the guard below: the handler
        is a closure inside ``start_dashboard``.
        """
        import pathlib as _pl

        repo_root = _pl.Path(__file__).resolve().parents[1]
        src = (repo_root / "src" / "kiro_crew" / "dashboard" / "server.py").read_text(
            encoding="utf-8"
        )
        body = src.split("def _clear_override_derived_trust(source: str) -> None:", 1)
        assert len(body) == 2, "the inherited-trust teardown was renamed; update this guard"
        teardown = body[1].split("\n    def _on_override_expired", 1)[0]
        assert "for slot in list(state._slots.values()):" in teardown, (
            "the teardown must iterate a SNAPSHOT -- a concurrent slot change would "
            "otherwise raise mid-loop and leave the remaining slots auto-approved"
        )
        assert "for slot in state._slots.values():" not in teardown

    def test_the_notifier_still_runs_the_inherited_teardown(self):
        """Splitting the handler must not drop the teardown from the TTL path.

        A TTL lapse fires ``on_expired`` and nothing else -- there is no separate
        synchronous call on that path -- so the notifier has to keep running the
        inherited-trust teardown itself. Asserted against the source because the
        handler is a closure inside ``start_dashboard`` and cannot be imported; the
        failure it guards is silent, since a lapsed grant would simply leave every slot
        at ``approval_policy="auto"`` for good.
        """
        import pathlib as _pl

        repo_root = _pl.Path(__file__).resolve().parents[1]
        src = (repo_root / "src" / "kiro_crew" / "dashboard" / "server.py").read_text(
            encoding="utf-8"
        )
        body = src.split("def _on_override_expired(source: str) -> None:", 1)
        assert len(body) == 2, "the expiry notifier was renamed; update this guard"
        notifier = body[1].split("\n    safety_override().on_expired", 1)[0]
        assert "_clear_override_derived_trust(source)" in notifier, (
            "the expiry notifier must still clear inherited trust -- a TTL lapse has "
            "no other path to it"
        )

    def test_a_raising_sync_teardown_does_not_stop_the_notification(self, monkeypatch):
        """The two halves are independent; neither may swallow the other."""
        from kiro_crew import safety_override as so

        _install_no_policy()
        so.reset_singleton()
        monkeypatch.setattr(so, "sel", lambda: MagicMock())
        override = so.safety_override()
        override.activate("dashboard")

        def _boom(_source: str) -> None:
            raise RuntimeError("session store unavailable")

        override.on_policy_revoked = _boom
        seen: list[str] = []
        override.on_expired = seen.append

        _deny("yolo")

        assert override.has_grant() is False
        assert seen == ["policy"]

    @pytest.mark.asyncio
    async def test_an_off_thread_install_schedules_the_callback_onto_the_loop(self, monkeypatch):
        from kiro_crew import safety_override as so

        _install_no_policy()
        so.reset_singleton()
        monkeypatch.setattr(so, "sel", lambda: MagicMock())
        override = so.safety_override()
        override.activate("dashboard")

        loop = asyncio.get_running_loop()
        fired = asyncio.Event()
        threads: list[str] = []

        def _handler(source: str) -> None:
            threads.append(threading.current_thread().name)
            loop.call_soon_threadsafe(fired.set)

        # Assigned from inside a running loop, which is how the dashboard installs it.
        override.on_expired = _handler
        assert override._on_expired_loop is loop

        main_thread = threading.current_thread().name
        await asyncio.to_thread(_deny, "yolo")

        # The teardown is NOT deferred: it is done by the time the install returns.
        assert override.has_grant() is False

        await asyncio.wait_for(fired.wait(), timeout=5)
        assert threads == [main_thread], (
            "the handler must run on the loop it was installed from, not on the "
            "worker thread that installed the ceiling"
        )

    def test_no_recorded_loop_means_the_callback_runs_inline(self, monkeypatch):
        """A sync CLI or test has no loop to schedule onto; dropping it is not an option."""
        from kiro_crew import safety_override as so

        _install_no_policy()
        so.reset_singleton()
        monkeypatch.setattr(so, "sel", lambda: MagicMock())
        override = so.safety_override()
        override.activate("dashboard")
        seen: list[str] = []
        override.on_expired = seen.append
        assert override._on_expired_loop is None

        _deny("yolo")

        assert seen == ["policy"]


class TestInheritedSlotTrustIsClearedOnADenyInstall:
    """Clearing the grant flag is NOT the whole revocation.

    A dashboard grant also writes ``approval_policy="auto"`` onto the slots and into
    the shared channel-trust mapping, and ``subagent_manager.admission`` reads THAT
    policy -- ``sessions.get_approval_policy(parent_session_key) == "auto"`` -- rather
    than any flag in ``safety_override``. So a revocation that stopped at the flag
    left ``spawn_run`` auto-approved, and a subagent already launched under it is not
    un-spawned when the deny lands. ``_on_expired`` is the handler that resets those
    policies, which is why the deny install fires it.
    """

    def test_the_admission_predicate_stops_reading_auto_after_a_deny_install(self, monkeypatch):
        from kiro_crew import safety_override as so

        policies: dict[str, str] = {}

        class _Sessions:
            def get_approval_policy(self, key: str) -> str:
                return policies.get(key, "")

            def set_approval_policy(self, key: str, value: str) -> None:
                policies[key] = value

        sessions = _Sessions()
        parent_key = "dashboard:s1"

        def _on_expired(_source: str) -> None:
            # The shape of the dashboard's own handler: every slot's inherited
            # policy is reset. Reproduced here rather than imported because the real
            # one is defined inside ``start_dashboard``.
            for key in list(policies):
                sessions.set_approval_policy(key, "")

        _install_no_policy()
        so.reset_singleton()
        monkeypatch.setattr(so, "sel", lambda: MagicMock())
        override = so.safety_override()
        override.on_expired = _on_expired
        assert override.activate("dashboard").active is True
        # What the dashboard grant does to the slot it was armed from.
        sessions.set_approval_policy(parent_key, "auto")
        assert sessions.get_approval_policy(parent_key) == "auto"

        _deny("yolo")

        assert sessions.get_approval_policy(parent_key) != "auto", (
            "spawn admission reads the slot's approval policy directly, so a "
            "revocation that leaves it at 'auto' keeps auto-approving subagents "
            "against a policy that denies yolo"
        )
        assert override.is_active() is False


class TestTheApprovalCardDoesNotRestoreTrustAfterADeny:
    """The grant's INHERITED half is written by the caller, outside the revoke's lock.

    ``POST /api/chat/slots/<slot>/approve`` with ``action=yolo`` arms the grant and
    then writes ``approval_policy="auto"`` onto every slot -- and
    ``admission.parent_trusted`` reads that policy directly rather than any flag in
    ``safety_override``. A denying ceiling installed after the arm returns revokes the
    grant and runs its ``_on_expired`` cleanup, so a write landing after that cleanup
    puts the inherited trust straight back with nothing left to clear it: a subagent
    spawned under it is auto-approved, and no later event un-spawns it.
    """

    @pytest.mark.asyncio
    async def test_a_deny_during_the_policy_write_leaves_no_inherited_auto(
        self, tmp_path, monkeypatch
    ):
        from kiro_crew import safety_override as so

        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        _install_no_policy()
        so.reset_singleton()
        monkeypatch.setattr(so, "sel", lambda: MagicMock())

        policies: dict[str, str] = {}
        state = _make_state(tmp_path)
        state.sessions.get_approval_policy = lambda key: policies.get(key, "")
        fired: list[int] = []

        def _set_policy(key: str, value: str) -> None:
            policies[key] = value
            # The deny lands mid-write, which is the reachable ordering: the arm has
            # already returned, so its own survival check cannot see this.
            if value == "auto" and not fired:
                fired.append(1)
                _deny("yolo")

        state.sessions.set_approval_policy = _set_policy
        slot = state.get_or_create_slot("s1")
        # A slot carrying STANDING trust. A Trust press is a separate, longer-lived
        # decision that no yolo deny expires, so the reconcile must leave it alone --
        # the same rule ``_on_override_expired`` applies.
        trusted = state.get_or_create_slot("s2")
        trusted._trust = True
        fut: asyncio.Future[str] = asyncio.get_running_loop().create_future()
        slot._approval_futures["req-1"] = fut

        async with TestClient(TestServer(_make_app(state))) as client:
            resp = await client.post(
                "/api/chat/slots/s1/approve",
                json={"action": "yolo", "request_id": "req-1"},
            )
            body = await resp.json()

        assert fired, "the deny must have landed during the policy write"
        assert body.get("ok") is True, "the operator's click still approves that tool"
        assert so.safety_override().has_grant() is False, "the deny revoked the grant"
        from kiro_crew.dashboard.chat_utils import effective_session_key

        plain_key = effective_session_key(slot)
        trusted_key = effective_session_key(trusted)
        assert policies.get(plain_key) != "auto", (
            "a slot left at approval_policy='auto' after the deny keeps auto-approving "
            "subagent spawns through admission.parent_trusted"
        )
        assert policies.get(trusted_key) == "auto", (
            "standing trust is not expired by a yolo deny -- clearing it here would "
            "revoke a grant nobody withdrew"
        )


class TestNothingResolvesGovernanceOffTheInstallPath:
    """Every consumer is a bare read, and that is a hard property, not a preference.

    ``is_active`` is the predicate every transport hands to ``TurnDriver`` (so it runs
    per TOOL CALL), ``status_snapshot`` rides the 5s WebSocket push, and arming is
    reached from the event loop by synchronous callers. Resolving the scope walks the
    profiles dir, so any of those doing it would put filesystem work where it cannot go.
    """

    @pytest.fixture()
    def _tripwire(self, monkeypatch):
        from kiro_crew import safety_override as so

        # Singleton reset FIRST: it forgets the pushed verdict, so installing the
        # ceiling afterwards is what leaves a resolved flag behind for the tripwire
        # to prove nothing re-reads.
        so.reset_singleton()
        _install_no_policy()  # resolves the verdict ONCE, here
        monkeypatch.setattr(so, "sel", lambda: MagicMock())
        calls: list[str] = []

        def _boom(mode: str) -> bool:
            calls.append(mode)
            return True

        monkeypatch.setattr(so, "approval_mode_permitted", _boom)
        return calls

    def test_the_hot_predicate_does_not_resolve(self, _tripwire):
        from kiro_crew import safety_override as so

        so.safety_override().activate("dashboard")
        for _ in range(20):
            so.safety_override().is_active()

        assert _tripwire == [], "is_active runs per tool call; it must never resolve"

    def test_arming_does_not_resolve(self, _tripwire):
        """Arming reads the flag too, so its async callers need no offload FOR THIS.

        They keep one anyway: the fail-closed SEL audit is still a filesystem write.
        """
        from kiro_crew import safety_override as so

        assert so.safety_override().activate("dashboard").active is True
        assert so.safety_override().activate_scoped("run:abc", "dashboard").active is True

        assert _tripwire == [], "arming must not walk the profiles dir"

    def test_the_scoped_consult_points_do_not_resolve(self, _tripwire):
        from kiro_crew import safety_override as so

        so.safety_override().activate_scoped("run:abc", "dashboard")
        for _ in range(20):
            so.safety_override().is_scope_active("run:abc")
            so.safety_override().renew_scoped("run:abc", "dashboard")

        assert _tripwire == []

    def test_the_status_reader_does_not_resolve(self, _tripwire):
        from kiro_crew import safety_override as so

        for _ in range(20):
            assert so.cached_disabled_approval_modes() == []

        assert _tripwire == []

    def test_the_status_field_cannot_contradict_enforcement(self):
        """The anti-divergence property, stated directly.

        Two caches of one question is what let the picker disagree with the gate.
        Whatever ``yolo_policy_permits`` says, the reported list must say.
        """
        from kiro_crew import safety_override as so

        _install_no_policy()
        assert so.yolo_policy_permits() is True
        assert so.cached_disabled_approval_modes() == []

        _deny("yolo")
        assert so.yolo_policy_permits() is False
        assert so.cached_disabled_approval_modes() == ["yolo"]

    @pytest.mark.asyncio
    async def test_status_snapshot_reports_the_pushed_verdict(self, tmp_path, monkeypatch):
        from kiro_crew import safety_override as so

        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        _deny("yolo")
        state = _make_state(tmp_path)

        snap = state.status_snapshot()

        # ``["trust", "yolo"]`` was the shape an earlier revision asserted here; the
        # backend can never emit it, because those modes are non-deniable.
        assert snap["disabled_approval_modes"] == ["yolo"]
        assert so.yolo_policy_permits() is False


class TestArmingIsStillOffloadedByItsAsyncCallers:
    """The offload survives this refactor, because the SEL audit is still I/O.

    Arming no longer resolves governance, but it still writes a fail-closed security
    event before the grant exists -- a synchronous filesystem write -- so an async
    caller that ran it inline would put that on the gateway's loop.
    """

    @pytest.mark.asyncio
    async def test_taskrunner_grant_run_trust_is_a_coroutine(self):
        import inspect

        from kiro_crew.taskrunner import TaskRunner

        assert inspect.iscoroutinefunction(TaskRunner._grant_run_trust), (
            "_grant_run_trust is reached from two async methods; a sync body puts "
            "the SEL write on the event loop"
        )

    def test_every_grant_run_trust_call_site_is_awaited(self):
        """A coroutine left unawaited silently does nothing at all.

        That failure mode is invisible: the trust grant simply never happens, and
        the run proceeds with `auto_approve` set on the object but no authoritative
        scoped grant behind it.
        """
        import pathlib as _pl

        # Anchored to THIS FILE, not the process CWD. A relative path here reads
        # whatever happens to sit under the worker's working directory -- so the
        # case either fails for the wrong reason or, worse, silently inspects a
        # different file and passes without checking anything.
        repo_root = _pl.Path(__file__).resolve().parents[1]
        src = (repo_root / "src" / "kiro_crew" / "taskrunner.py").read_text(encoding="utf-8")
        bare = [
            line.strip()
            for line in src.splitlines()
            if "self._grant_run_trust(" in line and "await " not in line and "def " not in line
        ]
        assert not bare, f"unawaited _grant_run_trust call(s): {bare}"


class TestEveryGrantBranchHonoursAMidSessionDeny:
    """A mid-session deny must reach EVERY branch that honours a grant.

    Revoking only the session-wide grant left the other kind, and the scoped branch
    is the reachable one: ``task_executor`` consults ``is_scope_active`` before every
    approval and slides the grant with ``renew_scoped``, so a taskrunner run armed
    while permitted kept auto-approving for up to 24h after the deny. The branch table
    this pins:

    | consult point        | grant kind           | revoked |
    |----------------------|----------------------|---------|
    | ``is_active``        | session-wide, TTL    | yes     |
    | ``is_active``        | declared, no TTL     | yes     |
    | ``is_scope_active``  | scoped, per run      | yes     |
    | ``renew_scoped``     | scoped, expiry slide | yes     |
    """

    def _armed_scope(self, monkeypatch):
        from kiro_crew import safety_override as so

        _install_no_policy()
        so.reset_singleton()
        monkeypatch.setattr(so, "sel", lambda: MagicMock())
        assert so.safety_override().activate_scoped("run:abc", "dashboard").active is True
        assert so.safety_override().is_scope_active("run:abc") is True
        return so

    def test_a_live_scoped_grant_stops_being_honoured(self, monkeypatch):
        so = self._armed_scope(monkeypatch)

        _deny("yolo")

        assert so.safety_override().is_scope_active("run:abc") is False

    def test_a_denied_scoped_grant_is_not_slid_forward(self, monkeypatch):
        """Renewal is what keeps an active run's grant alive to the ceiling."""
        so = self._armed_scope(monkeypatch)

        _deny("yolo")

        result = so.safety_override().renew_scoped("run:abc", "dashboard")
        assert result.renewed is False

    def test_a_denied_scoped_grant_is_revoked_not_merely_masked(self, monkeypatch):
        """Denial tears the grant down, so relaxing the policy cannot resurrect it."""
        so = self._armed_scope(monkeypatch)

        _deny("yolo")
        assert so.safety_override().is_scope_active("run:abc") is False

        _install_no_policy()
        assert (
            so.safety_override().is_scope_active("run:abc") is False
        ), "a policy-revoked scoped grant must stay revoked once the policy relaxes"

    def test_a_declared_grant_is_revoked_too(self, monkeypatch):
        """A permanent grant has no deadline, so policy is its ONLY off-switch."""
        from kiro_crew import safety_override as so

        _install_no_policy()
        so.reset_singleton()
        monkeypatch.setattr(so, "sel", lambda: MagicMock())
        so.safety_override().activate_declared()
        assert so.safety_override().is_active() is True

        _deny("yolo")

        assert so.safety_override().is_active() is False
        assert so.safety_override().has_grant() is False

    def test_a_session_wide_grant_stays_revoked_when_the_policy_relaxes(self, monkeypatch):
        """The revoke-vs-mask choice, from the caller that made it necessary.

        Slack's off-path is ``if is_yolo_mode(): disable_yolo()``. While a mask-only
        denial reported inactive, that branch cleared NOTHING, and a later policy
        relaxation resurrected the grant the operator had explicitly revoked. A fresh
        arm after the policy relaxes is the honest outcome.
        """
        from kiro_crew import safety_override as so

        _install_no_policy()
        so.reset_singleton()
        monkeypatch.setattr(so, "sel", lambda: MagicMock())
        so.safety_override().activate("dashboard")

        _deny("yolo")
        assert so.safety_override().is_active() is False

        _install_no_policy()
        assert (
            so.safety_override().is_active() is False
        ), "a policy-revoked grant must stay revoked once the policy relaxes"


class TestAnExplicitOffIsNotPolicyFiltered:
    """``is_active`` answers "may a tool be auto-approved", which policy can veto.

    An explicit off asks a DIFFERENT question -- "is there something to tear down" --
    and reading the policy-filtered answer for it inverted the control. A deny now
    destroys the grant at install time, so the two agree; ``disable_yolo`` still reads
    ``has_grant`` because the policy mask is the fail-closed floor for a teardown that
    did not complete, and that is exactly the state where an off has work to do.
    """

    def test_has_grant_ignores_policy_entirely(self, monkeypatch):
        from kiro_crew import safety_override as so

        _install_no_policy()
        so.reset_singleton()
        monkeypatch.setattr(so, "sel", lambda: MagicMock())
        override = so.safety_override()
        override.activate("dashboard")

        _deny("yolo")
        assert override.is_active() is False
        assert (
            override.has_grant() is False
        ), "a policy deny REVOKES, so after it there is genuinely nothing to clear"

    @pytest.mark.asyncio
    async def test_an_off_under_a_masked_grant_actually_revokes(self, monkeypatch):
        from kiro_crew import safety_override as so
        from kiro_crew.slack import handler as h

        _install_no_policy()
        so.reset_singleton()
        monkeypatch.setattr(so, "sel", lambda: MagicMock())
        monkeypatch.setattr(h, "clear_trusted_sessions", lambda: None)
        override = so.safety_override()
        override.activate("dashboard")

        # A masked-but-standing grant: the verdict denies while the grant survives,
        # which is what an install-time teardown that did not complete leaves behind.
        monkeypatch.setattr(so, "_yolo_policy_permitted", False)
        monkeypatch.setattr(so, "_yolo_policy_resolved", True)
        assert override.is_active() is False
        assert override.has_grant() is True, "the grant is there to be cleared"

        h.disable_yolo()

        with override._lock:
            assert (
                override._active is False
            ), "an explicit off must tear the grant down even while policy masks it"


class TestARefusedScopedArmDoesNotPersistAutoApprove:
    """``_grant_run_trust`` exists so the flag and the grant cannot diverge.

    It set ``run.auto_approve`` BEFORE arming, so a policy-refused arm still
    persisted and reported ``auto_approve: True`` with no authoritative grant behind
    it -- the very divergence the function's own docstring promises to prevent.
    """

    @pytest.mark.asyncio
    async def test_a_denied_policy_leaves_auto_approve_false(self, monkeypatch):
        from kiro_crew import safety_override as so
        from kiro_crew.taskrunner import TaskRunner

        _deny("yolo")
        so.reset_singleton()
        monkeypatch.setattr(so, "sel", lambda: MagicMock())

        run = MagicMock()
        run.task_id = "t-1"
        run.auto_approve = False

        await TaskRunner._grant_run_trust(MagicMock(), run, True)

        assert (
            run.auto_approve is False
        ), "a refused arm must not persist or report auto-approve it does not have"

    @pytest.mark.asyncio
    async def test_a_permitting_policy_still_enables_it(self, monkeypatch):
        """The fix must not make the flag unreachable."""
        from kiro_crew import safety_override as so
        from kiro_crew.taskrunner import TaskRunner

        _install_no_policy()
        so.reset_singleton()
        monkeypatch.setattr(so, "sel", lambda: MagicMock())

        run = MagicMock()
        run.task_id = "t-2"
        run.auto_approve = False

        await TaskRunner._grant_run_trust(MagicMock(), run, True)

        assert run.auto_approve is True


class TestALapsedGrantIsNotTornDownTwice:
    """A grant that reached its own deadline owes no second teardown.

    The natural-expiry branch clears ``_active`` but deliberately LEAVES
    ``_expires_at`` set -- ``deactivate`` reads that nonzero deadline to tell "lapsed"
    from "never armed", so it can still SEL-record an explicit off after a lapse. A
    policy revocation that treated that stale deadline as a grant would fire a second
    expiry teardown for a grant that had already expired: a duplicate owner DM and a
    redundant broadcast.
    """

    def test_a_deny_after_a_natural_lapse_fires_no_second_teardown(self, monkeypatch):
        import time as _time

        from kiro_crew import safety_override as so

        _install_no_policy()
        so.reset_singleton()
        monkeypatch.setattr(so, "sel", lambda: MagicMock())
        override = so.safety_override()
        override.activate("dashboard")

        seen: list[str] = []
        override.on_expired = seen.append

        # Force the deadline into the past, then let is_active reconcile it. That is
        # the natural-expiry path: it clears _active and leaves _expires_at set.
        with override._lock:
            override._expires_at = _time.monotonic() - 1.0

        assert override.is_active() is False
        assert seen == ["dashboard"], "the natural lapse fires its own teardown, once"
        with override._lock:
            assert override._expires_at > 0.0, (
                "the lapsed deadline is deliberately retained -- that is what makes a "
                "naive grant test double-fire"
            )

        # Policy now denies. There is no grant left to revoke.
        _deny("yolo")
        assert override.is_active() is False
        assert seen == ["dashboard"], f"a lapsed grant must not be torn down twice: {seen}"
