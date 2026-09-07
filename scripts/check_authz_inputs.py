#!/usr/bin/env python3
"""check_authz_inputs.py -- an authorization call must state every identity it has.

## The failure class

``resolve_active_scope`` decides which governance profile bounds a tool call, and
it accepts three identity inputs -- surface (``session_key``), caller (``agent``)
and ``app`` -- across precedence paths that each consult a different subset. The
optional ones default falsy, so a call site that omits one does not fail, does not
warn, and does not look wrong; the omission is discoverable only by reading the
resolver. (A fourth, ``task``, was accepted and bound but passed by nothing, and
bound the same namespace the ``agent`` branch already reaches -- it was deleted
rather than left as an input every call site had to declare.)

Be precise about the consequence, because the loose version of this claim is
false. The lookups are truthiness-gated (``if app:``, ``if agent:``), so an
omitted identity is byte-identical AT RUNTIME to an explicit ``""``. The scope
actually widens only when a bind EXISTS for the identity that was not passed --
an operator has configured a profile the call then fails to consult. This gate
cannot know whether such a bind exists, and does not claim to: see "What this gate
does and does not buy" below.

The class is not hypothetical. One in-review change walked the same gap in one
file across four consecutive review rounds -- a skipped scope query, an
undisclosed target, an unconsulted caller profile, a caller riding the ``app``
positional -- each round finding the next input the resolver accepts and the call
site did not pass, and each round fixing exactly one. A second open change carries
it on the Slack surface, where an unresolved project agent silently runs as the
default. (Both are open pull requests as this lands, so their numbers are
deliberately not cited here as verifiable evidence -- ``git log`` cannot confirm
an unmerged PR, and a spec citing one gives a future reader nothing to check.)

Nothing in the tree checks the class. ``AUTOSDE.yaml`` is prose a model
interprets; docs-lint checks that named things EXIST, so a missing ARGUMENT is
invisible to it -- it was green through all four of those rounds, because the
function existed and was cited correctly the whole time.

## What this gate does and does not buy

It does NOT change resolution. Converting a flagged site to ``app=""`` resolves
exactly the same profile it resolved before, because the resolver branches on
truthiness. Anyone expecting a behavioural fix from a green run is misreading it.

What it buys is that the two cases stop being textually identical. "This surface
has no app" and "nobody thought about the app" render as different source, so the
second is reviewable and greppable instead of invisible. That is a real but
BOUNDED claim, and it is bounded further by the fact that the gate cannot tell a
considered ``""`` from one pasted to clear CI -- the reviewer supplies that, the
gate only supplies the obligation to say something.

An ``_AuthContext`` with no defaults is strictly stronger: it makes omission a
type error, cannot be evaded by an alias or a wrapper, and makes a wrong-slot
positional unrepresentable. This gate is the cheap version, and it should be read
as a stopgap that keeps the class from growing while that refactor is unfunded --
not as a reason it is unnecessary.

## What counts as a violation

A call to one of the authorization entry points in ``AUTHZ_FUNCS`` that does not
mention one of that function's identity parameters.

**An explicit falsy value is NOT a violation.** ``app=""`` is the point: it is the
call site DECLARING that this surface has no app, which is a claim a reviewer can
check and a later reader cannot mistake for an oversight. Silence is the defect,
not emptiness -- and per the section above, the declaration is worth exactly its
reviewability, not any runtime difference.

Two further shapes are flagged, because both defeat the keyword check:

* **A positional identity**, but only where one can travel positionally at all.
  ``_vet_spawn_governance``'s third positional is ``app``, so passing the caller
  agent positionally binds it there, type-checks, runs, and enforces nothing. The
  budget -- how many arguments may legitimately precede the identities -- is
  DERIVED from each signature, never declared here, and the check is skipped for a
  keyword-only entry point, where a surplus positional is already a ``TypeError``
  rather than merely unseen. Three of the four are in that position, which is why
  no live finding is a positional one; the fourth joins them the moment its
  signature gains a ``*``, and this rule then retires itself.
* **An unresolvable ``**splat``.** A dict literal is read and its keys count. An
  opaque ``**kwargs`` forward cannot be proven complete, so it is flagged rather
  than trusted -- a forwarded bag is exactly where an omission hides.

Matching is BY NAME, following ``check_subprocess_encoding.py``: resolving imports
would miss this repository's wrappers and the monkeypatched aliases its tests
install, and an aliased import is an evasion this gate does not try to outrun. The
cost is that an unrelated method sharing one of these names is flagged. Exactly
one such call exists in the tree -- ``messaging/renderer.py`` dispatching to its
OWN ``on_tool_call`` display method, unrelated to ``HookManager``'s -- and it
carries the marker. Eleven other renderers define that method without calling it,
so they cost nothing.

A file that does not parse is a hard ERROR, never "clean" -- under a shrink-only
ratchet a parse failure reading as zero violations invites a prune that deletes
the file's real entry.

## The opt-out marker

A site that genuinely cannot supply an identity, and for which an explicit ``""``
would be a lie rather than a declaration, opts out with an inline COMMENT on any
line of the call:

    governance_permits("apps", name)  # authz-inputs: boot-time, no session yet

The marker must sit on a line the CALL SPANS -- for a multi-line call, trailing it
on the opening line works. It deliberately does NOT reach a comment block above
the call: a marker with forward reach can exempt a call it was never written for,
and a comment that merely mentions the phrase becomes a silent suppression.

Only a real comment token counts -- the phrase inside a string literal does not
exempt a call. The marker is an audit trail, not an escape hatch: it asserts the
author decided, and it names the reason a reviewer can disagree with.

## The ratchet

The repository predates this gate, so existing violations are recorded in
``.github/authz-inputs-baseline.txt`` as ``<count> <path>`` lines, mirroring
``check_subprocess_encoding.py`` (same problem, same shape):

* a file NOT in the baseline must be clean;
* a file IN the baseline may not exceed its recorded count;
* a file that improves is rewritten down by ``--update-baseline``, never up.

So the class cannot grow, and the backlog is visible rather than silently
tolerated. ``--list`` prints the backlog without failing.

Scope is ``src/kiro_crew`` only. ``test/`` is NOT scanned, and carries several
hundred calls of the same shape -- a test asserting on a partial identity set is
arguably testing the wrong contract, but the failure text here is written about
production resolution and would be false for a locally constructed ``HookManager``
with no session. Widening the scope needs its own decision and its own message.
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import io
import sys
import tempfile
import tokenize
from collections.abc import Iterator
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASELINE = ROOT / ".github" / "authz-inputs-baseline.txt"

DEFAULT_TARGETS = ("src/kiro_crew",)

MARKER = "authz-inputs:"


class AuthzFunc:
    """One authorization entry point: where it is defined, and which inputs it takes.

    ``defined_in`` is not documentation. Every derived fact about the signature --
    which identities exist, how many arguments may legitimately travel
    positionally, whether an identity is positionally reachable at all -- is read
    from that file, never written here. The first cut of this gate hand-maintained
    a positional budget and an identity set, and shipped demanding a
    ``caller_agent`` parameter the branch did not have, so its one flagged call
    site had no satisfiable fix while the self-test passed. A number a human keeps
    in step with a signature is the same defect class the gate exists to catch, so
    there is no such number here.
    """

    __slots__ = ("defined_in", "identities")

    def __init__(self, defined_in: str, identities: frozenset[str]):
        self.defined_in = defined_in
        self.identities = identities


#: `on_tool_call` deliberately does NOT list `resolved_agent`. Omitting it is
#: fail-closed, not widening: `hooks.py` keys `owner_app = app or
#: _builtin_app_for_agent(resolved_agent)` and its own comment records that an
#: empty `resolved_agent` "yields no identity -- fail-closed to interactive
#: approval". A gate whose premise is that omission WIDENS must not demand a
#: parameter whose omission narrows.
AUTHZ_FUNCS: dict[str, AuthzFunc] = {
    "governance_permits": AuthzFunc(
        "src/kiro_crew/platform/governance_profiles.py",
        frozenset({"session_key", "agent", "app"}),
    ),
    "resolve_active_scope": AuthzFunc(
        "src/kiro_crew/platform/governance_profiles.py",
        frozenset({"agent", "app"}),
    ),
    "_vet_spawn_governance": AuthzFunc(
        "src/kiro_crew/subagent.py",
        frozenset({"app"}),
    ),
    "on_tool_call": AuthzFunc(
        "src/kiro_crew/hooks.py",
        frozenset({"session_key", "agent", "app"}),
    ),
}

#: `asyncio.to_thread(fn, *args, **kwargs)` passes the callee as arg 0, so the real
#: call is one slot to the right. Without this the positional budget is off by one
#: for every threaded authorization call, and `cli_chat.py` routes the spawn
#: ceiling through exactly that shape.
THREAD_WRAPPERS = frozenset({"to_thread"})

#: `loop.run_in_executor(executor, fn, *args)` accepts NO keyword arguments, so an
#: identity can only reach the callee positionally: there is no form that both runs
#: and names what it carries. Judging it on the positional budget would flag every
#: legal form and pass every broken one, so wrapping an authorization call in it is
#: ITSELF the finding, and the fix is `asyncio.to_thread` or a `functools.partial`
#: that passes the identities as keywords. Nothing in the tree does this today; the
#: rule exists so that stays true.
KWARGLESS_WRAPPERS = frozenset({"run_in_executor"})

HEADER = """\
authz-inputs gate: an authorization call must state every identity it has.

An omitted identity is invisible: it does not fail, does not warn, and reads the
same as a considered decision. Where the operator has bound a profile to that
identity, the call resolves a wider scope than configured. Pass the value, or pass
an explicit "" to DECLARE the surface has none.

Note what "" does and does not do: the resolver's lookups are truthiness-gated, so
"" resolves EXACTLY what omitting it resolved. This is a declaration, not a
behaviour fix -- it makes a considered decision and an oversight stop looking the
same. If the identity is genuinely available here, pass the real value.
"""


class Signature:
    """The parts of a real signature this gate reasons about.

    ``positional`` excludes a leading ``self``/``cls``, because every call site in
    the tree invokes the bound method and never passes it.
    """

    __slots__ = ("positional", "keyword_only", "var_keyword")

    def __init__(self, positional: list[str], keyword_only: set[str], var_keyword: str):
        self.positional = positional
        self.keyword_only = keyword_only
        self.var_keyword = var_keyword

    @property
    def params(self) -> set[str]:
        names = set(self.positional) | set(self.keyword_only)
        return names | ({self.var_keyword} if self.var_keyword else set())


def _positional_contract(sig: Signature, identities: frozenset[str]) -> tuple[int, frozenset[str]]:
    """(budget, identities reachable positionally) -- both DERIVED, never declared.

    The budget is how many arguments may legitimately travel positionally: the
    positional parameters that are not identities. An identity beyond that is one
    riding a slot it does not name.

    ``positional_identities`` is what decides whether the rule applies at all. For
    a keyword-only entry point it is empty, and then a surplus positional is
    already a ``TypeError`` -- unrepresentable rather than merely unseen -- so the
    check is skipped instead of guarding a shape the interpreter forbids. Three of
    the four entry points are in that position today, which is why 0 of 38 findings
    were positional; when ``_vet_spawn_governance`` gains its ``*``, the last one
    joins them and this rule retires itself with no edit here.
    """
    reachable = frozenset(n for n in sig.positional if n in identities)
    budget = sum(1 for n in sig.positional if n not in identities)
    return budget, reachable


def _read_signature(rel: str, name: str) -> tuple[Signature | None, str]:
    """(parameter names, "") for *name* in *rel*, or (None, <why not>).

    Three failures used to collapse into one message, and the collapse produced a
    WRONG diagnosis: an unparseable file reported "no definition found" for a
    definition sitting right there, sending the reader to grep for something the
    gate could see all along. Each cause now names itself.

    Definitions are read from module scope and from class bodies only, never by
    walking the whole tree: `on_tool_call` is defined on twelve renderers in this
    repository, and a nested or shadowed def resolving first would validate the
    table against the wrong signature -- passing a real drift or failing a correct
    table, both silently.
    """
    path = ROOT / rel
    try:
        source = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None, f"{rel} does not exist -- has the definition moved?"
    except OSError as exc:
        return None, f"{rel} could not be read ({exc.__class__.__name__})"
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return None, f"{rel} does not parse (line {exc.lineno}) -- fix the file first"

    scopes: list[tuple[list[ast.stmt], bool]] = [(list(tree.body), False)]
    scopes += [(list(n.body), True) for n in tree.body if isinstance(n, ast.ClassDef)]
    found: ast.FunctionDef | ast.AsyncFunctionDef | None = None
    in_class = False
    for body, is_class_body in scopes:
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
                # Keep the LAST match in scope order, because that is the one
                # Python binds. Returning the first let a stale copy above the real
                # definition -- or the permissive half of an `if sys.platform:`
                # branch pair -- validate the table against a signature nothing
                # runs, which is byte-for-byte the drift this check exists to stop.
                found = node
                in_class = is_class_body
    if found is None:
        return None, f"no module-level or class-body definition of {name} in {rel}"
    a = found.args
    positional = [p.arg for p in (*a.posonlyargs, *a.args)]
    if in_class and positional and positional[0] in ("self", "cls"):
        positional = positional[1:]
    return (
        Signature(
            positional=positional,
            keyword_only={p.arg for p in a.kwonlyargs},
            var_keyword=a.kwarg.arg if a.kwarg else "",
        ),
        "",
    )


def _signature_params(rel: str, name: str) -> tuple[set[str] | None, str]:
    """Parameter NAMES only, for the drift check.

    A ``**kwargs`` catch-all accepts every identity at runtime, so no identity the
    table names can be proven absent: the declared set is folded in rather than
    reported as drift. Reporting it would brick the gate on every path with one
    exit -- deleting identities from the table, i.e. performing the de-gating this
    check exists to prevent.
    """
    sig, why = _read_signature(rel, name)
    if sig is None:
        return None, why
    params = sig.params
    if sig.var_keyword and name in AUTHZ_FUNCS:
        params |= set(AUTHZ_FUNCS[name].identities)
    return params, ""


def _table_drift() -> list[str]:
    """Every way AUTHZ_FUNCS disagrees with the signatures it claims to read.

    Runs on EVERY invocation, not only under --test, because a table that has
    drifted does not produce a wrong answer in some corner -- it produces findings
    whose prescribed fix is a TypeError, and the only exit is a suppression marker
    on a call site that was already correct.
    """
    problems: list[str] = []
    for name, spec in sorted(AUTHZ_FUNCS.items()):
        params, why = _signature_params(spec.defined_in, name)
        if params is None:
            problems.append(f"{name}: {why}")
            continue
        unknown = sorted(spec.identities - params)
        if unknown:
            problems.append(
                f"{name}: the table demands {', '.join(unknown)}, absent from the "
                f"signature in {spec.defined_in}"
            )
    return problems


_CONTRACTS: dict[str, tuple[int, frozenset[str]]] | None = None


def _contracts() -> dict[str, tuple[int, frozenset[str]]]:
    """name -> (positional budget, positionally-reachable identities), memoized.

    Resolved from the tree once per process. An entry whose signature cannot be
    read is omitted, which makes its positional rule inert -- ``_table_drift``
    is what turns an unreadable signature into a hard failure, and it runs before
    the gate can pass or write a baseline.
    """
    global _CONTRACTS
    if _CONTRACTS is None:
        out: dict[str, tuple[int, frozenset[str]]] = {}
        for name, spec in AUTHZ_FUNCS.items():
            sig, _ = _read_signature(spec.defined_in, name)
            if sig is not None:
                out[name] = _positional_contract(sig, spec.identities)
        _CONTRACTS = out
    return _CONTRACTS


class Finding:
    """One flagged call: where it is, and which of the three shapes it is."""

    __slots__ = ("line", "callee", "reason")

    def __init__(self, line: int, callee: str, reason: str) -> None:
        self.line = line
        self.callee = callee
        self.reason = reason

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        return f"{self.line}: {self.callee} -- {self.reason}"


def _callee_name(node: ast.Call) -> str:
    fn = node.func
    if isinstance(fn, ast.Attribute):
        return fn.attr
    if isinstance(fn, ast.Name):
        return fn.id
    return ""


def _marker_lines(source: str) -> set[int]:
    """Line numbers whose COMMENT token carries the opt-out marker.

    tokenize rather than a substring scan, so the marker phrase inside a string
    literal (a docstring describing the gate, for instance -- this file's own
    prose would otherwise exempt calls) cannot exempt a call.

    The marker must sit on a line the CALL ITSELF SPANS, and two looser rules were
    tried and reverted. Extending the match one line up, then absorbing the whole
    comment block below a marker, gave the marker an unbounded FORWARD reach: it
    was provably able to exempt a real ``HookManager.on_tool_call`` two lines below
    while flagging the display-method call the marker actually named, and any
    comment merely MENTIONING the marker phrase became a suppression for whatever
    call followed it. A trailing comment on a multi-line call's opening line
    reaches every shape in the tree, so the loose rules bought one call site in
    exchange for a suppression mechanism that leaks.
    """
    lines: set[int] = set()
    try:
        for tok in tokenize.generate_tokens(io.StringIO(source).readline):
            if tok.type == tokenize.COMMENT and MARKER in tok.string:
                lines.add(tok.start[0])
    except (tokenize.TokenError, IndentationError, SyntaxError):
        pass  # the AST parse of the same source decides parseability
    return lines


PARTIAL_NAMES = frozenset({"partial"})


def _resolve_call(node: ast.Call) -> tuple[str, list[ast.expr], int]:
    """(callee, positional args, splat count), unwrapping a thread wrapper or partial.

    `to_thread(_vet_spawn_governance, key, target, app=...)` is a call to the vet
    whose arg 0 is the callee itself, so both the name and the positional budget
    must be read one slot in. Returning the splat count lets the caller flag an
    opaque `**kwargs`, which no keyword scan can see through.

    `functools.partial(governance_permits, scope, item)` is ALSO a call to the
    authorization function -- it binds arguments now and runs later -- and it is
    judged on what it binds. Missing that made the gate's own prescribed remedy
    into a total evasion: the finding text for a kwargless wrapper names
    `functools.partial` as the fix, and every partial-wrapped authorization call
    read clean, whether inside a wrapper or assigned to a bare name.
    """
    name = _callee_name(node)
    args = list(node.args)
    splats = sum(1 for kw in node.keywords if kw.arg is None)
    if name in PARTIAL_NAMES and args:
        target = args[0]
        bound = args[1:]
        if isinstance(target, ast.Name) and target.id in AUTHZ_FUNCS:
            return target.id, bound, splats
        if isinstance(target, ast.Attribute) and target.attr in AUTHZ_FUNCS:
            return target.attr, bound, splats
        return "", args, splats
    if name in THREAD_WRAPPERS and args:
        inner = args[0]
        args = args[1:]
        if isinstance(inner, ast.Name):
            return inner.id, args, splats
        if isinstance(inner, ast.Attribute):
            return inner.attr, args, splats
        # A `partial(...)` in the callee slot is judged as its own node, which
        # `ast.walk` reaches independently -- returning "" here avoids counting the
        # same authorization call twice.
        return "", args, splats
    return name, args, splats


def _kwargless_target(node: ast.Call) -> str:
    """The authorization callee a KWARGLESS_WRAPPERS call INVOKES, if any.

    `run_in_executor(executor, fn, *args)` puts the callee in slot 1, and the bare
    one-argument form puts it in slot 0. Only that slot is read: scanning every
    positional also matched an authorization function passed as DATA (a callback
    being registered, say), and then reported that "no identity it carries can be
    named" about a call that carries none and invokes nothing.
    """
    if not node.args:
        return ""
    target = node.args[1] if len(node.args) >= 2 else node.args[0]
    if isinstance(target, ast.Name) and target.id in AUTHZ_FUNCS:
        return target.id
    if isinstance(target, ast.Attribute) and target.attr in AUTHZ_FUNCS:
        return target.attr
    return ""


def _mentioned_keywords(node: ast.Call) -> tuple[set[str], int]:
    """Identity names this call mentions, plus the count of UNRESOLVABLE splats.

    A `**{"agent": x}` dict literal is read and its string keys count as
    mentioned -- the value is deliberately not inspected, because an explicit
    falsy IS the declaration this gate wants. A `**kwargs` forward cannot be
    proven complete and is counted as unresolvable.
    """
    named = {kw.arg for kw in node.keywords if kw.arg}
    opaque = 0
    for kw in node.keywords:
        if kw.arg is not None:
            continue
        if isinstance(kw.value, ast.Dict):
            named |= {
                k.value
                for k in kw.value.keys
                if isinstance(k, ast.Constant) and isinstance(k.value, str)
            }
        else:
            opaque += 1
    return named, opaque


def _recursive_call_ids(tree: ast.AST) -> set[int]:
    """`id()` of each call that is a RECURSIVE forward inside its own definition.

    Scoped to the definition's own body, never the file. The first cut collected
    every matching NAME in the file and skipped all calls to it anywhere in that
    file, which took `governance_profiles.py` -- the resolver module this gate is
    about -- entirely dark, and let a nested two-line callback that happens to
    share a name blind a whole file by accident.
    """
    exempt: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.name not in AUTHZ_FUNCS:
            continue
        for inner in ast.walk(node):
            if isinstance(inner, ast.Call) and _resolve_call(inner)[0] == node.name:
                exempt.add(id(inner))
    return exempt


def _findings_in_source(source: str) -> list[Finding]:
    """Every flagged authorization call in one file.

    Raises SyntaxError for an unparseable file: the caller turns that into a hard
    error, because "could not parse" must never read as "clean".
    """
    tree = ast.parse(source)
    markers = _marker_lines(source)
    exempt = _recursive_call_ids(tree)
    out: list[Finding] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        end = node.end_lineno or node.lineno
        if markers & set(range(node.lineno, end + 1)):
            continue
        if _callee_name(node) in KWARGLESS_WRAPPERS:
            inner = _kwargless_target(node)
            if inner:
                out.append(
                    Finding(
                        node.lineno,
                        inner,
                        "run_in_executor accepts no keyword arguments, so no identity "
                        "it carries can be named; route the call through "
                        "asyncio.to_thread, or a functools.partial that passes the "
                        "identities as keywords",
                    )
                )
            continue
        callee, positional, _ = _resolve_call(node)
        if callee not in AUTHZ_FUNCS or id(node) in exempt:
            continue
        spec = AUTHZ_FUNCS[callee]
        named, opaque = _mentioned_keywords(node)
        starred = sum(1 for a in positional if isinstance(a, ast.Starred))
        if starred or opaque:
            bags = (["*args"] if starred else []) + (["**splat"] if opaque else [])
            bag = " and ".join(bags)
            out.append(
                Finding(
                    node.lineno,
                    callee,
                    f"an opaque {bag} cannot be proven to carry every identity; "
                    "pass them explicitly or add the marker",
                )
            )
            continue
        budget, reachable = _contracts().get(callee, (0, frozenset()))
        if reachable and len(positional) > budget:
            out.append(
                Finding(
                    node.lineno,
                    callee,
                    f"{len(positional)} positional args exceeds the {budget} this "
                    f"signature allows: {', '.join(sorted(reachable))} is reachable "
                    "positionally, so an identity is riding a slot it does not name",
                )
            )
            continue
        missing = sorted(spec.identities - named)
        if missing:
            out.append(
                Finding(
                    node.lineno,
                    callee,
                    "does not state " + ", ".join(missing) + ' (pass the value, or "" to '
                    "declare this surface has none)",
                )
            )
    out.sort(key=lambda f: (f.line, f.callee))
    return out


def _scan(targets: tuple[str, ...]) -> tuple[dict[str, list[Finding]], list[str]]:
    """(path -> findings, unparseable paths). Paths are repo-relative, sorted."""
    found: dict[str, list[Finding]] = {}
    broken: list[str] = []
    for target in targets:
        base = ROOT / target
        if not base.exists():
            continue
        for path in sorted(base.rglob("*.py")):
            rel = path.relative_to(ROOT).as_posix()
            try:
                findings = _findings_in_source(path.read_text(encoding="utf-8"))
            except SyntaxError:
                broken.append(rel)
                continue
            except OSError:
                broken.append(rel)
                continue
            if findings:
                found[rel] = findings
    return found, broken


class BaselineError(RuntimeError):
    """The baseline file exists but cannot be trusted."""


def _read_baseline(path: Path) -> dict[str, int]:
    """Parse the baseline, raising rather than skipping a line it cannot read.

    A malformed count used to be silently skipped, so corrupting the count token on
    every line made this return `{}` -- which `--update-baseline` then read as "no
    baseline yet, this is a first creation" and re-seeded from the raw current
    counts, reporting success. A ratchet whose file can be quietly emptied into a
    fresh seed is not a ratchet.
    """
    entries: dict[str, int] = {}
    if not path.exists():
        return entries
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        count, _, rel = line.partition(" ")
        try:
            entries[rel.strip()] = int(count)
        except ValueError:
            raise BaselineError(
                f"{path.name}:{lineno}: cannot read a count from {line!r} -- the "
                "baseline is malformed; restore it from git rather than letting a "
                "refresh re-seed from the current tree"
            ) from None
    return entries


def _shrunken_baseline(baseline: dict[str, int], current: dict[str, int]) -> dict[str, int]:
    """The refresh result: counts only ever lowered, clean/gone entries dropped.

    Ported from ``check_subprocess_encoding.py``, whose ``run_gate`` has always
    called it. The first cut of this gate claimed the same shrink-only property in
    its docstring AND stamped it into the baseline header, then wrote the raw
    current counts -- so ``--update-baseline`` silently RAISED an entry, and a
    1 -> 2 on one line of a 35-line file is exactly the diff that rides along
    unnoticed in a large PR. The property the whole ratchet rests on was
    documented and absent.
    """
    survivors: dict[str, int] = {}
    for rel, recorded in baseline.items():
        now = current.get(rel, 0)
        if now > 0:
            survivors[rel] = min(recorded, now)
    return survivors


def _write_baseline(path: Path, entries: dict[str, int]) -> None:
    body = "".join(f"{entries[rel]} {rel}\n" for rel in sorted(entries))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "# authz-inputs gate baseline -- see scripts/check_authz_inputs.py.\n"
        "# Shrink-only: a file may improve, never regress.\n" + body,
        encoding="utf-8",
    )


def run_gate(baseline_path: Path, *, update: bool, list_only: bool) -> int:
    # Drift is checked before the gate can PASS or write a baseline, but after
    # --list: a developer mid-refactor still needs to read the backlog, and the
    # first cut returned 1 with no output at all for the documented "print the
    # backlog without failing" mode.
    drift = [] if list_only else _table_drift()
    if drift:
        print(HEADER)
        for line in drift:
            print(f"  ERROR table drift: {line}")
        print(
            "\n-> AUTHZ_FUNCS no longer matches the signatures it reads, so every\n"
            "   finding it produces may name a parameter that cannot be passed"
        )
        return 1

    current, broken = _scan(DEFAULT_TARGETS)
    counts = {rel: len(f) for rel, f in current.items()}

    if broken:
        print(HEADER)
        for rel in broken:
            print(f"  ERROR unparseable: {rel}")
        print("\n-> a file that cannot be parsed must never read as clean")
        return 1

    if list_only:
        print(HEADER)
        total = sum(counts.values())
        for rel in sorted(current):
            print(f"\n{rel} ({len(current[rel])})")
            for f in current[rel]:
                print(f"  :{f.line}  {f.callee} -- {f.reason}")
        print(f"\n{total} finding(s) in {len(current)} file(s) (report only)")
        return 0

    if update:
        baseline = _read_baseline(baseline_path)
        if not baseline_path.exists():
            # A first creation cannot be a regression -- there is nothing to raise.
            # Keyed on the FILE, never on "parsed to zero entries": an existing file
            # whose entries were emptied is a corrupted ratchet, not a fresh start.
            _write_baseline(baseline_path, counts)
            print(f"authz-inputs baseline seeded: {sum(counts.values())} in {len(counts)} file(s)")
            return 0
        if not baseline:
            print(HEADER)
            print(
                f"  ERROR {baseline_path.name} exists but records no entries.\n"
                "\n-> restore it from git. Delete the file explicitly if you really\n"
                "   intend to re-seed from the current tree."
            )
            return 1
        survivors = _shrunken_baseline(baseline, counts)
        pruned = len(baseline) - len(survivors)
        lowered = sum(1 for rel in survivors if survivors[rel] < baseline[rel])
        _write_baseline(baseline_path, survivors)
        print(f"pruned {pruned} entr(y/ies), lowered {lowered}; {len(survivors)} remain")
        return 0

    baseline = _read_baseline(baseline_path)
    failures: list[str] = []
    for rel in sorted(current):
        allowed = baseline.get(rel)
        if allowed is None:
            failures.append(f"{rel}: {len(current[rel])} finding(s) in a file with no baseline")
        elif len(current[rel]) > allowed:
            failures.append(f"{rel}: {len(current[rel])} finding(s) exceeds baseline {allowed}")
        else:
            continue
        for f in current[rel]:
            failures.append(f"    :{f.line}  {f.callee} -- {f.reason}")

    if failures:
        print(HEADER)
        for line in failures:
            print(f"  {line}")
        print(
            '\n-> pass the identity, or an explicit "" to declare the surface has none;\n'
            f"   a site that genuinely cannot: add  # {MARKER} <reason>"
        )
        return 1

    shrunk = {rel: c for rel, c in baseline.items() if counts.get(rel, 0) < c}
    gone = sorted(set(baseline) - set(counts))
    print(
        f"authz-inputs gate passed: {sum(counts.values())} known finding(s) "
        f"in {len(counts)} file(s) still listed."
    )
    if shrunk or gone:
        print(
            f"   {len(shrunk) + len(gone)} baseline entr(y/ies) improved -- "
            "prune with --update-baseline"
        )
    return 0


def _self_test() -> int:
    """Prove the gate on hand-written sources, independent of the tree's state.

    Shaped like the sibling gates: a `clean` probe must yield nothing and a
    `flagged` probe must yield exactly one finding whose reason contains the
    needle. The needle is the load-bearing half rather than decoration -- several
    defects found in review were a rule firing for the WRONG reason, which a bare
    count cannot tell apart from a right one. A `want: int` column was carried
    here for a while and was pure redundancy: no probe ever wanted two findings,
    so which table a probe sits in already says what it expects.
    """
    clean: dict[str, str] = {
        "complete call is clean": 'governance_permits("apps", n, session_key=k, agent=a, app=p)',
        "explicit empty is a DECLARATION, not a violation": 'governance_permits("apps", n, session_key=k, agent="", app="")',
        "the spawn ceiling through to_thread is unwrapped and judged": "await asyncio.to_thread(_vet_spawn_governance, k, t, app=p)",
        "a dict-literal splat IS resolvable": 'governance_permits(s, n, **{"session_key": k, "agent": a, "app": p})',
        "the marker exempts, as a comment": 'governance_permits("apps", n)  # authz-inputs: boot-time, no session',
        "a trailing marker on a MULTI-LINE call's opening line exempts it": "await self.on_tool_call(  # authz-inputs: unrelated same-named method\n    a, b, c, d\n)\n",
        "the definition itself is not a call site": "def governance_permits(scope, item, *, session_key='', agent='', app=''):\n    return _resolve(scope)\n",
        "B3: a recursive forward inside its OWN def is exempt": "def governance_permits(scope, item, **kw):\n    return governance_permits(scope, item, **kw)\n",
        "N1: a partial that DOES bind every identity is clean": "cb = functools.partial(governance_permits, s, i, session_key=k, agent=a, app=p)",
        "N1: a partial over an unrelated function is not our business": "cb = functools.partial(some_helper, s, i)",
        "H4: an authz function passed as DATA is not an invocation": "loop.run_in_executor(None, register_hook, governance_permits)",
        "resolved_agent is NOT demanded -- omitting it fail-closes, not widens": "gate.hooks.on_tool_call(title, session_key=k, agent=a, app=p)",
    }
    flagged: dict[str, tuple[str, str]] = {
        "a silently omitted identity is the defect": (
            'governance_permits("apps", n, session_key=k, agent=a)',
            "does not state app",
        ),
        "two omitted identities are one finding naming both": (
            'governance_permits("apps", n, session_key=k)',
            "does not state agent, app",
        ),
        "a threaded ceiling call that omits the app it has": (
            "await asyncio.to_thread(_vet_spawn_governance, k, t)",
            "does not state app",
        ),
        "the positional-collision trap: caller riding the app slot": (
            "await asyncio.to_thread(_vet_spawn_governance, k, t, c)",
            "app is reachable positionally",
        ),
        "an opaque splat cannot be proven complete": (
            "governance_permits(s, n, **ctx)",
            "opaque **splat",
        ),
        "a marker ABOVE a call does NOT exempt it -- no forward reach": (
            "# authz-inputs: this justifies something else entirely\nhooks.on_tool_call(title, session_key=k, agent=a)\n",
            "does not state app",
        ),
        "a marker cannot reach PAST a comment block onto a later call": (
            "# authz-inputs: the renderer's own display method, not\n# HookManager.on_tool_call -- a name collision\nhooks.on_tool_call(title, session_key=k, agent=a)\n",
            "does not state app",
        ),
        "a comment merely MENTIONING the marker suppresses nothing": (
            "# the authz-inputs: gate demands every identity be stated\nhooks.on_tool_call(title, session_key=k, agent=a)\n",
            "does not state app",
        ),
        "the marker inside a STRING does not exempt": (
            'governance_permits("apps", n, doc="authz-inputs: nope")',
            "does not state",
        ),
        "B3: defining the name does NOT blind the rest of the file": (
            "def governance_permits(scope, item, *, session_key='', agent='', app=''):\n    return _resolve(scope)\n\ndef elsewhere(k):\n    return governance_permits('tools', 'x', session_key=k)\n",
            "does not state agent, app",
        ),
        "B3: a nested callback sharing the name blinds nothing": (
            "def unrelated():\n    def on_tool_call(x):\n        return x\n    return hooks.on_tool_call(title, session_key=k, agent=a)\n",
            "does not state app",
        ),
        "B4: run_in_executor cannot name an identity, so the wrapper is the finding": (
            "loop.run_in_executor(None, _vet_spawn_governance, k, t, p)",
            "accepts no keyword arguments",
        ),
        "B4: the bare one-arg form resolves the callee, not the executor": (
            "loop.run_in_executor(_vet_spawn_governance)",
            "accepts no keyword arguments",
        ),
        "N1: a bare functools.partial is judged on what it binds": (
            "cb = functools.partial(governance_permits, s, i)",
            "does not state agent, app, session_key",
        ),
        "N1: a partial inside a kwargless wrapper is still judged": (
            "loop.run_in_executor(None, functools.partial(governance_permits, s, i))",
            "does not state agent, app, session_key",
        ),
        "N1: a partial inside to_thread is still judged": (
            "await asyncio.to_thread(functools.partial(governance_permits, s, i))",
            "does not state agent, app, session_key",
        ),
        "M5: an opaque *args defeats the positional budget and is unresolvable": (
            "governance_permits(*args, session_key=k, agent=a, app=p)",
            "opaque *args",
        ),
        "the CLI hooks gate shape: the app it has is omitted": (
            "gate.hooks.on_tool_call(title, session_key=k, agent=a)",
            "does not state app",
        ),
    }

    # (label, passed) per assertion. The run total is COUNTED off this rather than
    # declared in a constant: a hand-maintained total is the same drift surface as
    # the hand-maintained positional budget this gate stopped carrying.
    results: list[tuple[str, bool]] = []

    def check(label: str, ok: bool, detail: str = "") -> None:
        results.append((label, ok))
        if ok:
            print(f"  ok   {label}")
        else:
            print(f"  FAIL {label}" + (f": {detail}" if detail else ""))

    @contextlib.contextmanager
    def rooted(path: Path) -> Iterator[None]:
        """Point signature reads at a probe tree; ROOT is module-level state."""
        saved = globals()["ROOT"]
        globals()["ROOT"] = path
        try:
            yield
        finally:
            globals()["ROOT"] = saved

    for label, source in clean.items():
        found = _findings_in_source(source)
        check(label, not found, f"wanted no finding, got {found}")

    for label, (source, needle) in flagged.items():
        found = _findings_in_source(source)
        check(
            label,
            len(found) == 1 and needle in found[0].reason,
            f"wanted one finding matching {needle!r}, got {found}",
        )

    try:
        _findings_in_source("def broken(:\n")
    except SyntaxError:
        check("an unparseable source raises rather than reading clean", True)
    else:
        check("an unparseable source raises rather than reading clean", False)

    # B1: the table must agree with the signatures it claims to read, and the check
    # must actually FIRE on a fabricated parameter -- the original defect was a
    # table naming `caller_agent` while every case here still passed.
    drift = _table_drift()
    check("AUTHZ_FUNCS matches every signature it reads", not drift, str(drift))

    spec = AUTHZ_FUNCS["governance_permits"]
    AUTHZ_FUNCS["governance_permits"] = AuthzFunc(
        spec.defined_in, spec.identities | {"no_such_param"}
    )
    try:
        caught = _table_drift()
    finally:
        AUTHZ_FUNCS["governance_permits"] = spec
    check(
        "a fabricated identity parameter is caught as table drift",
        any("no_such_param" in line for line in caught),
        str(caught),
    )

    # MAJOR regression: the three drift causes must name THEMSELVES. Reporting an
    # unparseable file as "no definition found" sent the reader grepping for a def
    # the gate could see all along.
    missing, why_missing = _signature_params("src/kiro_crew/no_such_module.py", "x")
    absent, why_absent = _signature_params("src/kiro_crew/hooks.py", "no_such_function")
    real, why_real = _signature_params("src/kiro_crew/hooks.py", "on_tool_call")
    check(
        "a missing file, an absent def and a real def are told apart",
        missing is None
        and "does not exist" in why_missing
        and absent is None
        and "no module-level or class-body definition" in why_absent
        and real is not None
        and "session_key" in real,
        f"{why_missing!r} / {why_absent!r} / {why_real!r}",
    )

    probe = Path(tempfile.mkdtemp())
    (probe / "p").mkdir()

    # H1: the LAST same-scope def is the one Python binds. Returning the first let a
    # stale copy -- or the permissive half of a branch pair -- validate the table
    # against a signature nothing runs.
    (probe / "p" / "dbl.py").write_text(
        "def governance_permits(scope, item):\n    return 1\n\n"
        "def governance_permits(scope, item, *, session_key='', agent='', app=''):\n"
        "    return 2\n",
        encoding="utf-8",
    )
    # H2: a **kwargs catch-all accepts every identity, so drift must NOT fire.
    (probe / "p" / "kw.py").write_text(
        "def governance_permits(scope, item, **kwargs):\n    return 1\n", encoding="utf-8"
    )
    # `self` must not count toward a bound method's budget: every call site in the
    # tree invokes the bound method and never passes it.
    (probe / "p" / "meth.py").write_text(
        "class H:\n"
        "    def on_tool_call(self, tool_name, *, session_key='', agent='', app=''):\n"
        "        return 1\n",
        encoding="utf-8",
    )
    with rooted(probe):
        shadowed, _ = _signature_params("p/dbl.py", "governance_permits")
        catchall, _ = _signature_params("p/kw.py", "governance_permits")
        method, _ = _read_signature("p/meth.py", "on_tool_call")

    check(
        "a shadowed def resolves to the one Python binds, not the first",
        shadowed is not None and "session_key" in shadowed,
        str(shadowed),
    )
    check(
        "a **kwargs catch-all does not brick the drift check",
        catchall is not None and AUTHZ_FUNCS["governance_permits"].identities <= catchall,
        str(catchall),
    )
    check(
        "a bound method's leading self is excluded from the budget",
        method is not None and method.positional == ["tool_name"],
        str(method and method.positional),
    )

    # The narrowing: every budget is DERIVED from its signature, and the positional
    # rule is inert wherever an identity cannot travel positionally. A hand-written
    # budget was the same drift surface the table itself had.
    derived = _contracts()
    check(
        "every positional budget is derived from its real signature",
        derived
        == {
            "governance_permits": (2, frozenset()),
            "resolve_active_scope": (1, frozenset()),
            "on_tool_call": (1, frozenset()),
            "_vet_spawn_governance": (2, frozenset({"app"})),
        },
        str(derived),
    )
    inert = sorted(name for name, (_, reach) in derived.items() if not reach)
    check(
        "the positional rule is inert on the three keyword-only entry points",
        inert == ["governance_permits", "on_tool_call", "resolve_active_scope"],
        str(inert),
    )

    # B2: a malformed baseline must ERROR, never parse to zero and invite a re-seed.
    bad = probe / "bad-baseline.txt"
    bad.write_text("# header\nxx src/a.py\n", encoding="utf-8")
    try:
        _read_baseline(bad)
    except BaselineError:
        check("a malformed baseline raises rather than reading as empty", True)
    else:
        check("a malformed baseline raises rather than reading as empty", False)

    # B2: --update-baseline must never raise a recorded count.
    lowered = _shrunken_baseline({"a.py": 1, "b.py": 5}, {"a.py": 9, "b.py": 2})
    check(
        "a baseline refresh lowers and prunes, never raises",
        lowered == {"a.py": 1, "b.py": 2},
        str(lowered),
    )
    check(
        "a file that went clean is pruned from the baseline",
        _shrunken_baseline({"a.py": 3}, {}) == {},
    )

    failed = [label for label, ok in results if not ok]
    print(f"\nself-test: {len(results) - len(failed)} passed, {len(failed)} failed")
    return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    ap.add_argument(
        "--update-baseline",
        action="store_true",
        help="lower/prune baseline entries to the current counts; never raises one",
    )
    ap.add_argument("--list", action="store_true", help="print the backlog without failing")
    # `--test`, not `--self-test`: flag parity with check_subprocess_encoding.py,
    # which CI already invokes this way. A near-miss name is one more thing for a
    # future workflow edit to get silently wrong.
    ap.add_argument("--test", action="store_true", help="run the gate's own test cases")
    args = ap.parse_args(argv)
    if args.test:
        return _self_test()
    try:
        return run_gate(args.baseline, update=args.update_baseline, list_only=args.list)
    except BaselineError as exc:
        print(HEADER)
        print(f"  ERROR {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
