"""Generic auto-approval daemon for agent permission prompts.

The ``AutoApprover`` polls a :class:`PaneSession` and delegates prompt
detection to the :class:`AgentCLITool` that owns the session.
"""

from __future__ import annotations

import re
import threading
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import bashlex
import bashlex.ast

from .. import logger

if TYPE_CHECKING:
    from .runner import AgentCLITool, ToolRequest
    from .wezterm import PaneSession


_QUOTED_HEREDOC_RE = re.compile(r"<<['\"](\w+)['\"]")
_OPERATOR_RE = re.compile(r"\s*(?:&&|\|\|?|;)\s*")
_ASSIGNMENT_RE = re.compile(r"^[A-Za-z_]\w*=")


# ------------------------------------------------------------------
# Data model
# ------------------------------------------------------------------


@dataclass
class Rule:
    name: str
    patterns: list[re.Pattern[str]]
    reason: str | None = None
    dir: str | None = None  # "project_root" or "!project_root"


@dataclass
class RuleSet:
    default_reason: str
    deny: list[Rule] = field(default_factory=list)
    allow: list[Rule] = field(default_factory=list)


def load_rules(path: Path) -> RuleSet:
    """Parse a TOML rules file into a :class:`RuleSet`."""
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    deny = [
        Rule(
            name=r["name"],
            patterns=[re.compile(p) for p in r["patterns"]],
            reason=r.get("reason"),
            dir=r.get("dir"),
        )
        for r in data.get("deny", [])
    ]
    allow = [
        Rule(
            name=r["name"],
            patterns=[re.compile(p) for p in r["patterns"]],
            reason=r.get("reason"),
            dir=r.get("dir"),
        )
        for r in data.get("allow", [])
    ]
    return RuleSet(
        default_reason=data.get("default_reason", "try another approach"),
        deny=deny,
        allow=allow,
    )


# ------------------------------------------------------------------
# Command parsing
# ------------------------------------------------------------------


def _has_wrap_artifacts(commands: list[str]) -> bool:
    """Detect terminal line-wrap artifacts in parsed commands.

    A first-word starting with ``-`` comes from a wrapped ``find``
    option; a single-character first-word (like ``l`` from a split
    ``ls``) is also suspicious.
    """
    for cmd in commands:
        first = cmd.split()[0]
        if first.startswith("-") or len(first) <= 1:
            return True
    return False


def _try_bashlex(text: str) -> list[str] | None:
    """Try to parse *text* with bashlex, returning command segments or
    ``None`` on failure / wrap artifacts."""
    normalized = _QUOTED_HEREDOC_RE.sub(r"<<\1", text)
    commands: list[str] = []
    try:
        for node in bashlex.parse(normalized):
            _walk_node(node, commands)
    except Exception:
        return None
    if not commands or _has_wrap_artifacts(commands):
        return None
    return commands


def _extract_commands(command: str) -> list[str]:
    """Parse a shell command and return each command segment
    (executable + arguments) as a single string.

    Handles pipes, ``&&``/``||``/``;`` chains, command-scoped env vars,
    heredocs, and multiline commands.  Falls back to splitting on shell
    operators when bashlex cannot parse (e.g. terminal-wrapped lines).
    """
    # 1. Try bashlex on the original text.
    result = _try_bashlex(command)
    if result is not None:
        return result

    # 2. If the text contains newlines (possible terminal wrapping),
    #    retry with newlines removed.
    unwrapped = command.replace("\n", "")
    if unwrapped != command:
        result = _try_bashlex(unwrapped)
        if result is not None:
            return result

    # 3. Flat-split fallback — undo wrapping, split on shell operators,
    #    take the full segment minus leading env-var assignments.
    commands: list[str] = []
    flat = " ".join(unwrapped.split())
    for segment in _OPERATOR_RE.split(flat):
        segment = segment.strip()
        if not segment:
            continue
        words = segment.split()
        # Strip leading env var assignments
        while words and _ASSIGNMENT_RE.match(words[0]):
            words.pop(0)
        if words:
            commands.append(" ".join(words))

    return commands


def _walk_node(node: bashlex.ast.node, out: list[str]) -> None:
    """Recursively collect the full command string (all words) for every
    command node."""
    kind = node.kind
    if kind == "command":
        words = []
        for part in node.parts:
            if part.kind == "word":
                words.append(part.word)
        if words:
            out.append(" ".join(words))
    elif kind in ("list", "pipeline", "compound"):
        for part in node.parts:
            if hasattr(part, "kind"):
                _walk_node(part, out)
    elif kind in ("if", "for", "while", "until"):
        for attr in ("parts", "list", "body"):
            for child in getattr(node, attr, []):
                if hasattr(child, "kind"):
                    _walk_node(child, out)


# ------------------------------------------------------------------
# AutoApprover
# ------------------------------------------------------------------


class AutoApprover:
    """Daemon thread: poll a pane session, detect prompts via the
    :class:`AgentCLITool`, approve Bash commands that pass the rule set.
    Non-Bash tool requests are always approved.  Denied commands are
    actively rejected with a reason message."""

    POLL_INTERVAL = 0.8
    COOLDOWN = 2.0

    def __init__(
        self,
        backend: AgentCLITool,
        session: PaneSession,
        rules_path: Path,
        project_root: Path | None = None,
    ) -> None:
        self._backend = backend
        self._session = session
        self._rules_path = rules_path
        self._rules = load_rules(rules_path)
        self._project_root = project_root.resolve() if project_root else None
        self._cwd = self._project_root  # best-effort cwd tracking
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_hash: int = 0
        n = len(self._rules.deny) + len(self._rules.allow)
        logger.info(f"Auto-approver loaded {n} rule groups from {rules_path.name}")

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="auto-approve"
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                text = self._session.get_text()
                h = hash(text)
                if text and h != self._last_hash:
                    self._last_hash = h
                    if self._check_and_respond(text):
                        self._stop.wait(self.COOLDOWN)
                        continue
            except Exception as exc:
                logger.debug(f"Auto-approve poll error: {exc}")
            self._stop.wait(self.POLL_INTERVAL)

    def _check_and_respond(self, text: str) -> bool:
        req = self._backend.detect_prompt(text)
        if req is None:
            return False

        allowed, reason = self._check_request(req)
        if allowed:
            logger.info(f"Auto-approving: {req}")
            self._session.send_keys(self._backend.approve_key)
            return True

        logger.warning(f"Denying: {req} ({reason})")
        self._deny(reason)
        return True

    def _deny(self, reason: str) -> None:
        self._session.send_keys(self._backend.deny_key)
        self._wait_for_prompt_clear()
        msg = (
            f"This command was denied: {reason}. "
            f"See @{self._rules_path} for the rules"
        )
        self._backend.send_text(self._session, msg)

    def _wait_for_prompt_clear(self, timeout: float = 5.0) -> None:
        """Poll until the permission prompt disappears, then wait a bit
        more for the input UI to render."""
        interval = 0.3
        elapsed = 0.0
        while elapsed < timeout:
            self._stop.wait(interval)
            elapsed += interval
            text = self._session.get_text()
            if not self._backend.detect_prompt(text):
                self._stop.wait(0.5)
                return
        logger.debug("Timed out waiting for prompt to clear")

    # ------------------------------------------------------------------
    # Rule evaluation
    # ------------------------------------------------------------------

    def _check_request(self, req: ToolRequest) -> tuple[bool, str]:
        """Return ``(allowed, reason)``.  *reason* is meaningful only on deny."""
        # Non-Bash tools are always approved.
        if req.tool != "Bash":
            return True, ""

        if not req.command:
            return False, self._rules.default_reason

        commands = _extract_commands(req.command)
        if not commands:
            return False, self._rules.default_reason

        # 1. Deny rules — first match wins (only if constraints pass).
        for rule in self._rules.deny:
            if any(
                any(pat.search(cmd) for pat in rule.patterns)
                for cmd in commands
            ):
                if self._check_constraints(rule):
                    return False, rule.reason or self._rules.default_reason

        # 2. Allow rules — every command must match at least one group
        #    whose constraints pass.
        for cmd in commands:
            matched_rule = None
            for rule in self._rules.allow:
                if any(pat.search(cmd) for pat in rule.patterns):
                    matched_rule = rule
                    break
            if matched_rule is None:
                return False, self._rules.default_reason
            if not self._check_constraints(matched_rule):
                return False, matched_rule.reason or self._rules.default_reason

        return True, ""

    def _check_constraints(self, rule: Rule) -> bool:
        """Check optional constraints on a rule.  Returns True if all pass."""
        if rule.dir is None:
            return True
        return self._check_dir_constraint(rule.dir)

    def _check_dir_constraint(self, dir_spec: str) -> bool:
        """Check that the tracked cwd satisfies the directory constraint.

        ``"project_root"``  → cwd must be within the project tree.
        ``"!project_root"`` → cwd must be *outside* the project tree.
        """
        negate = dir_spec.startswith("!")
        name = dir_spec.lstrip("!")

        if name == "project_root":
            if self._project_root is None or self._cwd is None:
                return True  # no project root configured — pass
            try:
                inside = self._cwd.is_relative_to(self._project_root)
            except (ValueError, TypeError):
                inside = False
            return not inside if negate else inside

        return True  # unknown dir name — pass


if __name__ == "__main__":

    def _test() -> None:
        ok = True

        def check(label: str, got: object, want: object) -> None:
            nonlocal ok
            status = "PASS" if got == want else "FAIL"
            if status == "FAIL":
                ok = False
            print(f"  [{status}] {label}: got={got!r} want={want!r}")

        # -- _extract_commands -------------------------------------------
        print("_extract_commands:")
        check("simple", _extract_commands("git status"), ["git status"])
        check("env var", _extract_commands("FOO=bar git log"), ["git log"])
        check("multi env", _extract_commands("A=1 B=2 ./pts build"), ["./pts build"])
        check(
            "pipe",
            _extract_commands("git log | grep main"),
            ["git log", "grep main"],
        )
        check(
            "chain &&",
            _extract_commands("cd /tmp && git status"),
            ["cd /tmp", "git status"],
        )
        check("chain ;", _extract_commands("echo hi ; ls"), ["echo hi", "ls"])
        check(
            "multiline",
            _extract_commands("git add .\ngit commit -m 'msg'"),
            ["git add .", "git commit -m msg"],
        )
        check(
            "pipe + env",
            _extract_commands("X=1 cat foo | sort"),
            ["cat foo", "sort"],
        )

        # Heredoc — just verify the single command starts with "git commit"
        hd = _extract_commands(
            "git commit -m \"$(cat <<'EOF'\nFix bug\n\nCo-Authored-By: Claude\nEOF\n)\""
        )
        check("heredoc len", len(hd), 1)
        check("heredoc prefix", hd[0].startswith("git commit"), True)

        # Terminal wrapping at a space: trailing space preserved by
        # _detect_prompt's lstrip(), so newline removal recovers the
        # original command.
        check(
            "terminal wrap at space",
            _extract_commands(
                'cd /c/Repos/PTStudio && find . -type f \\( -name "*.h" \n'
                '-o -name "*.cpp" \\)'
            ),
            [
                "cd /c/Repos/PTStudio",
                'find . -type f ( -name *.h -o -name *.cpp )',
            ],
        )

        # Terminal wrapping mid-word: 'ls' split as 'l\ns'
        check(
            "wrap mid-word",
            _extract_commands(
                'ls /c/Repos/PTStudio/_build/windows-x64/deps/*.dll 2>/dev/null; echo "---"; l\n'
                "s /"
            ),
            [
                "ls /c/Repos/PTStudio/_build/windows-x64/deps/*.dll",
                "echo ---",
                "ls /",
            ],
        )

        # -- check_request via RuleSet -----------------------------------
        from .runner import ToolRequest

        rules = RuleSet(
            default_reason="not allowed",
            deny=[
                Rule(
                    name="destructive_always",
                    patterns=[re.compile(r"^sudo\b")],
                    reason="sudo is forbidden",
                ),
                Rule(
                    name="destructive_outside",
                    patterns=[re.compile(r"^rm\b")],
                    dir="!project_root",
                    reason="rm only allowed inside project",
                ),
            ],
            allow=[
                Rule(
                    name="vcs",
                    patterns=[re.compile(r"^git\b"), re.compile(r"^grep\b")],
                ),
                Rule(
                    name="destructive_inside",
                    patterns=[re.compile(r"^rm\b")],
                    dir="project_root",
                ),
                Rule(
                    name="build",
                    patterns=[re.compile(r"^\./pts\b")],
                    dir="project_root",
                    reason="must be in project",
                ),
            ],
        )

        class _FakeApprover:
            """Minimal stand-in to test _check_request."""

            def __init__(self, rs: RuleSet, project_root: Path | None) -> None:
                self._rules = rs
                self._project_root = project_root
                self._cwd = project_root

            check_request = AutoApprover._check_request
            _check_constraints = AutoApprover._check_constraints
            _check_dir_constraint = AutoApprover._check_dir_constraint

        root = Path("/c/Repos/PTStudio")
        fa = _FakeApprover(rules, root)

        print("\ncheck_request:")
        # Allow
        a, r = fa.check_request(ToolRequest("Bash", "git status"))
        check("allow git", (a, r), (True, ""))

        a, r = fa.check_request(ToolRequest("Bash", "git log | grep main"))
        check("allow pipe", (a, r), (True, ""))

        # Deny via deny rule (no dir constraint — always denied)
        a, r = fa.check_request(ToolRequest("Bash", "sudo rm /etc/hosts"))
        check("deny sudo", (a, r), (False, "sudo is forbidden"))

        # rm inside project root → deny rule skipped (constraint !project_root
        # fails when inside), allow rule matches → allowed
        fa._cwd = root
        a, r = fa.check_request(ToolRequest("Bash", "rm temp.txt"))
        check("allow rm inside", (a, r), (True, ""))

        # rm outside project root → deny rule matches → denied
        fa._cwd = Path("/tmp")
        a, r = fa.check_request(ToolRequest("Bash", "rm temp.txt"))
        check("deny rm outside", (a, r), (False, "rm only allowed inside project"))

        # Deny via no-match (default reason)
        fa._cwd = root
        a, r = fa.check_request(ToolRequest("Bash", "curl http://x"))
        check("deny unlisted", (a, r), (False, "not allowed"))

        # Allow non-Bash tool
        a, r = fa.check_request(ToolRequest("Read"))
        check("non-bash tool", (a, r), (True, ""))

        # Allow ./pts when cwd is project root
        a, r = fa.check_request(ToolRequest("Bash", "./pts build"))
        check("allow ./pts at root", (a, r), (True, ""))

        # Deny ./pts when cwd is outside project
        fa._cwd = Path("/tmp")
        a, r = fa.check_request(ToolRequest("Bash", "./pts build"))
        check("deny ./pts outside", (a, r), (False, "must be in project"))

        # Allow ./pts in subdirectory
        fa._cwd = Path("/c/Repos/PTStudio/core")
        a, r = fa.check_request(ToolRequest("Bash", "./pts build"))
        check("allow ./pts subdir", (a, r), (True, ""))

        # No command
        a, r = fa.check_request(ToolRequest("Bash", None))
        check("no command", (a, r), (False, "not allowed"))

        print()
        print("ALL PASSED" if ok else "SOME TESTS FAILED")
        raise SystemExit(0 if ok else 1)

    _test()
