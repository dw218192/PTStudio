"""Subprocess execution and diagnostics shared by the Pixi tasks.

Ported helpers retain the MIT notice in tasks/LICENSE.
"""

from __future__ import annotations

import contextlib
import functools
import logging
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
import time
from collections.abc import Generator
from pathlib import Path
from typing import Any

from colorama import Fore, Style, just_fix_windows_console

just_fix_windows_console()


def _level_color(levelno: int) -> str:
    if levelno >= logging.ERROR:
        return Fore.RED
    if levelno >= logging.WARNING:
        return Fore.YELLOW
    return Fore.CYAN


class ToolFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        color = _level_color(record.levelno)
        message = record.getMessage()
        return f"{color}[{record.levelname.lower()}]{Style.RESET_ALL} {message}"


logger = logging.getLogger("ptstudio.tasks")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(ToolFormatter())
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False


def detect_platform_identifier(
    platform_override: str | None = None,
    conan_profile_path: Path | None = None,
) -> str:
    """Detect platform identifier for build directory structure.

    Priority: 1. Explicit override  2. Conan profile  3. Host auto-detect
    """
    if platform_override:
        return platform_override

    if conan_profile_path and conan_profile_path.exists():
        try:
            profile_content = conan_profile_path.read_text()
            os_match = re.search(r"^os=(\w+)", profile_content, re.MULTILINE)
            arch_match = re.search(r"^arch=(\w+)", profile_content, re.MULTILINE)
            if os_match and arch_match:
                return _map_platform_identifier(os_match.group(1), arch_match.group(1))
        except (OSError, UnicodeDecodeError):
            pass

    system = platform.system()
    machine = platform.machine().lower()

    if machine in ("x86_64", "amd64"):
        arch = "x64"
    elif machine in ("arm64", "aarch64", "armv8"):
        arch = "arm64"
    else:
        arch = machine

    if system == "Windows":
        return f"windows-{arch}"
    elif system == "Linux":
        return f"linux-{arch}"
    elif system == "Darwin":
        return f"macos-{arch}"
    else:
        return f"{system.lower()}-{arch}"


def _map_platform_identifier(os_val: str, arch_val: str) -> str:
    """Map Conan os/arch settings to platform identifier."""
    if os_val == "Emscripten" and arch_val == "wasm":
        return "emscripten"

    os_map = {
        "Windows": "windows",
        "Linux": "linux",
        "Macos": "macos",
        "Darwin": "macos",
    }
    os_normalized = os_map.get(os_val, os_val.lower())

    arch_map = {
        "x86_64": "x64",
        "x86": "x86",
        "armv8": "arm64",
        "armv8_32": "arm",
        "wasm": "wasm",
    }
    arch_normalized = arch_map.get(arch_val, arch_val.lower())

    return f"{os_normalized}-{arch_normalized}"


def is_windows() -> bool:
    return platform.system() == "Windows"


def _is_ci() -> bool:
    return os.environ.get("GITHUB_ACTIONS") == "true"


@contextlib.contextmanager
def log_section(title: str) -> Generator[None, None, None]:
    """Foldable CI section or styled terminal header."""
    if _is_ci():
        print(f"::group::{title}", flush=True)
    else:
        logger.info(f"-- {title} --")
    try:
        yield
    finally:
        if _is_ci():
            print("::endgroup::", flush=True)


def print_subprocess_line(line: str) -> None:
    text = line.rstrip()
    print(f"{Style.DIM}{text}{Style.RESET_ALL}")


@functools.cache
def find_executable(name: str) -> str:
    """Find an executable in the active Pixi environment or PATH."""
    adjacent = Path(sys.executable).parent / (name + (".exe" if is_windows() else ""))
    found = str(adjacent) if adjacent.is_file() else shutil.which(name)
    if not found:
        raise FileNotFoundError(f"{name} is unavailable; run pixi install")
    return found


def sanitized_subprocess_env() -> dict[str, str]:
    """Keep Pixi's PATH while isolating external Python interpreters."""
    return {"PYTHONPATH": "", "PYTHONHOME": ""}


class ShellCommand:
    """A command prepared for subprocess execution.

    Constructor handles env-script shell wrapping (platform-correct),
    suffix resolution, and environment merging.  Execution methods
    pass through to subprocess with the prepared state.
    """

    def __init__(
        self,
        cmd: list[str],
        *,
        env_script: Path | None = None,
        env: dict[str, str] | None = None,
        cwd: Path | None = None,
    ) -> None:
        self._cmd: list[str] | str = cmd
        self._shell = False
        self._env = {**os.environ, **env} if env else None
        self._cwd = cwd
        self._env_script: Path | None = None

        if env_script is not None:
            script = env_script
            if not script.suffix:
                script = script.with_suffix(".bat" if is_windows() else ".sh")
            self._env_script = script
            # Auto-sanitize when sourcing an env script -- the script sets
            # up the correct PATH for external tools, so strip the venv's
            # Python contamination to avoid DLL/PATH conflicts.
            sanitized = sanitized_subprocess_env()
            if env:
                sanitized.update(env)
            self._env = {**os.environ, **sanitized}
            if is_windows():
                cmd_str = subprocess.list2cmdline(cmd)
                self._cmd = f'call "{script}" >nul && {cmd_str}'
            else:
                cmd_str = shlex.join(cmd)
                self._cmd = f'. "{script}" >/dev/null && {cmd_str}'
            self._shell = True

    def run(self, **kwargs: Any) -> subprocess.CompletedProcess:
        """Execute via subprocess.run. Extra kwargs override defaults."""
        return subprocess.run(
            self._cmd,
            shell=self._shell,
            env=self._env,
            cwd=self._cwd,
            **kwargs,
        )

    def popen(self, **kwargs: Any) -> subprocess.Popen:
        """Execute via subprocess.Popen. Extra kwargs override defaults."""
        return subprocess.Popen(
            self._cmd,
            shell=self._shell,
            env=self._env,
            cwd=self._cwd,
            **kwargs,
        )

    def exec(self, log_file: Path | None = None) -> None:
        """Run with fail-loud semantics.

        Checks that the env script exists, optionally tees output to
        *log_file*, and calls ``sys.exit`` on non-zero return code.
        """
        if self._env_script is not None and not self._env_script.exists():
            logger.error(f"env_script not found: {self._env_script}")
            sys.exit(1)
        if log_file:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            with open(log_file, "w", encoding="utf-8", errors="replace") as f:
                process = self.popen(
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    bufsize=1,
                )
                for line in process.stdout:
                    print_subprocess_line(line)
                    f.write(line)
                process.wait()
                if process.returncode != 0:
                    sys.exit(process.returncode)
        else:
            try:
                self.run(check=True)
            except subprocess.CalledProcessError as e:
                sys.exit(e.returncode)


def remove_tree_with_retries(
    path: Path,
    attempts: int = 5,
    delay: float = 1.0,
) -> None:
    """Remove a directory tree with retry logic for locked files (Windows)."""
    for attempt in range(attempts):
        try:
            shutil.rmtree(path)
            return
        except PermissionError:
            if attempt < attempts - 1:
                logger.warning(
                    f"Permission denied removing {path}, "
                    f"retrying in {delay}s ({attempt + 1}/{attempts})"
                )
                time.sleep(delay)
            else:
                raise


class CommandGroup:
    """A labeled unit of work that runs commands and reports results.

    Usage::

        with CommandGroup("Building") as g:
            g.run(["cmake", "--build", "build"])
            g.run(["cmake", "--install", "build"])

    Features:
    - Labels each phase with a clear header
    - Tracks pass/fail per group
    - Dimmed subprocess output, summary on completion
    - Optional per-group log file
    - CI fold markers (``::group::``) in GitHub Actions
    """

    def __init__(
        self,
        label: str,
        log_file: Path | None = None,
        env_script: Path | None = None,
        cwd: Path | None = None,
        env: dict[str, str] | None = None,
    ) -> None:
        self.label = label
        self.log_file = log_file
        self.env_script = env_script
        self.cwd = cwd
        self.env = env
        self._commands_run = 0
        self._failed = False

    def __enter__(self) -> CommandGroup:
        if _is_ci():
            print(f"::group::{self.label}", flush=True)
        else:
            logger.info(f"-- {self.label} --")
        return self

    def __exit__(self, exc_type: type | None, exc_val: BaseException | None, exc_tb: Any) -> None:
        if _is_ci():
            print("::endgroup::", flush=True)
        if exc_type is not None:
            return  # let the exception propagate
        if self._failed:
            logger.error(f"  FAIL {self.label} failed")
        else:
            logger.info(f"  OK {self.label} ({self._commands_run} command(s))")

    def run(
        self,
        cmd: list[str],
        log_file: Path | None = None,
        env_script: Path | None = None,
        cwd: Path | None = None,
        env: dict[str, str] | None = None,
    ) -> None:
        """Run a command within this group.

        Per-call *log_file*, *env_script*, *cwd*, and *env* override the
        group defaults.  Per-call *env* is merged on top of group-level env.
        """
        lf = log_file or self.log_file
        es = env_script or self.env_script
        cw = cwd or self.cwd
        merged_env = {**(self.env or {}), **(env or {})} or None
        try:
            ShellCommand(cmd, env_script=es, env=merged_env, cwd=cw).exec(log_file=lf)
            self._commands_run += 1
        except SystemExit:
            self._failed = True
            raise


def to_cmake_build_type(value: str | None) -> str:
    if not value:
        return "Debug"
    mapping = {
        "debug": "Debug",
        "release": "Release",
        "relwithdebinfo": "RelWithDebInfo",
        "minsizerel": "MinSizeRel",
    }
    return mapping.get(str(value).casefold(), str(value))
