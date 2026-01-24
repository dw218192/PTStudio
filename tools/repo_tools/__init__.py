"""Shared utilities for repo tools."""

import functools
import logging
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

from colorama import Fore, Style, init as colorama_init

colorama_init()


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


logger = logging.getLogger("repo_tools")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(ToolFormatter())
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False


def is_windows() -> bool:
    return platform.system() == "Windows"


def is_linux() -> bool:
    return platform.system() == "Linux"


def is_macos() -> bool:
    return platform.system() == "Darwin"


def print_tool(message: str) -> None:
    print(f"{Fore.CYAN}[pts]{Style.RESET_ALL} {message}", flush=True)


def print_subprocess_line(line: str) -> None:
    text = line.rstrip("\n")
    print(f"{Style.DIM}{text}{Style.RESET_ALL}")


@functools.cache
def find_venv_executable(name: str) -> str:
    """Find an executable in the virtual environment, fallback to system PATH."""
    # Get the Scripts/bin directory relative to the Python executable
    python_exe = Path(sys.executable)
    scripts_dir = python_exe.parent
    exe_path = scripts_dir / (name + (".exe" if sys.platform == "win32" else ""))

    if exe_path.exists():
        return str(exe_path)

    # Fallback to system PATH
    exe_path = shutil.which(name)
    if exe_path:
        return exe_path

    logger.warning(f"Executable {name} not found in virtual environment or system PATH")
    return name


def run_command(cmd: list[str], log_file: Optional[Path] = None) -> None:
    """Run a command and optionally tee output to a log file."""
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with open(log_file, "w", encoding="utf-8", errors="replace") as f:
            process = subprocess.Popen(
                cmd,
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
        subprocess.run(cmd, check=True)


def ensure_conan_profile() -> None:
    """Ensure Conan profiles exist, run detect if needed."""
    profile_dir = Path.home() / ".conan2" / "profiles"

    if not profile_dir.exists() or not any(profile_dir.iterdir()):
        print_tool("No Conan profiles found. Running 'conan profile detect'...")
        conan_exe = find_venv_executable("conan")
        subprocess.run([conan_exe, "profile", "detect"], check=True)
    else:
        print_tool("Conan profiles already exist.")
