"""Launch subcommand implementation - runs executables."""

from __future__ import annotations

import http.server
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import webbrowser
import zipfile
import io
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.parse import quote, unquote
from urllib.request import urlopen

import click

from repo_tools.build.conan import load_conan_env
from repo_tools.core import (
    RepoTool,
    ShellCommand,
    ToolContext,
    detect_platform_identifier,
    is_windows,
    log_section,
    logger,
    to_cmake_build_type,
)

# Tracy release to download — update when upgrading tracy Conan package
_TRACY_VERSION = "0.13.1"
_TRACY_VIEWER_URL = (
    f"https://github.com/wolfpld/tracy/releases/download/v{_TRACY_VERSION}/"
    f"windows-{_TRACY_VERSION}.zip"
)


def _ensure_tracy_viewer(workspace_root: Path) -> Path:
    """Download Tracy profiler viewer if not cached."""
    cache_dir = workspace_root / "_build" / "tools" / "tracy"
    viewer = cache_dir / "tracy-profiler.exe"

    if viewer.exists():
        return viewer

    logger.info(f"Downloading Tracy profiler v{_TRACY_VERSION}...")
    cache_dir.mkdir(parents=True, exist_ok=True)

    resp = urlopen(_TRACY_VIEWER_URL)
    with zipfile.ZipFile(BytesIO(resp.read())) as zf:
        zf.extractall(cache_dir)

    if not viewer.exists():
        raise RuntimeError(
            f"tracy-profiler.exe not found in downloaded archive from {_TRACY_VIEWER_URL}"
        )

    logger.info(f"Tracy viewer cached at {viewer}")
    return viewer


def _interactive_select(exe_paths: list[Path]) -> Path | None:
    """Show interactive menu to select an executable."""
    from InquirerPy import inquirer

    choices = [exe.stem for exe in sorted(exe_paths)]
    if not choices:
        return None

    try:
        selected = inquirer.select(
            message="Select executable to launch:",
            choices=choices,
        ).execute()
    except KeyboardInterrupt:
        return None

    if selected is None:
        return None

    for exe in exe_paths:
        if exe.stem == selected:
            return exe
    return None


def _discover_executables(target_dir: Path, is_emscripten: bool = False) -> list[Path]:
    """Discover executable files in a directory recursively."""
    if not target_dir.exists() or not target_dir.is_dir():
        return []

    exe_paths = []
    try:
        for file in target_dir.rglob("*"):
            if not file.is_file():
                continue
            if is_emscripten:
                if file.suffix.lower() in (".html", ".js"):
                    exe_paths.append(file)
            elif is_windows():
                if file.suffix.lower() == ".exe":
                    exe_paths.append(file)
            else:
                try:
                    if os.access(file, os.X_OK):
                        exe_paths.append(file)
                except OSError:
                    continue
    except (PermissionError, OSError) as e:
        logger.warning(f"Could not scan {target_dir}: {e}")

    if is_emscripten:
        # Each Emscripten target emits .html + .js; deduplicate by stem, keep .js
        by_stem: dict[str, Path] = {}
        for f in exe_paths:
            if f.stem not in by_stem or f.suffix.lower() == ".js":
                by_stem[f.stem] = f
        exe_paths = list(by_stem.values())

    return exe_paths


def _resolve_runtime_deploy_dir(build_dir: Path, conan_deps_root: str | None = None) -> Path | None:
    """Return the runtime_deploy directory if it exists.

    When Conan's ``runtime_deploy`` deployer is used, shared libraries are
    copied into the deployer-folder as a flat directory.  This can be added
    to PATH directly, without needing a conanrun script.

    Uses *conan_deps_root* (from the ``{conan_deps_root}`` config token)
    when available, otherwise falls back to ``build_dir/../deps``.
    """
    deps_dir = Path(conan_deps_root) if conan_deps_root else build_dir.parent / "deps"
    if not deps_dir.exists():
        return None
    dll_ext = ".dll" if is_windows() else ".so"
    for _ in deps_dir.glob(f"*{dll_ext}"):
        return deps_dir
    return None


def _resolve_env_script(build_dir: Path, is_emscripten: bool) -> Path | None:
    """Return the correct Conan env script for the platform.

    Native builds use ``conanrun`` (runtime DLL paths).
    Emscripten builds use ``conanbuild`` (emsdk tools: node, emrun).
    """
    name = "conanbuild" if is_emscripten else "conanrun"
    script = build_dir / name
    suffix = ".bat" if is_windows() else ".sh"
    resolved = script.with_suffix(suffix)
    return resolved if resolved.exists() else None



def _find_browser() -> tuple[Path, list[str]] | None:
    """Find a browser executable and return (path, isolation_args).

    Isolation args prevent the browser from reusing an existing process so the
    launched process is trackable (we can detect when the window closes).
    """
    _chromium_args = [
        "--new-window",
        "--no-first-run",
        "--no-default-browser-check",
        "--disable-sync",
        "--disable-extensions",
        "--disable-background-networking",
    ]

    if is_windows():
        localappdata = os.environ.get("LOCALAPPDATA", "")
        search: list[tuple[Path, list[str]]] = []
        # Edge (always on Windows 10+)
        for d in [
            Path("C:/Program Files (x86)/Microsoft/Edge/Application"),
            Path("C:/Program Files/Microsoft/Edge/Application"),
        ]:
            search.append((d / "msedge.exe", _chromium_args))
        # Chrome
        chrome_dirs: list[Path] = []
        if localappdata:
            chrome_dirs.append(Path(localappdata) / "Google/Chrome/Application")
        chrome_dirs.extend([
            Path("C:/Program Files/Google/Chrome/Application"),
            Path("C:/Program Files (x86)/Google/Chrome/Application"),
        ])
        for d in chrome_dirs:
            search.append((d / "chrome.exe", _chromium_args))
        for exe, args in search:
            if exe.exists():
                return exe, args
    else:
        for name in ["google-chrome", "chromium-browser", "chromium", "firefox"]:
            which = shutil.which(name)
            if which:
                if "firefox" in name:
                    return Path(which), ["-new-window"]
                return Path(which), _chromium_args

    return None


class _WasmHandler(http.server.SimpleHTTPRequestHandler):
    """HTTP handler for Emscripten builds.

    Serves static files with COOP/COEP headers (required for
    SharedArrayBuffer / -pthread) and handles the ``--emrun`` POST protocol
    that forwards stdout/stderr from the browser to the terminal.
    """

    # Set by _serve_emscripten before the server starts.
    page_exit_code: int | None = None
    server_ref: http.server.HTTPServer | None = None
    capture_buffer: io.StringIO | None = None

    def end_headers(self) -> None:
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        super().end_headers()

    def do_POST(self) -> None:  # noqa: N802
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self.send_response(400)
            self.end_headers()
            return
        data = self.rfile.read(length).decode("utf-8", errors="replace")
        data = unquote(data.replace("+", " "))

        if data.startswith("^exit^"):
            code_str = data[6:]
            try:
                _WasmHandler.page_exit_code = int(code_str) if code_str else 0
            except ValueError:
                logger.warning(f"Malformed exit code: {code_str!r}")
                _WasmHandler.page_exit_code = 1
            self.send_response(200)
            self.end_headers()
            if _WasmHandler.server_ref:
                threading.Thread(
                    target=_WasmHandler.server_ref.shutdown, daemon=True,
                ).start()
            return

        if data.startswith(("^out^", "^err^")):
            # Format: ^out^SEQ^message or ^err^SEQ^message
            try:
                i = data.index("^", 5)
                msg = data[i + 1:]
            except ValueError:
                msg = data[5:]
            if _WasmHandler.capture_buffer is not None:
                _WasmHandler.capture_buffer.write(msg)
                if not msg.endswith("\n"):
                    _WasmHandler.capture_buffer.write("\n")
            else:
                stream = sys.stderr if data.startswith("^err^") else sys.stdout
                stream.write(msg)
                if not msg.endswith("\n"):
                    stream.write("\n")
                stream.flush()

        self.send_response(200)
        self.end_headers()

    def guess_type(self, path: str) -> str:
        if path.endswith(".wasm"):
            return "application/wasm"
        return super().guess_type(path)

    def log_message(self, format: str, *args: Any) -> None:
        # Suppress request-level noise; errors still go to stderr via log_error.
        pass


def _serve_emscripten(
    html_path: Path,
    args: list[str] | None = None,
    capture_output: bool = False,
    timeout: float | None = None,
    headless: bool = False,
) -> subprocess.CompletedProcess:
    """Serve an Emscripten build and open a tracked browser process.

    Uses a built-in HTTP server with COOP/COEP headers and handles the
    ``--emrun`` POST protocol so stdout/stderr from the WASM page appear
    in the terminal.

    Args:
        args: CLI arguments forwarded to the WASM app via URL query params.
        capture_output: Buffer stdout/stderr instead of printing.
        timeout: Kill browser after this many seconds (None = wait forever).
        headless: Launch Chromium with ``--headless=new`` (no visible window).
    """
    serve_dir = str(html_path.parent)
    port = 6931

    def make_handler(*a: Any, **kw: Any) -> _WasmHandler:
        return _WasmHandler(*a, directory=serve_dir, **kw)
    try:
        server = http.server.ThreadingHTTPServer(("localhost", port), make_handler)
    except OSError as e:
        logger.error(f"Port {port} already in use: {e}")
        return subprocess.CompletedProcess(args=[str(html_path)], returncode=1)
    _WasmHandler.page_exit_code = None
    _WasmHandler.server_ref = server
    _WasmHandler.capture_buffer = io.StringIO() if capture_output else None

    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    # Emscripten's runtime parses window.location.search by splitting on '&'
    # and URI-decoding each segment into Module.arguments (argv for main()).
    url = f"http://localhost:{port}/{html_path.name}"
    if args:
        # Emscripten decodes with decodeURI() which preserves RFC-2396
        # reserved chars (%2F, %3D, etc.).  Use a broad safe set so only
        # truly unsafe chars (space, &, #, ?) get encoded.
        query = "&".join(quote(a, safe="/:=@!$'()*+,;-._~") for a in args)
        url += f"?{query}"
    logger.info(f"Serving {serve_dir} at http://localhost:{port}/")

    # Launch browser with isolation flags for process tracking.
    browser_proc: subprocess.Popen | None = None
    temp_profile: str | None = None
    browser_info = _find_browser()

    if browser_info:
        browser_exe, browser_args = browser_info
        browser_name = browser_exe.stem.lower()

        # Chromium: --user-data-dir forces a new instance whose lifetime we
        # can track.  Firefox: -no-remote -profile does the same.
        temp_profile = tempfile.mkdtemp(prefix="ptstudio_browser_")
        if "firefox" in browser_name:
            browser_args = ["-no-remote", "-profile", temp_profile, "-new-window"]
            if headless:
                browser_args.insert(0, "-headless")
        else:
            browser_args = [f"--user-data-dir={temp_profile}"] + browser_args
            if headless:
                browser_args.insert(0, "--headless=new")

        logger.info(f"Opening {url} in {browser_exe.name}")
        try:
            browser_proc = subprocess.Popen(
                [str(browser_exe), *browser_args, url],
            )
        except OSError as e:
            logger.error(f"Failed to launch {browser_exe.name}: {e}")
    else:
        logger.info(f"Opening {url} in default browser (process not tracked)")
        webbrowser.open(url)

    # Wait until browser closes, page calls exit(), or Ctrl+C.
    exit_code = 0
    deadline = (time.monotonic() + timeout) if timeout else None
    try:
        while server_thread.is_alive():
            if browser_proc is not None and browser_proc.poll() is not None:
                logger.info("Browser closed, shutting down server")
                server.shutdown()
                break
            if deadline and time.monotonic() >= deadline:
                logger.warning("Smoke test timed out")
                exit_code = 1
                server.shutdown()
                if browser_proc and browser_proc.poll() is None:
                    browser_proc.terminate()
                break
            time.sleep(0.5)
    except KeyboardInterrupt:
        server.shutdown()
        if browser_proc is not None and browser_proc.poll() is None:
            browser_proc.terminate()
    finally:
        if server_thread.is_alive():
            server.shutdown()
        server_thread.join(timeout=3)
        server.server_close()
        _WasmHandler.server_ref = None
        if temp_profile:
            shutil.rmtree(temp_profile, ignore_errors=True)

    if _WasmHandler.page_exit_code is not None:
        exit_code = _WasmHandler.page_exit_code

    stdout = _WasmHandler.capture_buffer.getvalue() if _WasmHandler.capture_buffer else None
    _WasmHandler.capture_buffer = None

    return subprocess.CompletedProcess(
        args=[str(html_path)], returncode=exit_code, stdout=stdout,
    )


def _run_executable(
    exe_path: Path,
    args: list[str],
    context: dict[str, Any],
    capture_output: bool = False,
) -> subprocess.CompletedProcess:
    """Run an executable inside the appropriate Conan env script.

    For native builds, prefers the conanrun env script (always fresh after a
    build) and falls back to the runtime_deploy directory (flat DLL copy for
    CI test jobs where conan install wasn't run).

    For Emscripten builds, serves via a built-in HTTP server for interactive
    launches, and uses Node.js for headless/captured output (tests).
    """
    build_dir = Path(context["build_dir"])
    is_emscripten = exe_path.suffix.lower() in (".js", ".html")

    # Interactive Emscripten launch — bypass batch wrapping entirely
    if is_emscripten and not capture_output:
        html_path = exe_path.with_suffix(".html") if exe_path.suffix.lower() != ".html" else exe_path
        logger.info(f"Launching {html_path.name} in browser")
        return _serve_emscripten(html_path, args=args)

    # All other paths: use ShellCommand with env_script
    env_script: Path | None = None
    extra_env: dict[str, str] = {}
    if not is_emscripten:
        env_script = _resolve_env_script(build_dir, is_emscripten=False)
        if not env_script:
            runtime_dir = _resolve_runtime_deploy_dir(
                build_dir, context.get("conan_deps_root")
            )
            if runtime_dir:
                logger.debug(f"Using runtime_deploy: {runtime_dir}")
                path_sep = ";" if is_windows() else ":"
                extra_env["PATH"] = f"{runtime_dir}{path_sep}{os.environ.get('PATH', '')}"
    else:
        env_script = _resolve_env_script(build_dir, is_emscripten=True)

    if is_emscripten:
        js_path = exe_path.with_suffix(".js") if exe_path.suffix.lower() == ".html" else exe_path
        logger.info(f"Running {js_path.name} with Node.js")
        cmd = ["node", str(js_path)] + args
    else:
        cmd = [str(exe_path)] + args

    sc = ShellCommand(cmd, env_script=env_script, env=extra_env or None)

    try:
        if capture_output:
            return sc.run(
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace",
            )
        return sc.run()
    except KeyboardInterrupt:
        sys.exit(0)


def _can_run(context: dict[str, Any]) -> bool:
    """Check if executables can be run on this host."""
    build_dir = Path(context["build_dir"])
    if context["platform"] == "emscripten":
        # Interactive launches need emrun (resolved from CMakePresets.json).
        # Headless test runs need Node.js (resolved from conanbuild script or
        # system PATH).
        conan_env = load_conan_env(build_dir, preset_type="configure")
        if conan_env.get("EMSCRIPTEN"):
            return True
        if _resolve_env_script(build_dir, is_emscripten=True) is not None:
            return True
        return shutil.which("node") is not None
    if _resolve_env_script(build_dir, is_emscripten=False) is not None:
        return True
    if _resolve_runtime_deploy_dir(build_dir, context.get("conan_deps_root")) is not None:
        return True
    # Inline check: can only run natively if target matches host
    return context["platform"] == detect_platform_identifier()


def _run_tests(context: dict[str, Any], verbose: bool, from_package: bool = False) -> int:
    """Run all test executables and return exit code."""
    is_emscripten = context["platform"] == "emscripten"
    if from_package:
        # CI path: everything is in the package dir.  Override build_dir
        # so _run_executable finds env scripts in the package too.
        root_dir = Path(context["package_dir"]) / context["build_type"]
        scenes_dir = Path(context["package_dir"]) / "assets" / "scenes"
        context = {**context, "build_dir": str(root_dir), "conan_deps_root": None}
    else:
        # Local dev: build output + source-tree assets.
        root_dir = Path(context["build_dir"])
        scenes_dir = Path(context["workspace_root"]) / "assets" / "scenes"
    bin_dir = root_dir / "bin"
    test_dir = bin_dir / "tests"
    logs_dir = Path(context["logs_root"])
    logs_dir.mkdir(parents=True, exist_ok=True)

    test_executables = _discover_executables(test_dir, is_emscripten)
    if not test_executables:
        logger.error(f"No test executables found in: {test_dir}")
        logger.info("Build the project first: ./repo build")
        return 1

    logger.info(f"Found {len(test_executables)} test executable(s)")

    passed = 0
    failed = 0
    failed_tests = []

    for test_exe in sorted(test_executables):
        test_name = test_exe.stem
        log_file = logs_dir / f"test_{test_name}.log"
        test_args = ["--verbose"] if verbose else []

        with log_section(f"Test: {test_name}"):
            try:
                result = _run_executable(test_exe, test_args, context, capture_output=True)

                with open(log_file, "w", encoding="utf-8", errors="replace") as f:
                    f.write(f"Test: {test_name}\n")
                    f.write(f"Executable: {test_exe}\n")
                    f.write(f"Exit code: {result.returncode}\n")
                    f.write("=" * 70 + "\n")
                    f.write(result.stdout or "")

                if result.stdout:
                    sys.stdout.write(result.stdout)
                    if not result.stdout.endswith("\n"):
                        sys.stdout.write("\n")

                if result.returncode == 0:
                    logger.info(f"PASSED: {test_name}")
                    passed += 1
                else:
                    logger.error(f"FAILED: {test_name} (exit code: {result.returncode})")
                    failed += 1
                    failed_tests.append(test_name)

            except Exception as e:
                logger.error(f"FAILED: {test_name} (exception: {e})")
                with open(log_file, "w", encoding="utf-8", errors="replace") as f:
                    f.write(f"Test: {test_name}\nException: {e}\n")
                failed += 1
                failed_tests.append(test_name)

    # Smoke tests: launch the editor with --capture-and-quit for each built-in
    # demo scene. Exercises the full GPU pipeline (device, shaders, BVH,
    # rendering, readback) with real geometry.
    # On Emscripten: runs headlessly via Node.js; scenes are embedded in MEMFS,
    # so we pass relative paths and only validate exit code (no host-side PNG).
    if is_emscripten:
        editor_exe = bin_dir / "editor.html"
    else:
        editor_exe = bin_dir / ("editor.exe" if is_windows() else "editor")
    scenes: list[Path] = sorted(scenes_dir.glob("*.usdz")) if scenes_dir.is_dir() else []
    if not editor_exe.exists():
        logger.error("FAILED: smoke tests — editor executable not found")
        failed += 1
        failed_tests.append("editorSmoke (missing editor)")
    elif not scenes:
        logger.error(
            "FAILED: smoke tests — no .usdz scene files in "
            f"{scenes_dir}. Run './repo build' to generate them."
        )
        failed += 1
        failed_tests.append("editorSmoke (missing scenes)")
    else:
        with tempfile.TemporaryDirectory(prefix="pts_smoke_") as tmp_dir:
            for scene_path in scenes:
                scene_name = scene_path.stem
                test_name = f"editorSmoke_{scene_name}"
                log_file = logs_dir / f"test_{test_name}.log"
                capture_path = Path(tmp_dir) / f"{scene_name}.png"
                # Emscripten scenes are embedded in MEMFS at their
                # source-relative path; native uses absolute host paths.
                if is_emscripten:
                    usd_arg = f"assets/scenes/{scene_path.name}"
                else:
                    usd_arg = str(scene_path)
                # On Emscripten the capture path must be on MEMFS, not
                # a Windows host path.  /tmp exists in MEMFS by default.
                if is_emscripten:
                    em_capture = f"/tmp/{scene_name}.png"
                    smoke_args = [
                        f"--capture-and-quit={em_capture}",
                        "--frames", "3",
                        "--usd", usd_arg,
                    ]
                else:
                    smoke_args = [
                        f"--capture-and-quit={capture_path}",
                        "--frames", "3",
                        "--usd", usd_arg,
                    ]
                with log_section(f"Test: {test_name}"):
                    try:
                        if is_emscripten:
                            # Browser-based: Node.js lacks navigator.gpu, so
                            # run via _serve_emscripten with output capture,
                            # headless browser, and a timeout.
                            result = _serve_emscripten(
                                editor_exe,
                                args=smoke_args,
                                capture_output=True,
                                timeout=120,
                                headless=True,
                            )
                        else:
                            result = _run_executable(
                                editor_exe,
                                smoke_args,
                                context,
                                capture_output=True,
                            )
                        with open(log_file, "w", encoding="utf-8", errors="replace") as f:
                            f.write(f"Test: {test_name}\n")
                            f.write(f"Scene: {scene_path}\n")
                            f.write(f"Capture: {capture_path}\n")
                            f.write(f"Exit code: {result.returncode}\n")
                            f.write("=" * 70 + "\n")
                            f.write(result.stdout or "")

                        if result.stdout:
                            sys.stdout.write(result.stdout)
                            if not result.stdout.endswith("\n"):
                                sys.stdout.write("\n")

                        if result.returncode != 0:
                            logger.error(
                                f"FAILED: {test_name} (exit code: {result.returncode})"
                            )
                            failed += 1
                            failed_tests.append(test_name)
                        elif not is_emscripten and not capture_path.exists():
                            logger.error(f"FAILED: {test_name} (no capture produced)")
                            failed += 1
                            failed_tests.append(test_name)
                        else:
                            logger.info(f"PASSED: {test_name}")
                            passed += 1

                    except Exception as e:
                        logger.error(f"FAILED: {test_name} (exception: {e})")
                        with open(log_file, "w", encoding="utf-8", errors="replace") as f:
                            f.write(f"Test: {test_name}\nException: {e}\n")
                        failed += 1
                        failed_tests.append(test_name)

    with log_section("Test summary"):
        logger.info(f"Total:  {passed + failed}")
        logger.info(f"Passed: {passed}")
        logger.info(f"Failed: {failed}")

        if failed > 0:
            logger.error("Failed tests:")
            for name in failed_tests:
                logger.error(f"  - {name}")
            return 1
        logger.info("All tests passed!")
    return 0


class LaunchTool(RepoTool):
    name = "launch"
    help = "Launch executables"

    def setup(self, cmd: click.Command) -> click.Command:
        cmd = click.argument(
            "executable",
            required=False,
            default=None,
        )(cmd)
        cmd = click.option(
            "-c", "--config",
            type=click.Choice(
                ["debug", "release", "relwithdebinfo", "minsizerel"],
                case_sensitive=False,
            ),
            default=None,
            help="Build configuration (overrides --build-type)",
        )(cmd)
        cmd = click.option(
            "--env",
            multiple=True,
            help="Environment override (KEY=VALUE). Repeatable.",
        )(cmd)
        cmd = click.option(
            "-i", "--interactive",
            is_flag=True,
            default=None,
            help="Interactive menu to select executable",
        )(cmd)
        cmd = click.option(
            "--profile",
            is_flag=False,
            flag_value="viewer",
            default=None,
            help="Launch Tracy profiler (Windows only). "
            "Bare --profile opens the GUI viewer with auto-connect. "
            "--profile trace.tracy uses headless capture with auto-save.",
        )(cmd)
        return cmd

    def default_args(self, tokens: dict[str, str]) -> dict[str, Any]:
        return {
            "executable": "editor",
            "config": None,
            "env": (),
            "interactive": False,
            "profile": None,
        }

    def execute(self, ctx: ToolContext, args: dict[str, Any]) -> None:
        root = ctx.workspace_root

        # Support both --config (tool-specific) and --build-type (group-level).
        # --config takes precedence when explicitly provided.
        config_val = args.get("config")
        if config_val:
            build_type = to_cmake_build_type(config_val)
        else:
            build_type = ctx.dimensions.get("build_type", "Debug")

        platform_id = ctx.dimensions.get("platform", "")
        is_emscripten = platform_id == "emscripten"

        # Build a minimal context dict from tokens for helper functions
        context: dict[str, Any] = {
            "workspace_root": str(root),
            "build_dir": ctx.tokens["build_dir"],
            "conan_deps_root": ctx.tokens["conan_deps_root"],
            "platform": platform_id,
            "build_type": build_type,
            "logs_root": ctx.tokens["logs_root"],
        }

        build_dir = Path(context["build_dir"])

        # Apply --env overrides to the process environment so they propagate
        # through the shell-wrapped command.
        env_val = args.get("env")
        if env_val:
            for item in env_val:
                text = str(item).strip()
                if "=" in text:
                    key, value = text.split("=", 1)
                    os.environ[key] = value

        # Check if we can run
        if not _can_run(context):
            if is_emscripten:
                logger.error("emsdk not found. Build with --platform emscripten first.")
            else:
                logger.error(f"Cannot run {context['platform']} binaries on this host")
                logger.info(f"Host platform: {detect_platform_identifier()}")
            sys.exit(1)

        # Run single executable
        bin_dir = build_dir / "bin"
        exe_paths = _discover_executables(bin_dir, is_emscripten)

        if not exe_paths:
            logger.error(f"No executables found in: {bin_dir}")
            logger.info("Build the project first: ./repo build")
            sys.exit(1)

        # Interactive mode
        if args.get("interactive"):
            target_exe = _interactive_select(exe_paths)
            if target_exe is None:
                logger.info("No executable selected.")
                sys.exit(0)
        else:
            executable_name = args.get("executable") or "editor"
            target_exe = None
            for exe in exe_paths:
                if exe.stem == executable_name:
                    target_exe = exe
                    break

        if target_exe is None:
            executable_name = args.get("executable") or "editor"
            logger.error(f"Executable not found: {executable_name}")
            logger.info("Available executables:")
            for exe in exe_paths:
                logger.info(f"  {exe.stem}")
            sys.exit(1)

        tracy_proc: subprocess.Popen | None = None
        profile_val = args.get("profile")
        if profile_val is not None:
            if sys.platform != "win32":
                raise RuntimeError(
                    "Tracy profiler pre-built binaries are only available for Windows. "
                    "On Linux/macOS, build from source: "
                    "https://github.com/wolfpld/tracy"
                )
            tracy_dir = _ensure_tracy_viewer(root).parent
            if profile_val == "viewer":
                # GUI viewer with auto-connect
                tracy_exe = tracy_dir / "tracy-profiler.exe"
                tracy_proc = subprocess.Popen([str(tracy_exe), "-a", "127.0.0.1"])
                logger.info("Tracy viewer started (auto-connect to 127.0.0.1)")
            else:
                # Headless capture with auto-save
                tracy_exe = tracy_dir / "tracy-capture.exe"
                out_path = Path(profile_val)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                tracy_proc = subprocess.Popen(
                    [str(tracy_exe), "-a", "127.0.0.1", "-o", str(out_path), "-f"],
                )
                logger.info(f"Tracy capture started → {out_path}")

        try:
            result = _run_executable(target_exe, ctx.passthrough_args, context)
        except KeyboardInterrupt:
            result = subprocess.CompletedProcess(args=[], returncode=0)
        finally:
            if tracy_proc is not None and tracy_proc.poll() is None:
                if profile_val != "viewer":
                    # tracy-capture exits on its own after the app disconnects;
                    # wait for it to flush the trace file.
                    logger.info("Waiting for Tracy capture to finish writing...")
                    try:
                        tracy_proc.wait(timeout=30)
                        logger.info("Tracy capture finished")
                    except subprocess.TimeoutExpired:
                        logger.warning("Tracy capture timed out, terminating")
                        tracy_proc.terminate()
                else:
                    tracy_proc.terminate()

        sys.exit(result.returncode)
