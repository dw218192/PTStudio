"""Project paths and platform-specific configuration for direct task commands.

Ported config helpers retain the MIT notice in tasks/LICENSE.
"""

from __future__ import annotations

import dataclasses
import glob
import os
import string
from pathlib import Path
from typing import Any

import click
import yaml

from .process import detect_platform_identifier, is_windows, to_cmake_build_type

ROOT = Path(__file__).resolve().parents[2]


def project_options(command):
    """The same explicit build selectors on each independent command."""
    command = click.option(
        "--platform",
        type=click.Choice(["windows-x64", "linux-x64", "emscripten"]),
        default=detect_platform_identifier,
        show_default="native host",
    )(command)
    return click.option(
        "--build-type",
        type=click.Choice(
            ["Debug", "Release", "RelWithDebInfo", "MinSizeRel"],
            case_sensitive=False,
        ),
        default="Debug",
        show_default=True,
    )(command)


def load_config(root: Path) -> dict[str, Any]:
    """Read project settings, with optional local overrides."""
    config: dict[str, Any] = {}
    for name in ("config.yaml", "config.local.yaml"):
        path = root / name
        if not path.exists():
            if name == "config.yaml":
                raise FileNotFoundError(path)
            continue
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        if data is not None:
            if not isinstance(data, dict):
                raise TypeError(f"{name} must contain a mapping")
            config = _deep_merge(config, data)
    return config


@dataclasses.dataclass(frozen=True)
class ProjectContext:
    workspace_root: Path
    config: dict[str, Any]
    tokens: dict[str, str]
    dimensions: dict[str, str]
    passthrough_args: list[str] = dataclasses.field(default_factory=list)

    def section(self, name: str) -> dict[str, Any]:
        values = self.config.get(name, {})
        if not isinstance(values, dict):
            raise TypeError(f"{name} must contain a mapping")
        return expand_config(values, self.tokens, self.config)

    def arguments(self, section: str, defaults: dict, overrides: dict) -> dict:
        return {
            **defaults,
            **self.section(section),
            **{key: value for key, value in overrides.items() if value is not None},
        }


def load_context(
    platform: str | None = None,
    build_type: str = "Debug",
    *,
    root: Path = ROOT,
    passthrough: list[str] | None = None,
) -> ProjectContext:
    root = root.resolve()
    dimensions = {
        "platform": platform or detect_platform_identifier(),
        "build_type": to_cmake_build_type(build_type),
    }
    config = resolve_filters(load_config(root), dimensions)
    paths = config.get("paths", {})
    if not isinstance(paths, dict):
        raise TypeError("paths must contain a mapping")
    tokens = {
        **{key: str(value) for key, value in paths.items()},
        "workspace_root": root.as_posix(),
        **dimensions,
        "exe_ext": ".exe" if is_windows() else "",
        "shell_ext": ".bat" if is_windows() else ".sh",
    }
    formatter = TokenFormatter(tokens, config)
    tokens = {key: formatter.resolve(value) for key, value in tokens.items()}
    return ProjectContext(root, config, tokens, dimensions, list(passthrough or []))


class _ConfigProxy:
    """Proxy enabling {cfg:section.key} config cross-references.

    When format_map encounters {cfg:package.output_dir}, Python calls
    _ConfigProxy.__format__("package.output_dir"), which walks
    config["package"]["output_dir"]. Arbitrary nesting is supported:
    {cfg:build.windowing} reads the configured window backend.
    The leaf value must be a string; it may contain token placeholders
    that are resolved by subsequent passes.
    """

    def __init__(self, config: dict[str, Any]):
        self._config = config

    def __format__(self, spec: str) -> str:
        parts = spec.split(".")
        if len(parts) < 2:
            raise KeyError(f"Invalid config reference: cfg:{spec}")
        current: Any = self._config
        for i, part in enumerate(parts):
            if not isinstance(current, dict):
                path = ".".join(parts[:i])
                raise KeyError(f"'{path}' is not a dict in config")
            if part not in current:
                path = ".".join(parts[: i + 1])
                raise KeyError(f"No config key '{path}'")
            current = current[part]
        if not isinstance(current, str):
            raise KeyError(f"'{spec}' is not a string")
        return current


class _EnvProxy:
    """Proxy enabling {env:VAR_NAME} inline environment variable access.

    When format_map encounters {env:UNITY_EDITOR}, Python calls
    _EnvProxy.__format__("UNITY_EDITOR"), which returns
    os.environ["UNITY_EDITOR"].
    """

    def __format__(self, spec: str) -> str:
        if not spec:
            raise KeyError("Empty env var name in {env:...}")
        value = os.environ.get(spec)
        if value is None:
            raise KeyError(f"Environment variable '{spec}' is not set")
        return value


class TokenFormatter(string.Formatter):
    """Format string subclass with circular-reference detection.

    Tokens can reference other tokens: ``{conan_deps_root}`` may expand
    to ``{build_root}/deps``.  This formatter recursively resolves until
    stable, but raises on cycles.
    """

    MAX_DEPTH = 10

    def __init__(self, tokens: dict[str, str], config: dict[str, Any] | None = None) -> None:
        tokens = dict(tokens)  # don't mutate caller's dict
        if config:
            tokens["cfg"] = _ConfigProxy(config)
        tokens["env"] = _EnvProxy()
        self._tokens = tokens

    def resolve(self, template: str) -> str:
        result = template
        for _ in range(self.MAX_DEPTH):
            try:
                expanded = result.format_map(self._tokens)
            except KeyError as exc:
                missing = exc.args[0] if exc.args else "unknown"
                raise KeyError(f"Missing token: {missing}") from exc
            if expanded == result:
                return expanded
            result = expanded
        remaining = _extract_references(result)
        raise ValueError(
            f"Token expansion exceeded {self.MAX_DEPTH} iterations"
            f" (unresolved: {', '.join(sorted(remaining))})"
        )


def _extract_references(template: str) -> set[str]:
    """Return the set of token names referenced by ``{name}`` placeholders.

    Uses ``string.Formatter().parse()`` which correctly ignores escaped
    braces (``{{``/``}}``), returning ``field_name=None`` for those.
    """
    refs: set[str] = set()
    for _, field_name, _, _ in string.Formatter().parse(template):
        if field_name is not None:
            refs.add(field_name)
    return refs


def _deep_merge(base: dict, overlay: dict) -> dict:
    """Recursively merge *overlay* into *base* (overlay wins).

    Dicts are merged recursively.  A key ending in ``+`` whose value is a
    list extends the base list instead of replacing it (e.g. ``paths+: [x]``
    appends to ``paths``).  All other types (including plain lists) are
    replaced wholesale by the overlay value.
    """
    result = dict(base)
    for key, value in overlay.items():
        if key.endswith("+") and isinstance(value, list):
            base_key = key[:-1]
            existing = result.get(base_key, [])
            if isinstance(existing, list):
                result[base_key] = existing + value
            else:
                result[base_key] = value
        elif key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def resolve_filters(config: dict[str, Any], dimension_values: dict[str, str]) -> dict[str, Any]:
    """Walk config dict, resolve ``key@filter`` entries.

    Filter syntax:
    - ``@value`` -- matches any dimension whose current value equals *value*
    - ``@val1,val2`` -- AND across different dimensions
    - ``@!value`` -- negation
    - ``@val1,!val2`` -- compound

    More-specific filters (more conditions) win over less-specific ones.
    """
    # Build reverse lookup: value -> dimension name
    dim_lookup: dict[str, str] = {}
    for dim_name, dim_val in dimension_values.items():
        dim_lookup[dim_val] = dim_name

    return _walk_filters(config, dimension_values, dim_lookup)


def _walk_filters(
    obj: Any,
    dim_values: dict[str, str],
    dim_lookup: dict[str, str],
) -> Any:
    if isinstance(obj, dict):
        # Collect base keys and filtered keys
        base: dict[str, Any] = {}
        filtered: dict[
            str, list[tuple[str, int, Any]]
        ] = {}  # base_key -> [(filter, specificity, value)]

        for key, value in obj.items():
            if "@" in str(key):
                parts = str(key).split("@", 1)
                base_key = parts[0]
                filter_str = parts[1]
                match, specificity = _match_filter(filter_str, dim_values, dim_lookup)
                if match:
                    filtered.setdefault(base_key, []).append((filter_str, specificity, value))
            else:
                base[key] = value

        # Resolve: most-specific filter wins over base
        result: dict[str, Any] = {}
        for key, value in base.items():
            if key in filtered:
                # Pick most specific
                candidates = filtered.pop(key)
                candidates.sort(key=lambda x: x[1], reverse=True)
                result[key] = _walk_filters(candidates[0][2], dim_values, dim_lookup)
            else:
                result[key] = _walk_filters(value, dim_values, dim_lookup)

        # Remaining filtered keys with no base
        for key, candidates in filtered.items():
            candidates.sort(key=lambda x: x[1], reverse=True)
            result[key] = _walk_filters(candidates[0][2], dim_values, dim_lookup)

        return result

    if isinstance(obj, list):
        return [_walk_filters(item, dim_values, dim_lookup) for item in obj]

    return obj


def _match_filter(
    filter_str: str,
    dim_values: dict[str, str],
    dim_lookup: dict[str, str],
) -> tuple[bool, int]:
    """Check if a filter matches the current dimension values.

    Returns ``(matches, specificity)`` where specificity = number of conditions.
    """
    conditions = [c.strip() for c in filter_str.split(",") if c.strip()]
    if not conditions:
        return True, 0

    for cond in conditions:
        negate = cond.startswith("!")
        value = cond.lstrip("!")

        # Find which dimension this value belongs to
        matched_any = False
        for dim_name, dim_val in dim_values.items():
            if value in (dim_val, dim_name):
                matched_any = True
                if negate and value == dim_val:
                    return False, 0  # Negation failed
                break

        # Also check if value is a known dimension value (not current)
        if not matched_any:
            if value in dim_lookup:
                # It's a known dimension value but not the current one
                if negate:
                    pass  # !other_value is true (we don't have that value)
                else:
                    return False, 0  # Wanted a value we don't have
            else:
                # Unknown value -- treat as no match for positive, match for negative
                if not negate:
                    return False, 0

    return True, len(conditions)


def resolve_path(root: Path, template: str, tokens: dict[str, str]) -> Path:
    """Resolve a path template using tokens."""
    formatter = TokenFormatter(tokens)
    resolved = formatter.resolve(template)
    path = Path(resolved)
    if not path.is_absolute():
        path = root / path
    return path


def expand_config(
    values: dict[str, Any],
    tokens: dict[str, str],
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Recursively resolve token references in tool config values.

    Walks *values* and expands ``{token}`` references in string values.
    Nested dicts and lists are recursed into. Unresolvable references
    (e.g. JSON-like braces) are left as-is. Idempotent.
    """
    formatter = TokenFormatter(tokens, config)

    def _resolve_value(v: Any) -> Any:
        if isinstance(v, str) and "{" in v:
            try:
                return formatter.resolve(v)
            except (KeyError, ValueError):
                return v
        if isinstance(v, dict):
            return {k: _resolve_value(val) for k, val in v.items()}
        if isinstance(v, list):
            return [_resolve_value(item) for item in v]
        return v

    return _resolve_value(values)


def glob_paths(pattern: Path | str) -> list[Path]:
    """Expand a glob pattern to a sorted list of matching file paths.

    Returns a single-element list for non-glob paths.
    """
    pattern_text = str(pattern)
    if any(char in pattern_text for char in ("*", "?", "[")):
        return sorted(Path(match) for match in glob.glob(pattern_text, recursive=True))
    return [Path(pattern_text)]
