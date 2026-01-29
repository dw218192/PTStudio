"""Configuration helpers for the AI agent tooling."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_ALLOWLIST = [
    "re:^git\\s+.*$",
    "re:^pts\\.cmd\\s+.*$",
    "re:^coderabbit\\s+review\\s+--plain$",
    "re:^gh\\s+pr\\s+(create|review\\s+--approve|merge\\s+--auto\\s+--merge)$",
]


@dataclass
class AgentConfig:
    workspace_root: Path
    state_dir: Path
    allowlist: list[str]
    human_review_mode: str
    max_steps: int
    max_context_files: int
    cursor_command_template: list[str]
    coderabbit_command_template: list[str]
    prompts_root: Path


def _as_list(value: Any, fallback: list[str]) -> list[str]:
    if isinstance(value, list):
        return [str(entry) for entry in value]
    if isinstance(value, str):
        return [value]
    return list(fallback)


def _get_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _normalize_review_mode(value: Any, default: str = "balanced") -> str:
    text = str(value or default).strip().lower()
    if text in {"strict", "balanced", "auto"}:
        return text
    return default


def load_agent_config(
    root: Path, repo_config: dict, human_review_mode: str | None = None
) -> AgentConfig:
    ai_config = repo_config.get("ai_agent", {}) if isinstance(repo_config, dict) else {}

    state_dir_value = ai_config.get("state_dir", "_ai_agent")
    state_dir = Path(state_dir_value)
    if not state_dir.is_absolute():
        state_dir = root / state_dir_value

    allowlist = _as_list(ai_config.get("allowlist"), DEFAULT_ALLOWLIST)
    limits = ai_config.get("limits", {}) if isinstance(ai_config.get("limits"), dict) else {}
    max_steps = _get_int(limits.get("max_steps", 25), 25)
    max_context_files = _get_int(limits.get("max_context_files", 25), 25)

    cursor_cli = ai_config.get("cursor_cli", {}) if isinstance(ai_config.get("cursor_cli"), dict) else {}
    cursor_command_template = _as_list(
        cursor_cli.get(
            "command_template",
            [
                "cursor",
                "agent",
                "--prompt-file",
                "{prompt_file}",
                "--output-file",
                "{output_file}",
            ],
        ),
        [],
    )

    coderabbit = (
        ai_config.get("coderabbit", {}) if isinstance(ai_config.get("coderabbit"), dict) else {}
    )
    coderabbit_command_template = _as_list(
        coderabbit.get("command_template", ["coderabbit", "review", "--plain"]),
        ["coderabbit", "review", "--plain"],
    )

    prompts_root_value = ai_config.get(
        "prompts_root", "tools/repo_tools/ai_agent/prompts"
    )
    prompts_root = Path(prompts_root_value)
    if not prompts_root.is_absolute():
        prompts_root = root / prompts_root_value

    review_mode = _normalize_review_mode(
        human_review_mode if human_review_mode is not None else ai_config.get("human_review_mode"),
        default="balanced",
    )

    return AgentConfig(
        workspace_root=root,
        state_dir=state_dir,
        allowlist=allowlist,
        human_review_mode=review_mode,
        max_steps=max_steps,
        max_context_files=max_context_files,
        cursor_command_template=cursor_command_template,
        coderabbit_command_template=coderabbit_command_template,
        prompts_root=prompts_root,
    )
