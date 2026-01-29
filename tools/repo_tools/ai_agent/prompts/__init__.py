"""Prompt templates for the AI agent tool."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

from repo_tools import replace_tokens


def _format_list(items: list[str]) -> str:
    if not items:
        return "(none)"
    return "\n".join(f"- {item}" for item in items)


def _load_template(prompts_root: Path, name: str) -> str:
    prompt_file = prompts_root / f"{name}.md"
    return prompt_file.read_text(encoding="utf-8", errors="replace")


def render_prompt(
    prompts_root: Path, name: str, replacements: Mapping[str, str]
) -> str:
    template = _load_template(prompts_root, name)
    rendered = replace_tokens(template, replacements)
    return rendered.rstrip() + "\n"


def render_orchestrator_prompt(
    *,
    prompts_root: Path,
    goal: str,
    prd_text: str,
    state_dir: Path,
    output_file: Path,
) -> str:
    return render_prompt(
        prompts_root,
        "orchestrator",
        {
            "goal": goal,
            "prd_text": prd_text if prd_text else "(none)",
            "state_dir": str(state_dir),
            "output_file": str(output_file),
        },
    )


def render_dev_prompt(
    *,
    prompts_root: Path,
    goal: str,
    milestone: str,
    context_dir: Path,
    context_files: list[str],
    allowlist: list[str],
    output_file: Path,
) -> str:
    return render_prompt(
        prompts_root,
        "dev",
        {
            "goal": goal,
            "milestone": milestone,
            "context_dir": str(context_dir),
            "context_files": _format_list(context_files),
            "allowlist": _format_list(allowlist),
            "output_file": str(output_file),
        },
    )
