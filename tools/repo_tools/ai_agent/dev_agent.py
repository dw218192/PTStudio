"""Internal dev step runner."""

from __future__ import annotations

import json
from pathlib import Path

from repo_tools import logger
from repo_tools.ai_agent.agents.base import AgentRunner
from repo_tools.ai_agent.prompts import render_dev_prompt
from repo_tools.ai_agent.schema import Step, StepResult


def _load_result(output_file: Path, fallback_output: str) -> StepResult:
    if output_file.exists():
        try:
            payload = json.loads(output_file.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                return StepResult.model_validate(payload)
        except json.JSONDecodeError as exc:
            logger.warning(f"Failed to parse JSON output {output_file}: {exc}")
    result = StepResult(
        summary="Agent did not produce valid JSON output.",
        needs_human=True,
        uncertainty="Missing or invalid JSON output.",
        raw_output=fallback_output.strip(),
    )
    return result


def run_dev_step(
    *,
    agent: AgentRunner,
    step: Step,
    overall_goal: str,
    milestone: str,
    context_dir: Path,
    context_files: list[str],
    allowlist: list[str],
    prompts_root: Path,
    output_file: Path,
) -> StepResult:
    prompt = render_dev_prompt(
        prompts_root=prompts_root,
        goal=overall_goal,
        milestone=milestone,
        context_dir=context_dir,
        context_files=context_files,
        allowlist=allowlist,
        output_file=output_file,
    )
    run_result = agent.run_step(
        prompt,
        context_dir,
        output_file,
    )
    result = _load_result(output_file, run_result.stdout + "\n" + run_result.stderr)
    if run_result.returncode != 0 and not result.uncertainty:
        result.needs_human = True
        result.uncertainty = f"Agent command failed with exit code {run_result.returncode}."
    return result
