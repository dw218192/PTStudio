"""Main orchestration loop for the AI agent tooling."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from uuid import uuid4

from repo_tools import logger, print_tool
from repo_tools.ai_agent.agents.base import AgentRunner
from repo_tools.ai_agent.config import AgentConfig
from repo_tools.ai_agent.dev_agent import run_dev_step
from repo_tools.ai_agent.prompts import render_orchestrator_prompt
from pydantic import BaseModel, ValidationError

from repo_tools.ai_agent.schema import (
    MilestonesPlan,
    SessionState,
    Step,
    StepRequest,
    StepResult,
)
from repo_tools.ai_agent.state import (
    acquire_lock,
    clear_human_outbox,
    ensure_state_dirs,
    load_state,
    mark_corrupt_state,
    read_human_outbox,
    release_lock,
    save_state,
    save_step,
    write_human_inbox,
)




def _parse_json_output(output_file: Path, fallback_output: str) -> dict[str, Any]:
    if output_file.exists():
        try:
            payload = json.loads(output_file.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                return payload
        except json.JSONDecodeError as exc:
            logger.warning(f"Failed to parse JSON output {output_file}: {exc}")
    return {"raw_output": fallback_output.strip()}


def _parse_model_output(
    output_file: Path,
    fallback_output: str,
    model_cls: type[BaseModel],
    *,
    default: BaseModel,
) -> BaseModel:
    payload = _parse_json_output(output_file, fallback_output)
    try:
        return model_cls.model_validate(payload)
    except ValidationError as exc:
        logger.warning(f"Failed to validate JSON output {output_file}: {exc}")
        fallback = default.model_copy(deep=True)
        if hasattr(fallback, "raw_output"):
            setattr(fallback, "raw_output", payload.get("raw_output", ""))
        if hasattr(fallback, "needs_human"):
            setattr(fallback, "needs_human", True)
        if hasattr(fallback, "uncertainty"):
            setattr(fallback, "uncertainty", "Invalid JSON output.")
        return fallback




def _should_block(result: StepResult, review_mode: str) -> bool:
    if review_mode == "strict":
        return bool(result.needs_human or result.uncertainty or not result.summary)
    if review_mode == "balanced":
        return bool(result.needs_human)
    return False


def _load_prd_text(prd_path: str | None, max_chars: int = 12000) -> str:
    if not prd_path:
        return ""
    path = Path(prd_path)
    if not path.exists():
        return f"(missing PRD: {prd_path})"
    text = path.read_text(encoding="utf-8", errors="replace")
    return text[:max_chars]


def _derive_goal_from_prd(prd_path: str | None) -> str:
    if not prd_path:
        return ""
    text = _load_prd_text(prd_path, max_chars=2000)
    for line in text.splitlines():
        candidate = line.strip()
        if not candidate:
            continue
        return candidate.lstrip("#").strip()
    return f"Follow PRD at {prd_path}"


def _ensure_goal(state: SessionState, state_dir: Path) -> bool:
    if state.goal:
        return True
    if state.prd_path:
        state.goal = _derive_goal_from_prd(state.prd_path)
        state.status = "active"
        return True
    human_response = read_human_outbox(state_dir)
    if human_response:
        state.goal = human_response
        state.status = "active"
        clear_human_outbox(state_dir)
        return True
    write_human_inbox(
        state_dir,
        "Goal missing. Provide a goal or PRD summary in human/outbox.md and rerun.",
    )
    state.status = "needs_goal"
    return False


def _apply_human_input(state: SessionState, state_dir: Path) -> bool:
    response = read_human_outbox(state_dir)
    if not response:
        return False
    state.pending_question = None
    state.blocked_reason = None
    if not state.goal:
        state.goal = response
    state.status = "active"
    clear_human_outbox(state_dir)
    return True


def _init_state(config: AgentConfig, goal: str, prd_path: str | None) -> SessionState:
    return SessionState(
        session_id=uuid4().hex[:8],
        status="active" if goal else "needs_goal",
        goal=goal,
        prd_path=prd_path,
        milestones=[],
        current_milestone=0,
        max_steps=config.max_steps,
        step_count=0,
        human_review_mode=config.human_review_mode,
    )


def _generate_milestones(
    *,
    agent: AgentRunner,
    config: AgentConfig,
    state: SessionState,
    context_dir: Path,
    output_file: Path,
) -> tuple[list[str], bool, str]:
    prompt = render_orchestrator_prompt(
        prompts_root=config.prompts_root,
        goal=state.goal,
        prd_text=_load_prd_text(state.prd_path),
        state_dir=config.state_dir,
        output_file=output_file,
    )
    run_result = agent.run_step(
        prompt,
        context_dir,
        output_file,
        skill="orchestrator",
    )
    plan = _parse_model_output(
        output_file,
        run_result.stdout + "\n" + run_result.stderr,
        MilestonesPlan,
        default=MilestonesPlan(),
    )
    milestones = [item for item in plan.milestones if item.strip()]
    needs_human = plan.needs_human
    notes = plan.notes
    uncertainty = plan.uncertainty
    if run_result.returncode != 0:
        needs_human = True
        if not notes:
            notes = f"Orchestrator command failed (exit code {run_result.returncode})."
    if uncertainty and config.human_review_mode == "strict":
        needs_human = True
    return milestones, needs_human, notes or uncertainty



def run_agent_loop(
    *,
    agent: AgentRunner,
    config: AgentConfig,
    goal: str,
    prd_path: str | None,
) -> None:
    ensure_state_dirs(config.state_dir)

    if not acquire_lock(config.state_dir):
        print_tool("Another agent session is running. Remove lock to proceed.")
        return

    try:
        state = load_state(config.state_dir)
        if state is None:
            existing_state = (config.state_dir / "state.json").exists()
            if existing_state:
                mark_corrupt_state(config.state_dir, "state")
            state = _init_state(config, goal, prd_path)
            save_state(config.state_dir, state)
        elif goal or prd_path:
            if goal and not state.goal:
                state.goal = goal
                state.status = "active"
            if prd_path and not state.prd_path:
                state.prd_path = prd_path
            save_state(config.state_dir, state)

        state.max_steps = config.max_steps
        state.human_review_mode = config.human_review_mode

        if _apply_human_input(state, config.state_dir):
            save_state(config.state_dir, state)

        if not _ensure_goal(state, config.state_dir):
            save_state(config.state_dir, state)
            return

        if state.status == "blocked":
            write_human_inbox(
                config.state_dir,
                state.pending_question
                or state.blocked_reason
                or "Agent is blocked. Provide guidance in human/outbox.md.",
            )
            return

        while state.step_count < state.max_steps:
            if not state.milestones:
                orchestrator_dir = config.state_dir / "context" / "orchestrator"
                orchestrator_output = config.state_dir / "milestones.json"
                milestones, needs_human, notes = _generate_milestones(
                    agent=agent,
                    config=config,
                    state=state,
                    context_dir=orchestrator_dir,
                    output_file=orchestrator_output,
                )
                if needs_human or not milestones:
                    state.status = "blocked"
                    state.pending_question = notes or "Milestones missing."
                    save_state(config.state_dir, state)
                    write_human_inbox(
                        config.state_dir,
                        state.pending_question or "Milestones missing.",
                    )
                    return
                state.milestones = milestones
                state.current_milestone = 0
                save_state(config.state_dir, state)

            if state.current_milestone >= len(state.milestones):
                state.status = "completed"
                save_state(config.state_dir, state)
                print_tool("All milestones completed.")
                return

            milestone = state.milestones[state.current_milestone]
            step_id = f"{state.step_count + 1:04d}"
            context_dir = config.state_dir / "context" / f"step_{step_id}"
            context_dir.mkdir(parents=True, exist_ok=True)
            context_files = list(state.last_changed_files)

            step_request = StepRequest(
                step_id=step_id,
                goal=milestone,
                context_dir=str(context_dir),
                files=context_files,
            )
            step = Step(request=step_request, result=None)
            save_step(config.state_dir, step)

            output_file = context_dir / "dev_result.json"
            dev_result = run_dev_step(
                agent=agent,
                step=step,
                overall_goal=state.goal,
                milestone=milestone,
                context_dir=context_dir,
                context_files=context_files,
                allowlist=config.allowlist,
                prompts_root=config.prompts_root,
                output_file=output_file,
            )
            step.result = dev_result
            save_step(config.state_dir, step)

            if _should_block(dev_result, state.human_review_mode):
                state.status = "blocked"
                state.blocked_reason = dev_result.uncertainty or "Dev step needs review."
                save_state(config.state_dir, state)
                write_human_inbox(config.state_dir, state.blocked_reason)
                return
            if dev_result.follow_up:
                for item in reversed(dev_result.follow_up):
                    if item.strip():
                        state.milestones.insert(state.current_milestone + 1, item)

            save_step(config.state_dir, step)
            state.last_changed_files = list(dev_result.changed_files)
            state.last_step_id = step_id
            state.last_summary = dev_result.summary
            state.step_count += 1
            state.current_milestone += 1
            save_state(config.state_dir, state)

        state.status = "blocked"
        state.blocked_reason = "Reached max steps limit."
        save_state(config.state_dir, state)
        write_human_inbox(config.state_dir, state.blocked_reason)
    finally:
        release_lock(config.state_dir)
