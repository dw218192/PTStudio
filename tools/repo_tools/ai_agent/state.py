"""State and persistence helpers for the AI agent loop."""

from __future__ import annotations

import json
import os
from pathlib import Path
from datetime import datetime
from typing import Any

from repo_tools import logger
from repo_tools.ai_agent.schema import SessionState, Step


STATE_FILE_NAME = "state.json"
STEP_FILE_NAME = "step.json"
LOCK_FILE_NAME = "lock.json"


def ensure_state_dirs(state_dir: Path) -> dict[str, Path]:
    state_dir.mkdir(parents=True, exist_ok=True)
    context_dir = state_dir / "context"
    human_dir = state_dir / "human"
    logs_dir = state_dir / "logs"
    for path in (context_dir, human_dir, logs_dir):
        path.mkdir(parents=True, exist_ok=True)
    return {"context": context_dir, "human": human_dir, "logs": logs_dir}


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning(f"Failed to read JSON from {path}: {exc}")
        return None


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def load_state(state_dir: Path) -> SessionState | None:
    data = _read_json(state_dir / STATE_FILE_NAME)
    if not isinstance(data, dict):
        return None
    try:
        return SessionState.model_validate(data)
    except Exception as exc:
        logger.warning(f"Failed to parse session state: {exc}")
        return None


def save_state(state_dir: Path, state: SessionState) -> None:
    _write_json(state_dir / STATE_FILE_NAME, state.model_dump())


def load_step(state_dir: Path) -> Step | None:
    data = _read_json(state_dir / STEP_FILE_NAME)
    if not isinstance(data, dict):
        return None
    try:
        return Step.model_validate(data)
    except Exception as exc:
        logger.warning(f"Failed to parse step state: {exc}")
        return None


def save_step(state_dir: Path, step: Step) -> None:
    _write_json(state_dir / STEP_FILE_NAME, step.model_dump())


def mark_corrupt_state(state_dir: Path, label: str) -> None:
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    for name in (STATE_FILE_NAME, STEP_FILE_NAME):
        path = state_dir / name
        if path.exists():
            backup = state_dir / f"{path.stem}_corrupt_{label}_{timestamp}.json"
            try:
                path.rename(backup)
            except OSError:
                logger.warning(f"Failed to archive corrupt state file: {path}")


def acquire_lock(state_dir: Path) -> bool:
    lock_path = state_dir / LOCK_FILE_NAME
    if lock_path.exists():
        logger.warning(f"Agent lock already exists: {lock_path}")
        return False
    payload = {
        "pid": os.getpid(),
        "started_at": datetime.utcnow().isoformat() + "Z",
    }
    _write_json(lock_path, payload)
    return True


def release_lock(state_dir: Path) -> None:
    lock_path = state_dir / LOCK_FILE_NAME
    if lock_path.exists():
        try:
            lock_path.unlink()
        except OSError:
            logger.warning(f"Failed to remove lock file: {lock_path}")


def human_inbox_path(state_dir: Path) -> Path:
    return state_dir / "human" / "inbox.md"


def human_outbox_path(state_dir: Path) -> Path:
    return state_dir / "human" / "outbox.md"


def read_human_outbox(state_dir: Path) -> str:
    path = human_outbox_path(state_dir)
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace").strip()


def clear_human_outbox(state_dir: Path) -> None:
    path = human_outbox_path(state_dir)
    if path.exists():
        path.write_text("", encoding="utf-8")


def write_human_inbox(state_dir: Path, message: str) -> None:
    path = human_inbox_path(state_dir)
    path.write_text(message.strip() + "\n", encoding="utf-8")
