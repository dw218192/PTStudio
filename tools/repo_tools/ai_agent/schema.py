"""Schemas for AI agent state and steps."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


def _coerce_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(entry) for entry in value]
    return [str(value)]


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


class StepRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    step_id: str = ""
    goal: str = ""
    context_dir: str = ""
    files: list[str] = Field(default_factory=list)

    @field_validator("files", mode="before")
    @classmethod
    def _normalize_lists(cls, value: Any) -> list[str]:
        return _coerce_list(value)


class StepResult(BaseModel):
    model_config = ConfigDict(extra="ignore")

    summary: str = ""
    changed_files: list[str] = Field(default_factory=list)
    needs_human: bool = False
    uncertainty: str = ""
    follow_up: list[str] = Field(default_factory=list)
    raw_output: str = ""

    @field_validator("changed_files", "follow_up", mode="before")
    @classmethod
    def _normalize_lists(cls, value: Any) -> list[str]:
        return _coerce_list(value)

    @field_validator("needs_human", mode="before")
    @classmethod
    def _normalize_bool(cls, value: Any) -> bool:
        return _coerce_bool(value)


class Step(BaseModel):
    model_config = ConfigDict(extra="ignore")

    request: StepRequest
    result: StepResult | None = None


class SessionState(BaseModel):
    model_config = ConfigDict(extra="ignore")

    session_id: str
    status: str
    goal: str
    prd_path: str | None
    milestones: list[str]
    current_milestone: int
    max_steps: int
    step_count: int
    human_review_mode: str
    blocked_reason: str | None = None
    pending_question: str | None = None
    last_step_id: str | None = None
    last_summary: str | None = None
    last_changed_files: list[str] = Field(default_factory=list)

    @field_validator("milestones", "last_changed_files", mode="before")
    @classmethod
    def _normalize_lists(cls, value: Any) -> list[str]:
        return _coerce_list(value)



class MilestonesPlan(BaseModel):
    model_config = ConfigDict(extra="ignore")

    milestones: list[str] = Field(default_factory=list)
    notes: str = ""
    needs_human: bool = False
    uncertainty: str = ""

    @field_validator("milestones", mode="before")
    @classmethod
    def _normalize_lists(cls, value: Any) -> list[str]:
        return _coerce_list(value)

    @field_validator("needs_human", mode="before")
    @classmethod
    def _normalize_bool(cls, value: Any) -> bool:
        return _coerce_bool(value)
