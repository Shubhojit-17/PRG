"""Pydantic models for agent input and output data."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel


class PromptVersion(BaseModel):
    """Represents a specific prompt version."""

    version_id: str
    prompt_text: str
    version_tag: str
    created_at: datetime
    is_baseline: bool = False


class EvalScore(BaseModel):
    """Represents a single evaluation score."""

    dimension: str
    score: float
    explanation: str


class ExperimentResult(BaseModel):
    """Represents an evaluation experiment outcome."""

    experiment_id: str
    prompt_version_id: str
    scores: list[EvalScore]
    mean_score: float
    ran_at: datetime


class RegressionReport(BaseModel):
    """Detailed report of prompt regression analysis."""

    prompt_name: str
    baseline_version: PromptVersion
    candidate_version: PromptVersion
    baseline_result: ExperimentResult
    candidate_result: ExperimentResult
    score_delta: float
    regressed_dimensions: list[str]
    verdict: Literal["pass", "fail", "inconclusive"]
    summary: str


class AgentRunResult(BaseModel):
    """Summary of a full guardian run."""

    run_id: str
    started_at: datetime
    finished_at: datetime
    prompts_checked: int
    regressions_found: int
    reports: list[RegressionReport]
    error: str | None = None
