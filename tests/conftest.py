"""Pytest fixtures for the Prompt Regression Guardian tests."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from agent.models import EvalScore, ExperimentResult


@pytest.fixture()
def sample_experiment_result() -> ExperimentResult:
    """Provide a sample experiment result fixture.

    Returns:
        ExperimentResult: Sample experiment result.
    """

    return ExperimentResult(
        experiment_id="exp-123",
        prompt_version_id="version-1",
        scores=[
            EvalScore(dimension="relevance", score=0.85, explanation="ok"),
            EvalScore(dimension="faithfulness", score=0.80, explanation="ok"),
            EvalScore(dimension="toxicity", score=0.95, explanation="ok"),
        ],
        mean_score=0.867,
        ran_at=datetime.now(timezone.utc),
    )


@pytest.fixture()
def settings_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """Override environment variables for settings tests.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
    """

    monkeypatch.setenv("REGRESSION_THRESHOLD", "0.05")
    monkeypatch.setenv("PHOENIX_HOST", "http://localhost:6006")
