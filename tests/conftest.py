"""Pytest fixtures for the Prompt Regression Guardian tests."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent.mcp_client import PhoenixMCPClient
from agent.models import EvalScore, ExperimentResult, PromptVersion


@pytest.fixture()
def mock_mcp_client() -> MagicMock:
    """Return a mocked PhoenixMCPClient with async methods.

    Returns:
        MagicMock: Mocked MCP client.
    """

    client = MagicMock(spec=PhoenixMCPClient)
    client.list_prompts = AsyncMock(return_value=[])
    client.list_datasets = AsyncMock(return_value=[])
    client.run_experiment = AsyncMock(return_value="exp-1")
    client.get_experiment_results = AsyncMock()
    client.annotate_prompt_version = AsyncMock(return_value=None)

    sample_version = PromptVersion(
        version_id="v1",
        prompt_text="prompt",
        version_tag="v1.0",
        created_at=datetime.now(timezone.utc),
        is_baseline=True,
    )
    client.get_prompt_version = AsyncMock(return_value=sample_version)
    return client


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
