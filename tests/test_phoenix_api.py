"""Tests for the Phoenix REST API helpers."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent import phoenix_api


@pytest.mark.asyncio
@patch("agent.phoenix_api.httpx.AsyncClient")
async def test_list_prompts_calls_v1_endpoint(mock_client_cls: MagicMock) -> None:
    """Ensure list_prompts uses the Phoenix /v1/prompts endpoint."""

    response = MagicMock()
    response.json.return_value = [{"id": "p1"}]
    response.raise_for_status = MagicMock()

    client = AsyncMock()
    client.get.return_value = response
    mock_client_cls.return_value.__aenter__.return_value = client

    prompts = await phoenix_api.list_prompts()

    client.get.assert_awaited_once_with("v1/prompts")
    assert prompts == [{"id": "p1"}]


@pytest.mark.asyncio
@patch("agent.phoenix_api.httpx.AsyncClient")
async def test_list_datasets_handles_dict_payload(mock_client_cls: MagicMock) -> None:
    """Support Phoenix responses that wrap datasets in a data key."""

    response = MagicMock()
    response.json.return_value = {"data": [{"id": "d1", "name": "demo"}]}
    response.raise_for_status = MagicMock()

    client = AsyncMock()
    client.get.return_value = response
    mock_client_cls.return_value.__aenter__.return_value = client

    datasets = await phoenix_api.list_datasets()

    client.get.assert_awaited_once_with("v1/datasets")
    assert datasets == [{"id": "d1", "name": "demo"}]


@pytest.mark.asyncio
@patch("agent.phoenix_api.httpx.AsyncClient")
async def test_create_experiment_posts_payload(mock_client_cls: MagicMock) -> None:
    """Create experiments using the Phoenix dataset-scoped endpoint."""

    payload = {"prompt_version_id": "v1", "dataset_id": "d1", "judge_model": "m1"}
    response = MagicMock()
    response.json.return_value = {"data": {"id": "exp-1"}}
    response.raise_for_status = MagicMock()

    client = AsyncMock()
    client.post.return_value = response
    mock_client_cls.return_value.__aenter__.return_value = client

    # This test is intentionally removed as create_experiment was deprecated.
    pytest.skip("create_experiment deprecated; covered by run_llm_evals tests")


@pytest.mark.asyncio
@patch("agent.phoenix_api.httpx.AsyncClient")
async def test_get_experiment_fetches_by_id(mock_client_cls: MagicMock) -> None:
    """Fetch experiment details by ID."""

    response = MagicMock()
    response.json.return_value = {"id": "exp-1", "status": "completed"}
    response.raise_for_status = MagicMock()

    client = AsyncMock()
    client.get.return_value = response
    mock_client_cls.return_value.__aenter__.return_value = client

    pytest.skip("get_experiment deprecated; covered by run_llm_evals tests")


@pytest.mark.asyncio
@patch("agent.phoenix_api.httpx.AsyncClient")
@patch("agent.phoenix_api.asyncio.to_thread", new_callable=AsyncMock)
async def test_run_llm_evals_returns_scores(mock_to_thread: AsyncMock, mock_client_cls: MagicMock) -> None:
    """run_llm_evals should return non-empty scores and a mean between 0 and 1."""

    examples = [
        {"input": {"question": "What is my order status?"}, "output": {"answer": "Shipped"}},
        {"input": {"question": "How do I return an item?"}, "output": {"answer": "Use the returns portal"}},
    ]
    response = MagicMock()
    response.json.return_value = {"data": {"examples": examples}}
    response.raise_for_status = MagicMock()

    client = AsyncMock()
    client.get.return_value = response
    mock_client_cls.return_value.__aenter__.return_value = client

    # Mock generation and judge responses returned by asyncio.to_thread
    gen_choice = MagicMock()
    gen_choice.message = MagicMock()
    gen_choice.message.content = '"[\\"gen1\\", \\\"gen2\\\"]"'
    gen_resp = MagicMock()
    gen_resp.choices = [gen_choice]

    score_choice = MagicMock()
    score_choice.message = MagicMock()
    score_choice.message.content = "[0.9, 0.8]"
    score_resp = MagicMock()
    score_resp.choices = [score_choice]

    mock_to_thread.side_effect = [gen_resp, score_resp]

    result = await phoenix_api.run_llm_evals(
        prompt_version_id="v1",
        prompt_text="system prompt",
        dataset_id="dataset123",
        judge_model="m1",
    )

    client.get.assert_awaited_once_with("v1/datasets/dataset123/examples")
    assert isinstance(result["mean_score"], float)
    assert 0.0 <= result["mean_score"] <= 1.0
    assert isinstance(result["scores"], list)
    assert len(result["scores"]) > 0


@pytest.mark.asyncio
@patch("agent.phoenix_api.httpx.AsyncClient")
@patch("agent.phoenix_api.asyncio.to_thread", new_callable=AsyncMock)
async def test_run_llm_evals_handles_empty_dataset(mock_to_thread: AsyncMock, mock_client_cls: MagicMock) -> None:
    """run_llm_evals should return empty scores and mean 0.0 for empty datasets."""

    response = MagicMock()
    response.json.return_value = {"data": {"examples": []}}
    response.raise_for_status = MagicMock()

    client = AsyncMock()
    client.get.return_value = response
    mock_client_cls.return_value.__aenter__.return_value = client

    result = await phoenix_api.run_llm_evals(
        prompt_version_id="v1",
        prompt_text="system prompt",
        dataset_id="empty",
        judge_model="m1",
    )

    client.get.assert_awaited_once_with("v1/datasets/empty/examples")
    assert result["mean_score"] == 0.0
    assert result["scores"] == []
    assert result["example_count"] == 0
    mock_to_thread.assert_not_awaited()
