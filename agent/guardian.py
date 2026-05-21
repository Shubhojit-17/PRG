"""Google ADK agent for prompt regression checks."""

from __future__ import annotations

from datetime import datetime, timezone
import os
from typing import Any

import httpx
import structlog
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from pydantic import ValidationError

from agent.alerter import SlackAlerter
from agent.evaluator import RegressionEvaluator
from agent.models import EvalScore, ExperimentResult, PromptVersion, RegressionReport
from agent.phoenix_api import (
    fetch_prompt_versions,
    list_datasets,
    list_prompts as phoenix_list_prompts,
    run_llm_evals,
)
from config.settings import settings

SYSTEM_PROMPT = """
You are the Prompt Regression Guardian agent. You MUST follow these steps in exact order using the tools available to you:

STEP 1: Call `fetch_latest_prompt_version` immediately with no arguments. This returns all prompts registered in Phoenix with their baseline and candidate versions.
STEP 2: For each prompt returned, call `fetch_eval_dataset` using the prompt name to get the evaluation dataset.
STEP 3: Call `evaluate_prompt_version` with the `baseline_version` object returned by `fetch_latest_prompt_version`.
STEP 4: Call `evaluate_prompt_version` with the `candidate_version` object returned by `fetch_latest_prompt_version`.
STEP 5: Compare the two mean scores — if candidate drops more than 0.05, it's a regression.
STEP 6: The application handles alerts after the run finishes, so do not call `send_regression_alert`.

You must call the tools - do not reason about what the tools would return without calling them.
""".strip()

BASELINE_LABEL = "stable"
VERDICT_LABELS = {
    "pass": "stable",
    "fail": "regression-detected",
    "inconclusive": "needs-review",
}

logger = structlog.get_logger()


def _build_model() -> Any:
    use_groq = (
        os.environ.get("USE_GROQ", str(settings.google_cloud.use_groq)).lower()
        == "true"
    )
    if use_groq:
        groq_key = os.environ.get("GROQ_API_KEY", settings.google_cloud.groq_api_key)
        if groq_key:
            os.environ["GROQ_API_KEY"] = groq_key
        return LiteLlm(model="groq/llama-3.3-70b-versatile")
    return settings.google_cloud.model


def _ensure_google_api_key() -> None:
    if settings.google_cloud.google_api_key:
        os.environ["GOOGLE_API_KEY"] = settings.google_cloud.google_api_key


def _coerce_datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            logger.warning("phoenix_datetime_parse_failed", value=value)
    return datetime.now(timezone.utc)


def _find_baseline_version(
    prompt_meta: dict[str, Any],
    versions: list[dict[str, Any]] | None = None,
) -> str | None:
    baseline_version_id = prompt_meta.get("baseline_version_id")
    if isinstance(baseline_version_id, str):
        return baseline_version_id
    if versions is None:
        versions = prompt_meta.get("versions") or []
    if not isinstance(versions, list):
        return None
    for version in versions:
        labels = set(version.get("labels") or version.get("tags") or [])
        if BASELINE_LABEL in labels:
            candidate = version.get("version_id") or version.get("id")
            if isinstance(candidate, str):
                return candidate
        description = str(version.get("description") or "").lower()
        if "baseline" in description:
            candidate = version.get("version_id") or version.get("id")
            if isinstance(candidate, str):
                return candidate
    return None


def _find_candidate_version(
    versions: list[dict[str, Any]],
    baseline_version_id: str | None,
) -> str | None:
    for version in versions:
        description = str(version.get("description") or "").lower()
        if "candidate" in description:
            candidate = version.get("version_id") or version.get("id")
            if isinstance(candidate, str):
                return candidate
    for version in versions:
        candidate = version.get("version_id") or version.get("id")
        if isinstance(candidate, str) and candidate != baseline_version_id:
            return candidate
    return None


def _select_version(
    versions: list[dict[str, Any]], version_id: str | None, is_baseline: bool
) -> PromptVersion:
    fallback_id = version_id or "unknown"
    fallback_tag = str(version_id or "latest")
    selected: dict[str, Any] | None = None
    for version in versions:
        candidate = version.get("version_id") or version.get("id")
        if candidate == version_id:
            selected = version
            break

    if selected is None:
        return PromptVersion(
            version_id=fallback_id,
            prompt_text="",
            version_tag=fallback_tag,
            created_at=datetime.now(timezone.utc),
            is_baseline=is_baseline,
        )

    return PromptVersion(
        version_id=str(selected.get("version_id") or selected.get("id") or fallback_id),
        prompt_text=_extract_prompt_text(selected),
        version_tag=str(selected.get("version_tag") or selected.get("tag") or fallback_tag),
        created_at=_coerce_datetime(
            selected.get("created_at")
            or selected.get("createdAt")
            or selected.get("inserted_at")
        ),
        is_baseline=is_baseline,
    )


def _extract_prompt_text(version: dict[str, Any]) -> str:
    template = version.get("template")
    if isinstance(template, dict):
        messages = template.get("messages")
        if isinstance(messages, list):
            for message in messages:
                if not isinstance(message, dict):
                    continue
                if message.get("role") == "system":
                    content = message.get("content")
                    if isinstance(content, str) and content.strip():
                        return content.strip()
            for message in messages:
                if not isinstance(message, dict):
                    continue
                content = message.get("content")
                if isinstance(content, str) and content.strip():
                    return content.strip()
    for key in ("prompt_text", "text", "description"):
        value = version.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


async def _annotate_prompt_version(
    prompt_id: str, version_id: str, label: str, note: str
) -> None:
    payload = {"label": label, "note": note}
    endpoint = f"/v1/prompts/{prompt_id}/versions/{version_id}/labels"
    async with httpx.AsyncClient(
        base_url=settings.phoenix.host,
        headers={"Content-Type": "application/json"},
        timeout=httpx.Timeout(10),
    ) as client:
        if settings.phoenix.api_key:
            client.headers["Authorization"] = f"Bearer {settings.phoenix.api_key}"
        try:
            response = await client.post(endpoint, json=payload)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            logger.warning(
                "phoenix_annotation_failed",
                prompt_id=prompt_id,
                version_id=version_id,
                error=str(exc),
            )


async def fetch_latest_prompt_version(
    project_name: str | None = None,
) -> list[dict[str, Any]]:
    """Fetch latest prompt versions and baseline metadata."""

    prompts = await phoenix_list_prompts()
    if project_name:
        target = project_name.lower()
        filtered: list[dict[str, Any]] = []
        for prompt in prompts:
            prompt_project = str(
                prompt.get("project_name") or prompt.get("project") or ""
            ).lower()
            if prompt_project:
                if prompt_project == target:
                    filtered.append(prompt)
            else:
                filtered.append(prompt)
        prompts = filtered
    results: list[dict[str, Any]] = []
    for prompt in prompts:
        prompt_id = str(prompt.get("id") or "")
        prompt_name = str(prompt.get("name") or "")
        versions = await fetch_prompt_versions(prompt_id)
        if not isinstance(versions, list):
            versions = []
        baseline_version_id = _find_baseline_version(prompt, versions)
        candidate_version_id = prompt.get("latest_version_id") or prompt.get(
            "latestVersionId"
        )
        if not isinstance(candidate_version_id, str) or not candidate_version_id:
            candidate_version_id = _find_candidate_version(
                versions,
                baseline_version_id,
            )
        baseline_version = _select_version(versions, baseline_version_id, True)
        candidate_version = _select_version(versions, candidate_version_id, False)
        results.append(
            {
                "prompt_id": prompt_id,
                "prompt_name": prompt_name,
                "baseline_version": baseline_version.model_dump(mode="json"),
                "candidate_version": candidate_version.model_dump(mode="json"),
                "baseline_version_id": baseline_version.version_id,
                "candidate_version_id": candidate_version.version_id,
            }
        )
    return results


async def fetch_eval_dataset(prompt_name: str) -> dict[str, Any]:
    """Fetch the evaluation dataset for a prompt name."""

    datasets = await list_datasets()
    for dataset in datasets:
        if str(dataset.get("name") or "").lower() == prompt_name.lower():
            return dataset
    return {}


async def evaluate_prompt_version(
    prompt_name: str,
    prompt_version: dict[str, Any],
    dataset_id: str,
) -> dict[str, Any]:
    """Run Phoenix-style evals for a prompt version against a dataset."""

    prompt_version_id = str(
        prompt_version.get("version_id") or prompt_version.get("id") or "unknown"
    )
    prompt_text = str(prompt_version.get("prompt_text") or "")

    use_groq = (
        os.environ.get("USE_GROQ", str(settings.google_cloud.use_groq)).lower()
        == "true"
    )
    judge_model = (
        "groq/llama-3.3-70b-versatile"
        if use_groq
        else settings.agent.eval_judge_model
    )

    result = await run_llm_evals(
        prompt_version_id=prompt_version_id,
        prompt_text=prompt_text,
        dataset_id=dataset_id,
        judge_model=judge_model,
    )
    return {
        "prompt_name": prompt_name,
        "version_id": prompt_version_id,
        "prompt_text": prompt_text,
        "dataset_id": dataset_id,
        "is_baseline": bool(prompt_version.get("is_baseline")),
        **result,
    }


async def send_regression_alert(report: dict[str, Any]) -> dict[str, Any]:
    """Send a Slack alert for a regression report."""

    try:
        regression_report = RegressionReport.model_validate(report)
    except ValidationError:
        logger.warning("regression_alert_skipped_invalid_report", report=report)
        return {"status": "skipped", "reason": "invalid_report"}
    if regression_report.verdict != "fail":
        return {"status": "skipped", "reason": "verdict_not_fail"}
    alerter = SlackAlerter(webhook_url=settings.slack.webhook_url)
    await alerter.send_regression_alert(regression_report)
    return {"status": "sent"}


_ensure_google_api_key()

root_agent = Agent(
    name="prompt_regression_guardian",
    model=_build_model(),
    instruction=SYSTEM_PROMPT,
    tools=[
        fetch_latest_prompt_version,
        fetch_eval_dataset,
        evaluate_prompt_version,
        send_regression_alert,
    ],
)
