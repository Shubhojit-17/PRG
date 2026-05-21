"""FastAPI entrypoint for the Prompt Regression Guardian service."""

from __future__ import annotations

import json
import os
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any

import structlog
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ValidationError

from agent.evaluator import RegressionEvaluator
from agent.guardian import evaluate_prompt_version, fetch_eval_dataset, fetch_latest_prompt_version
from agent.instrumentation import setup_tracing
from agent.models import AgentRunResult, EvalScore, ExperimentResult, PromptVersion, RegressionReport
from config.settings import settings

ENV_DEVELOPMENT = "development"
ENV_PRODUCTION = "production"

logger = structlog.get_logger()

LAST_RUN_CACHE: dict[str, AgentRunResult] = {}


class RunRequest(BaseModel):
    """Request payload for triggering a guardian run."""

    project_name: str | None = None


def configure_logging() -> None:
    """Configure structlog for JSON or pretty output."""

    environment = os.getenv("ENV", ENV_DEVELOPMENT).lower()
    processors: list[Any] = [
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]
    if environment == ENV_PRODUCTION:
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(20),
        cache_logger_on_first_use=True,
    )


def _extract_reports(raw_output: Any) -> list[RegressionReport]:
    if raw_output is None:
        return []
    payload: Any = raw_output
    if isinstance(raw_output, str):
        try:
            payload = json.loads(raw_output)
        except json.JSONDecodeError:
            return []
    if isinstance(payload, dict):
        payload = payload.get("reports") or payload.get("results") or []
    if not isinstance(payload, list):
        return []
    reports: list[RegressionReport] = []
    for item in payload:
        try:
            reports.append(RegressionReport.model_validate(item))
        except ValidationError:
            logger.warning("runner_report_parse_failed")
    return reports


def _build_prompt_version(payload: dict[str, Any], is_baseline: bool) -> Any:
    try:
        return PromptVersion.model_validate(payload)
    except ValidationError:
        return PromptVersion(
            version_id=str(payload.get("version_id") or "unknown"),
            prompt_text=str(payload.get("prompt_text") or ""),
            version_tag=str(payload.get("version_tag") or payload.get("version_id") or "latest"),
            created_at=datetime.now(timezone.utc),
            is_baseline=is_baseline,
        )


def _build_eval_result(payload: dict[str, Any]) -> ExperimentResult:
    score_entries = payload.get("scores") or []
    scores: list[EvalScore] = []
    if isinstance(score_entries, list):
        for item in score_entries:
            try:
                scores.append(
                    EvalScore(
                        dimension="relevance",
                        score=float(item),
                        explanation="",
                    )
                )
            except (TypeError, ValueError):
                continue
    mean_score = float(payload.get("mean_score") or 0)
    version_id = str(payload.get("version_id") or payload.get("prompt_version_id") or "unknown")
    return ExperimentResult(
        experiment_id=version_id,
        prompt_version_id=version_id,
        scores=scores,
        mean_score=mean_score,
        ran_at=datetime.now(timezone.utc),
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize resources on startup and teardown on shutdown.

    Args:
        app: FastAPI application instance.

    Yields:
        None
    """

    configure_logging()
    use_groq = (
        os.environ.get("USE_GROQ", str(settings.google_cloud.use_groq)).lower()
        == "true"
    )
    logger.info(
        "model_selected",
        provider="groq/llama-3.3-70b-versatile" if use_groq else "gemini",
        use_groq=use_groq,
    )
    logger.info("app_startup", settings=settings.to_redacted_dict())
    logger.info(
        "phoenix_env_loaded",
        host=settings.phoenix.host,
        api_key="***" if settings.phoenix.api_key else "",
    )
    setup_tracing()
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    """Health check endpoint.

    Returns:
        dict[str, str]: Health status and timestamp.
    """

    return {"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat()}


@app.post("/run")
async def run_guardian(payload: RunRequest | None = None) -> AgentRunResult:
    """Trigger a full guardian run and return the result.

    Args:
        payload: Optional request payload containing project overrides.

    Returns:
        AgentRunResult: Guardian run output.
    """

    project_name = (
        payload.project_name
        if payload and payload.project_name
        else settings.phoenix.project_name
    )
    goal = "Check all prompts for regressions and alert if any are found."

    run_id = str(uuid.uuid4())
    started_at = datetime.now(timezone.utc)
    try:
        prompt_versions = await fetch_latest_prompt_version(project_name)
        evaluator = RegressionEvaluator(threshold=settings.agent.regression_threshold)
        reports: list[RegressionReport] = []
        for prompt_bundle in prompt_versions:
            if not isinstance(prompt_bundle, dict):
                continue

            prompt_name = str(prompt_bundle.get("prompt_name") or "")
            baseline_meta = prompt_bundle.get("baseline_version") or {}
            candidate_meta = prompt_bundle.get("candidate_version") or {}
            if not prompt_name or not isinstance(baseline_meta, dict) or not isinstance(candidate_meta, dict):
                continue

            dataset = await fetch_eval_dataset(prompt_name)
            dataset_id = str(dataset.get("id") or dataset.get("dataset_id") or "")
            if not dataset_id:
                logger.warning("dataset_missing_for_prompt", prompt_name=prompt_name)
                continue

            baseline_eval = await evaluate_prompt_version(
                prompt_name=prompt_name,
                prompt_version=baseline_meta,
                dataset_id=dataset_id,
            )
            candidate_eval = await evaluate_prompt_version(
                prompt_name=prompt_name,
                prompt_version=candidate_meta,
                dataset_id=dataset_id,
            )

            baseline_version = _build_prompt_version(baseline_meta, True)
            candidate_version = _build_prompt_version(candidate_meta, False)
            baseline_result = _build_eval_result(baseline_eval)
            candidate_result = _build_eval_result(candidate_eval)
            score_delta, regressed_dimensions, verdict = evaluator.compare(
                baseline_result,
                candidate_result,
            )
            summary = evaluator.build_summary(
                RegressionReport(
                    prompt_name=prompt_name,
                    baseline_version=baseline_version,
                    candidate_version=candidate_version,
                    baseline_result=baseline_result,
                    candidate_result=candidate_result,
                    score_delta=score_delta,
                    regressed_dimensions=regressed_dimensions,
                    verdict=verdict,
                    summary="",
                )
            )
            reports.append(
                RegressionReport(
                    prompt_name=prompt_name,
                    baseline_version=baseline_version,
                    candidate_version=candidate_version,
                    baseline_result=baseline_result,
                    candidate_result=candidate_result,
                    score_delta=score_delta,
                    regressed_dimensions=regressed_dimensions,
                    verdict=verdict,
                    summary=summary,
                )
            )

        prompts_checked = len(prompt_versions)
        regressions_found = sum(1 for report in reports if report.verdict == "fail")
        result = AgentRunResult(
            run_id=run_id,
            started_at=started_at,
            finished_at=datetime.now(timezone.utc),
            prompts_checked=prompts_checked,
            regressions_found=regressions_found,
            reports=reports,
            error=None,
        )
    except Exception as exc:
        logger.exception("guardian_run_failed", run_id=run_id, error=str(exc))
        result = AgentRunResult(
            run_id=run_id,
            started_at=started_at,
            finished_at=datetime.now(timezone.utc),
            prompts_checked=0,
            regressions_found=0,
            reports=[],
            error=str(exc),
        )

    LAST_RUN_CACHE["last"] = result
    return result


@app.get("/status")
async def status() -> AgentRunResult:
    """Return the most recent guardian run result.

    Returns:
        AgentRunResult: Cached guardian run result.

    Raises:
        HTTPException: If no run has completed yet.
    """

    cached = LAST_RUN_CACHE.get("last")
    if cached is None:
        raise HTTPException(status_code=404, detail="No runs completed yet.")
    return cached


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("agent.main:app", host="0.0.0.0", port=8080, workers=1)
