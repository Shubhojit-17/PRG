"""FastAPI entrypoint for the Prompt Regression Guardian service."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any

import structlog
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from agent.alerter import SlackAlerter
from agent.evaluator import RegressionEvaluator
from agent.guardian import PromptRegressionGuardian
from agent.mcp_client import PhoenixMCPClient
from agent.models import AgentRunResult
from config.settings import Settings, settings

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


def setup_tracing(active_settings: Settings) -> None:
    """Configure OpenTelemetry tracing for Phoenix and Google GenAI.

    Args:
        active_settings: Settings to read tracing configuration from.
    """

    from phoenix.otel import register
    from openinference.instrumentation.google_genai import GoogleGenAIInstrumentor

    headers: dict[str, str] = {}
    if active_settings.phoenix.api_key:
        headers["api_key"] = active_settings.phoenix.api_key

    register(
        project_name=active_settings.phoenix.project_name,
        endpoint=active_settings.phoenix.host,
        headers=headers or None,
    )
    GoogleGenAIInstrumentor().instrument()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize resources on startup and teardown on shutdown.

    Args:
        app: FastAPI application instance.

    Yields:
        None
    """

    configure_logging()
    logger.info("app_startup", settings=settings.to_redacted_dict())
    setup_tracing(settings)
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

    active_settings = settings
    if payload and payload.project_name:
        active_settings = settings.model_copy(deep=True)
        active_settings.phoenix.project_name = payload.project_name

    async with PhoenixMCPClient(base_url=active_settings.phoenix.host) as client:
        evaluator = RegressionEvaluator(threshold=active_settings.agent.regression_threshold)
        alerter = SlackAlerter(webhook_url=active_settings.slack.webhook_url)
        guardian = PromptRegressionGuardian(client, evaluator, alerter, active_settings)
        result = await guardian.run_check()

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
