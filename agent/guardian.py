"""Core orchestration loop for prompt regression checks."""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone
from typing import Any

import structlog

from agent.alerter import SlackAlerter
from agent.evaluator import RegressionEvaluator
from agent.mcp_client import PhoenixMCPClient
from agent.models import AgentRunResult, RegressionReport
from config.settings import Settings

BASELINE_LABEL = "stable"
VERDICT_LABELS = {
    "pass": "stable",
    "fail": "regression-detected",
    "inconclusive": "needs-review",
}
MAX_CONCURRENT_PROMPTS = 3

logger = structlog.get_logger()


class PromptRegressionGuardian:
    """Coordinates prompt regression checks using Phoenix MCP."""

    def __init__(
        self,
        mcp_client: PhoenixMCPClient,
        evaluator: RegressionEvaluator,
        alerter: SlackAlerter,
        settings: Settings,
    ) -> None:
        """Initialize the guardian.

        Args:
            mcp_client: Phoenix MCP client.
            evaluator: Regression evaluator.
            alerter: Slack alerter.
            settings: Application settings.
        """

        self._mcp_client = mcp_client
        self._evaluator = evaluator
        self._alerter = alerter
        self._settings = settings

    async def run_check(self) -> AgentRunResult:
        """Run regression checks for all prompts.

        Returns:
            AgentRunResult: Result of the run.
        """

        run_id = str(uuid.uuid4())
        started_at = datetime.now(timezone.utc)
        try:
            logger.info(
                "guardian_run_started",
                run_id=run_id,
                project=self._settings.phoenix.project_name,
            )
            prompts = await self._mcp_client.list_prompts()
            datasets = await self._mcp_client.list_datasets(
                self._settings.phoenix.project_name
            )
            dataset_by_name = {dataset.get("name"): dataset for dataset in datasets}
            semaphore = asyncio.Semaphore(MAX_CONCURRENT_PROMPTS)

            async def process_prompt(prompt_meta: dict[str, Any]) -> RegressionReport | None:
                """Process a single prompt for regression analysis.

                Args:
                    prompt_meta: Prompt metadata.

                Returns:
                    RegressionReport | None: The regression report, if created.
                """

                async with semaphore:
                    prompt_name = str(prompt_meta.get("name"))
                    prompt_id = str(prompt_meta.get("id"))
                    candidate_version_id = prompt_meta.get("latest_version_id")
                    baseline_version_id = self._find_baseline_version(prompt_meta)
                    if not baseline_version_id or not candidate_version_id:
                        logger.warning(
                            "baseline_missing",
                            prompt_name=prompt_name,
                            prompt_id=prompt_id,
                        )
                        return None
                    dataset = dataset_by_name.get(prompt_name)
                    if not dataset:
                        logger.warning("dataset_missing", prompt_name=prompt_name)
                        return None

                    baseline_version, candidate_version = await asyncio.gather(
                        self._mcp_client.get_prompt_version(prompt_id, baseline_version_id),
                        self._mcp_client.get_prompt_version(prompt_id, candidate_version_id),
                    )

                    baseline_experiment_id, candidate_experiment_id = await asyncio.gather(
                        self._mcp_client.run_experiment(
                            baseline_version_id,
                            dataset.get("id"),
                            self._settings.agent.eval_judge_model,
                        ),
                        self._mcp_client.run_experiment(
                            candidate_version_id,
                            dataset.get("id"),
                            self._settings.agent.eval_judge_model,
                        ),
                    )

                    baseline_result, candidate_result = await asyncio.gather(
                        self._mcp_client.get_experiment_results(baseline_experiment_id),
                        self._mcp_client.get_experiment_results(candidate_experiment_id),
                    )

                    score_delta, regressed, verdict = self._evaluator.compare(
                        baseline_result, candidate_result
                    )
                    report = RegressionReport(
                        prompt_name=prompt_name,
                        baseline_version=baseline_version,
                        candidate_version=candidate_version,
                        baseline_result=baseline_result,
                        candidate_result=candidate_result,
                        score_delta=score_delta,
                        regressed_dimensions=regressed,
                        verdict=verdict,
                        summary="",
                    )
                    summary = self._evaluator.build_summary(report)
                    report = report.model_copy(update={"summary": summary})

                    label = VERDICT_LABELS.get(verdict, "needs-review")
                    await self._mcp_client.annotate_prompt_version(
                        prompt_id, candidate_version_id, label, summary
                    )

                    if verdict == "fail":
                        await self._alerter.send_regression_alert(report)

                    return report

            tasks = [process_prompt(prompt) for prompt in prompts]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            reports: list[RegressionReport] = []
            for result in results:
                if isinstance(result, RegressionReport):
                    reports.append(result)
                elif isinstance(result, Exception):
                    logger.error("prompt_processing_failed", error=str(result))

            regressions_found = sum(
                1 for report in reports if report.verdict == "fail"
            )
            finished_at = datetime.now(timezone.utc)
            return AgentRunResult(
                run_id=run_id,
                started_at=started_at,
                finished_at=finished_at,
                prompts_checked=len(reports),
                regressions_found=regressions_found,
                reports=reports,
                error=None,
            )
        except Exception as exc:
            logger.exception("guardian_run_failed", run_id=run_id, error=str(exc))
            finished_at = datetime.now(timezone.utc)
            return AgentRunResult(
                run_id=run_id,
                started_at=started_at,
                finished_at=finished_at,
                prompts_checked=0,
                regressions_found=0,
                reports=[],
                error=str(exc),
            )

    def _find_baseline_version(self, prompt_meta: dict[str, Any]) -> str | None:
        """Find the most recent stable baseline version ID.

        Args:
            prompt_meta: Prompt metadata.

        Returns:
            str | None: Baseline version ID if available.
        """

        baseline_version_id = prompt_meta.get("baseline_version_id")
        if isinstance(baseline_version_id, str):
            return baseline_version_id
        versions = prompt_meta.get("versions") or []
        if not isinstance(versions, list):
            return None
        for version in versions:
            labels = set(version.get("labels") or version.get("tags") or [])
            if BASELINE_LABEL in labels:
                candidate = version.get("version_id") or version.get("id")
                if isinstance(candidate, str):
                    return candidate
        return None
