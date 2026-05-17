"""Slack alerting utilities for regression reports."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import httpx
import structlog

from agent.models import RegressionReport
from config.settings import settings

SLACK_HEADER_FAIL = "🚨 Prompt regression detected"
SLACK_HEADER_PASS = "✅ Prompt check passed"
SLACK_TEXT_FALLBACK = "Prompt regression check result"
SLACK_TIMEOUT_SECONDS = 10

logger = structlog.get_logger()


@dataclass
class AlertDeliveryError(Exception):
    """Raised when an alert fails to deliver."""

    message: str


class SlackAlerter:
    """Sends regression alerts to Slack via incoming webhooks."""

    def __init__(self, webhook_url: str) -> None:
        """Initialize the Slack alerter.

        Args:
            webhook_url: Slack incoming webhook URL.
        """

        self._webhook_url = webhook_url

    async def send_regression_alert(self, report: RegressionReport) -> None:
        """Send a regression alert to Slack.

        Args:
            report: Regression report to send.

        Raises:
            AlertDeliveryError: If Slack returns a non-2xx response.
        """

        header_text = SLACK_HEADER_FAIL if report.verdict == "fail" else SLACK_HEADER_PASS
        delta_str = f"{report.score_delta:+.2f}"
        delta_display = f":red_circle: {delta_str}" if report.score_delta < 0 else delta_str
        dimensions = ", ".join(report.regressed_dimensions) or "none"
        timestamp = datetime.now(timezone.utc).isoformat()

        blocks: list[dict[str, Any]] = [
            {"type": "header", "text": {"type": "plain_text", "text": header_text}},
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": (
                        f"*{report.prompt_name}* — {report.candidate_version.version_tag} "
                        f"({report.verdict})"
                    ),
                },
            },
            {
                "type": "section",
                "fields": [
                    {
                        "type": "mrkdwn",
                        "text": f"*Baseline score*\n{report.baseline_result.mean_score:.2f}",
                    },
                    {
                        "type": "mrkdwn",
                        "text": (
                            f"*Candidate score*\n{report.candidate_result.mean_score:.2f}"
                        ),
                    },
                    {"type": "mrkdwn", "text": f"*Score delta*\n{delta_display}"},
                    {"type": "mrkdwn", "text": f"*Dimensions affected*\n{dimensions}"},
                ],
            },
            {"type": "section", "text": {"type": "mrkdwn", "text": report.summary}},
            {
                "type": "context",
                "elements": [
                    {
                        "type": "mrkdwn",
                        "text": (
                            f"Triggered at {timestamp} | Project: "
                            f"{settings.phoenix.project_name}"
                        ),
                    }
                ],
            },
        ]

        if report.verdict == "fail":
            blocks.append(
                {
                    "type": "actions",
                    "elements": [
                        {
                            "type": "button",
                            "text": {"type": "plain_text", "text": "View in Phoenix"},
                            "url": (
                                f"{settings.phoenix.host}/experiments/"
                                f"{report.candidate_result.experiment_id}"
                            ),
                        }
                    ],
                }
            )

        payload = {"text": SLACK_TEXT_FALLBACK, "blocks": blocks}
        timeout = httpx.Timeout(SLACK_TIMEOUT_SECONDS)
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(self._webhook_url, json=payload)
        if response.is_success:
            logger.info("slack_alert_sent", status_code=response.status_code)
            return
        raise AlertDeliveryError(
            message=f"Slack webhook failed with status {response.status_code}."
        )
