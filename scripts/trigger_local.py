"""Trigger the guardian run locally via HTTP."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from typing import Any

import httpx
import structlog
from dotenv import load_dotenv

DEFAULT_AGENT_URL = "http://localhost:8080/run"
REQUEST_TIMEOUT_SECONDS = 120

logger = structlog.get_logger()


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser.

    Returns:
        argparse.ArgumentParser: Configured parser.
    """

    parser = argparse.ArgumentParser(description="Trigger a guardian run.")
    parser.add_argument(
        "--agent-url",
        default=os.getenv("AGENT_URL", DEFAULT_AGENT_URL),
        help="Guardian /run endpoint URL.",
    )
    parser.add_argument(
        "--project-name",
        default=None,
        help="Optional Phoenix project override.",
    )
    return parser


async def trigger_run(agent_url: str, project_name: str | None) -> dict[str, Any]:
    """Send the run request to the guardian service.

    Args:
        agent_url: Guardian /run URL.
        project_name: Optional project override.

    Returns:
        dict[str, Any]: JSON response from the agent.
    """

    payload = {"project_name": project_name} if project_name else None
    timeout = httpx.Timeout(REQUEST_TIMEOUT_SECONDS)
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(agent_url, json=payload)
        response.raise_for_status()
        return response.json()


async def run() -> None:
    """Run the local trigger script."""

    load_dotenv()
    args = build_parser().parse_args()
    logger.info("triggering_guardian", agent_url=args.agent_url)
    result = await trigger_run(args.agent_url, args.project_name)
    logger.info("guardian_completed", result=result)
    print(json.dumps(result, indent=2, sort_keys=True))


def main() -> None:
    """Entry point for the trigger script."""

    asyncio.run(run())


if __name__ == "__main__":
    main()
