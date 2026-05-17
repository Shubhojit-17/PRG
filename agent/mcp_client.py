"""Async MCP client for Phoenix tool calls over stdio."""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from typing import Any

import structlog
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from agent.models import ExperimentResult, PromptVersion
from config.settings import settings

JSONRPC_VERSION = "2.0"
MCP_METHOD_INITIALIZE = "initialize"
MCP_METHOD_INITIALIZED = "initialized"
MCP_METHOD_TOOLS_CALL = "tools/call"
MCP_METHOD_SHUTDOWN = "shutdown"
MCP_PROTOCOL_VERSION = "2024-11-05"
MCP_CLIENT_NAME = "prompt-regression-guardian"
MCP_CLIENT_VERSION = "0.1.0"
MCP_TOOL_TIMEOUT_SECONDS = 10
MCP_EXPERIMENT_TIMEOUT_SECONDS = 120
MCP_EXPERIMENT_POLL_SECONDS = 3
MCP_SUBPROCESS_COMMAND = ("npx", "-y", "@arizeai/phoenix-mcp", "--baseUrl")

TOOL_LIST_PROMPTS = "list_prompts"
TOOL_GET_PROMPT = "get_prompt"
TOOL_LIST_DATASETS = "list_datasets"
TOOL_RUN_EXPERIMENT = "run_experiment"
TOOL_GET_EXPERIMENT = "get_experiment"
TOOL_ANNOTATE_PROMPT = "annotate_prompt"

logger = structlog.get_logger()


@dataclass
class MCPToolError(Exception):
    """Exception raised when an MCP tool reports an error."""

    tool_name: str
    message: str


class PhoenixMCPClient:
    """Async MCP client that wraps Phoenix MCP tool calls."""

    def __init__(self, base_url: str | None = None) -> None:
        """Initialize the MCP client.

        Args:
            base_url: Phoenix base URL for the MCP server.
        """

        self._base_url = base_url or settings.phoenix.host
        self._process: asyncio.subprocess.Process | None = None
        self._stdin: asyncio.StreamWriter | None = None
        self._stdout: asyncio.StreamReader | None = None
        self._request_id = 0

    async def __aenter__(self) -> "PhoenixMCPClient":
        """Start the MCP subprocess and perform the initialize handshake.

        Returns:
            PhoenixMCPClient: The active client instance.
        """

        self._process = await asyncio.create_subprocess_exec(
            *MCP_SUBPROCESS_COMMAND,
            self._base_url,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        if self._process.stdin is None or self._process.stdout is None:
            raise ConnectionError("Failed to open MCP stdio pipes.")
        self._stdin = self._process.stdin
        self._stdout = self._process.stdout
        await self._send_initialize()
        await self._read_response(MCP_TOOL_TIMEOUT_SECONDS)
        await self._send_json({"jsonrpc": JSONRPC_VERSION, "method": MCP_METHOD_INITIALIZED})
        return self

    async def __aexit__(self, exc_type, exc, traceback) -> None:
        """Shutdown the MCP subprocess.

        Args:
            exc_type: Exception type.
            exc: Exception instance.
            traceback: Exception traceback.
        """

        if self._process is None:
            return
        try:
            await self._send_json({"jsonrpc": JSONRPC_VERSION, "method": MCP_METHOD_SHUTDOWN})
        except ConnectionError:
            logger.warning("mcp_shutdown_failed")
        self._process.terminate()
        await self._process.wait()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=2, max=8),
        retry=retry_if_exception_type((ConnectionError, asyncio.TimeoutError)),
        reraise=True,
    )
    async def list_prompts(self) -> list[dict[str, Any]]:
        """List prompts in Phoenix.

        Returns:
            list[dict[str, Any]]: Prompt metadata objects.
        """

        result = await self._call_tool(TOOL_LIST_PROMPTS, {})
        if not isinstance(result, list):
            raise ConnectionError("Unexpected list_prompts response.")
        return result

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=2, max=8),
        retry=retry_if_exception_type((ConnectionError, asyncio.TimeoutError)),
        reraise=True,
    )
    async def get_prompt_version(self, prompt_id: str, version_id: str) -> PromptVersion:
        """Fetch a specific prompt version.

        Args:
            prompt_id: Prompt identifier.
            version_id: Prompt version identifier.

        Returns:
            PromptVersion: The prompt version details.
        """

        result = await self._call_tool(
            TOOL_GET_PROMPT, {"prompt_id": prompt_id, "version_id": version_id}
        )
        if not isinstance(result, dict):
            raise ConnectionError("Unexpected get_prompt response.")
        return PromptVersion.model_validate(result)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=2, max=8),
        retry=retry_if_exception_type((ConnectionError, asyncio.TimeoutError)),
        reraise=True,
    )
    async def list_datasets(self, project: str) -> list[dict[str, Any]]:
        """List datasets for the given project.

        Args:
            project: Phoenix project name.

        Returns:
            list[dict[str, Any]]: Dataset metadata objects.
        """

        result = await self._call_tool(TOOL_LIST_DATASETS, {"project": project})
        if not isinstance(result, list):
            raise ConnectionError("Unexpected list_datasets response.")
        return result

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=2, max=8),
        retry=retry_if_exception_type((ConnectionError, asyncio.TimeoutError)),
        reraise=True,
    )
    async def run_experiment(
        self, prompt_version_id: str, dataset_id: str, judge_model: str
    ) -> str:
        """Run an evaluation experiment.

        Args:
            prompt_version_id: Prompt version identifier.
            dataset_id: Dataset identifier.
            judge_model: Model used as the judge.

        Returns:
            str: Experiment identifier.
        """

        result = await self._call_tool(
            TOOL_RUN_EXPERIMENT,
            {
                "prompt_version_id": prompt_version_id,
                "dataset_id": dataset_id,
                "judge_model": judge_model,
            },
        )
        if isinstance(result, dict):
            experiment_id = result.get("experiment_id") or result.get("id")
            if isinstance(experiment_id, str):
                return experiment_id
        raise ConnectionError("Unexpected run_experiment response.")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=2, max=8),
        retry=retry_if_exception_type((ConnectionError, asyncio.TimeoutError)),
        reraise=True,
    )
    async def get_experiment_results(self, experiment_id: str) -> ExperimentResult:
        """Poll until an experiment completes and return its results.

        Args:
            experiment_id: Experiment identifier.

        Returns:
            ExperimentResult: Completed experiment results.
        """

        deadline = time.monotonic() + MCP_EXPERIMENT_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            result = await self._call_tool(TOOL_GET_EXPERIMENT, {"experiment_id": experiment_id})
            if not isinstance(result, dict):
                raise ConnectionError("Unexpected get_experiment response.")
            status = str(result.get("status", "")).lower()
            if status == "completed":
                return ExperimentResult.model_validate(result)
            await asyncio.sleep(MCP_EXPERIMENT_POLL_SECONDS)
        raise asyncio.TimeoutError("Experiment polling timed out.")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=2, max=8),
        retry=retry_if_exception_type((ConnectionError, asyncio.TimeoutError)),
        reraise=True,
    )
    async def annotate_prompt_version(
        self, prompt_id: str, version_id: str, label: str, note: str
    ) -> None:
        """Annotate a prompt version in Phoenix.

        Args:
            prompt_id: Prompt identifier.
            version_id: Prompt version identifier.
            label: Annotation label.
            note: Annotation note.
        """

        result = await self._call_tool(
            TOOL_ANNOTATE_PROMPT,
            {"prompt_id": prompt_id, "version_id": version_id, "label": label, "note": note},
        )
        if result is None:
            return
        if isinstance(result, dict) and result.get("status") == "ok":
            return
        raise ConnectionError("Unexpected annotate_prompt response.")

    async def _send_initialize(self) -> None:
        """Send the MCP initialize request."""

        self._request_id += 1
        payload = {
            "jsonrpc": JSONRPC_VERSION,
            "id": self._request_id,
            "method": MCP_METHOD_INITIALIZE,
            "params": {
                "protocolVersion": MCP_PROTOCOL_VERSION,
                "clientInfo": {"name": MCP_CLIENT_NAME, "version": MCP_CLIENT_VERSION},
                "capabilities": {},
            },
        }
        await self._send_json(payload)

    async def _call_tool(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        """Send a tools/call request and return the tool response.

        Args:
            tool_name: Name of the tool to invoke.
            arguments: Tool arguments.

        Returns:
            Any: Tool response payload.
        """

        self._request_id += 1
        payload = {
            "jsonrpc": JSONRPC_VERSION,
            "id": self._request_id,
            "method": MCP_METHOD_TOOLS_CALL,
            "params": {"name": tool_name, "arguments": arguments},
        }
        await self._send_json(payload)
        response = await self._read_response(MCP_TOOL_TIMEOUT_SECONDS)
        result = response.get("result")
        if isinstance(result, dict) and result.get("isError"):
            message = str(result.get("message") or result.get("content") or "Unknown error")
            raise MCPToolError(tool_name=tool_name, message=message)
        if isinstance(result, dict) and "content" in result:
            return result.get("content")
        return result

    async def _send_json(self, payload: dict[str, Any]) -> None:
        """Send a newline-delimited JSON payload to MCP.

        Args:
            payload: JSON payload to send.
        """

        if self._stdin is None:
            raise ConnectionError("MCP stdin is not available.")
        line = json.dumps(payload).encode("utf-8") + b"\n"
        self._stdin.write(line)
        await self._stdin.drain()

    async def _read_response(self, timeout_seconds: int) -> dict[str, Any]:
        """Read a single JSON response from MCP stdout.

        Args:
            timeout_seconds: Timeout in seconds.

        Returns:
            dict[str, Any]: Parsed JSON response.
        """

        if self._stdout is None:
            raise ConnectionError("MCP stdout is not available.")
        try:
            line = await asyncio.wait_for(self._stdout.readline(), timeout=timeout_seconds)
        except asyncio.TimeoutError as exc:
            raise asyncio.TimeoutError("Timed out waiting for MCP response.") from exc
        if not line:
            raise ConnectionError("MCP stdout closed unexpectedly.")
        return json.loads(line.decode("utf-8"))
