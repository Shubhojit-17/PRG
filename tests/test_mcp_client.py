"""Tests for the PhoenixMCPClient implementation."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent.mcp_client import (
    JSONRPC_VERSION,
    MCP_CLIENT_NAME,
    MCP_CLIENT_VERSION,
    MCP_METHOD_INITIALIZE,
    MCP_PROTOCOL_VERSION,
    PhoenixMCPClient,
    MCPToolError,
)


class FakeStdin:
    """Fake stdin stream for subprocess testing."""

    def __init__(self) -> None:
        """Initialize the fake stdin."""

        self.writes: list[bytes] = []
        self.write = MagicMock(side_effect=self._write)
        self.drain = AsyncMock()

    def _write(self, data: bytes) -> None:
        """Store written data.

        Args:
            data: Raw bytes written to stdin.
        """

        self.writes.append(data)


class FakeStdout:
    """Fake stdout stream for subprocess testing."""

    def __init__(self, lines: list[bytes]) -> None:
        """Initialize the fake stdout.

        Args:
            lines: Lines to emit on readline.
        """

        self.readline = AsyncMock(side_effect=lines)


class FakeProcess:
    """Fake subprocess process for MCP tests."""

    def __init__(self, stdout_lines: list[bytes]) -> None:
        """Initialize the fake process.

        Args:
            stdout_lines: Lines returned by stdout.
        """

        self.stdin = FakeStdin()
        self.stdout = FakeStdout(stdout_lines)
        self.stderr = AsyncMock()
        self.terminate = MagicMock()
        self.wait = AsyncMock()


def build_response(payload: dict[str, Any]) -> bytes:
    """Serialize a response payload as a JSON line.

    Args:
        payload: Response payload.

    Returns:
        bytes: Serialized JSON line.
    """

    return (json.dumps(payload) + "\n").encode("utf-8")


@pytest.mark.asyncio
async def test_initialize_handshake_sends_correct_json() -> None:
    """Ensure initialize handshake matches the MCP spec payload."""

    init_response = build_response({"jsonrpc": "2.0", "id": 1, "result": {}})
    process = FakeProcess([init_response])

    with patch("asyncio.create_subprocess_exec", new=AsyncMock(return_value=process)):
        async with PhoenixMCPClient(base_url="http://localhost:6006"):
            pass

    sent = json.loads(process.stdin.writes[0].decode("utf-8"))
    assert sent["jsonrpc"] == JSONRPC_VERSION
    assert sent["method"] == MCP_METHOD_INITIALIZE
    assert sent["params"]["protocolVersion"] == MCP_PROTOCOL_VERSION
    assert sent["params"]["clientInfo"]["name"] == MCP_CLIENT_NAME
    assert sent["params"]["clientInfo"]["version"] == MCP_CLIENT_VERSION


@pytest.mark.asyncio
async def test_list_prompts_parses_response() -> None:
    """Return prompt metadata from the list_prompts tool call."""

    init_response = build_response({"jsonrpc": "2.0", "id": 1, "result": {}})
    list_response = build_response(
        {
            "jsonrpc": "2.0",
            "id": 2,
            "result": {
                "isError": False,
                "content": [{"id": "p1", "name": "demo", "latest_version_id": "v2"}],
            },
        }
    )
    process = FakeProcess([init_response, list_response])

    with patch("asyncio.create_subprocess_exec", new=AsyncMock(return_value=process)):
        async with PhoenixMCPClient(base_url="http://localhost:6006") as client:
            prompts = await client.list_prompts()

    assert len(prompts) == 1
    assert set(prompts[0].keys()) == {"id", "name", "latest_version_id"}


@pytest.mark.asyncio
async def test_retries_on_connection_error() -> None:
    """Retry list_prompts up to three times on ConnectionError."""

    init_response = build_response({"jsonrpc": "2.0", "id": 1, "result": {}})
    process = FakeProcess([init_response])

    with patch("asyncio.create_subprocess_exec", new=AsyncMock(return_value=process)):
        async with PhoenixMCPClient(base_url="http://localhost:6006") as client:
            with patch.object(
                PhoenixMCPClient,
                "_call_tool",
                new=AsyncMock(
                    side_effect=[
                        ConnectionError("fail-1"),
                        ConnectionError("fail-2"),
                        [{"id": "p1"}],
                    ]
                ),
            ) as call_mock:
                prompts = await client.list_prompts()

    assert len(prompts) == 1
    assert call_mock.call_count == 3


@pytest.mark.asyncio
async def test_raises_mcp_tool_error_on_is_error_true() -> None:
    """Raise MCPToolError when the MCP response indicates an error."""

    init_response = build_response({"jsonrpc": "2.0", "id": 1, "result": {}})
    error_response = build_response(
        {
            "jsonrpc": "2.0",
            "id": 2,
            "result": {"isError": True, "message": "boom"},
        }
    )
    process = FakeProcess([init_response, error_response])

    with patch("asyncio.create_subprocess_exec", new=AsyncMock(return_value=process)):
        async with PhoenixMCPClient(base_url="http://localhost:6006") as client:
            with pytest.raises(MCPToolError) as exc:
                await client.list_prompts()

    assert exc.value.tool_name == "list_prompts"
