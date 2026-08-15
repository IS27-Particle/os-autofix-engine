"""Unit tests validating payload formatting and connection handling for Ollama and Open-WebUI endpoints."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from config.settings import LLMConfig
from engine.action_schema import AgentAction
from engine.client import PolicyClient


def test_ollama_endpoint_configuration() -> None:
    """Test Ollama configuration bindings and URL resolution."""
    cfg = LLMConfig(
        backend="ollama",
        ollama_base_url="http://10.0.0.25:11434/v1",
        model_name="qwen2.5-coder:7b",
    )
    client = PolicyClient(cfg)

    assert client.config.active_endpoint == "http://10.0.0.25:11434/v1"
    assert client.config.model_name == "qwen2.5-coder:7b"
    assert client.config.active_api_key == "ollama"


def test_open_webui_endpoint_configuration() -> None:
    """Test Open-WebUI endpoint URL resolution and Bearer auth header injection."""
    cfg = LLMConfig(
        backend="open-webui",
        open_webui_base_url="https://ai.is27.duckdns.org/api",
        open_webui_api_key="sk-test-token-12345",
        model_name="llama3.2:latest",
    )
    client = PolicyClient(cfg)

    assert client.config.active_endpoint == "https://ai.is27.duckdns.org/api"
    assert client.config.active_api_key == "sk-test-token-12345"
    assert client.http_client.headers.get("Authorization") == "Bearer sk-test-token-12345"


@pytest.mark.asyncio
async def test_client_payload_formatting_and_schema_enforcement() -> None:
    """Test standard chat completion request format with json_object schema hint."""
    cfg = LLMConfig(backend="ollama", ollama_base_url="http://10.0.0.25:11434/v1")
    client = PolicyClient(cfg)

    mock_resp = MagicMock()
    mock_resp.choices = [
        MagicMock(
            message=MagicMock(
                content=json.dumps(
                    {
                        "thought": "Investigating network issue",
                        "command": "ip route show",
                        "timeout_seconds": 10,
                        "is_done": False,
                        "confidence": 0.9,
                    }
                )
            )
        )
    ]

    mock_create = AsyncMock(return_value=mock_resp)
    client.openai_client.chat.completions.create = mock_create  # type: ignore[method-assign]

    messages = [{"role": "user", "content": "Fix routing"}]
    action, raw = await client.get_next_action(messages)

    assert isinstance(action, AgentAction)
    assert action.thought == "Investigating network issue"
    assert action.command == "ip route show"
    assert action.is_done is False
    assert mock_create.called


@pytest.mark.asyncio
async def test_ollama_native_api_fallback() -> None:
    """Test fallback to native Ollama /api/chat endpoint when OpenAI format encounters error."""
    cfg = LLMConfig(backend="ollama", ollama_base_url="http://10.0.0.25:11434/v1")
    client = PolicyClient(cfg)

    # Fail OpenAI format call
    client.openai_client.chat.completions.create = AsyncMock(  # type: ignore[method-assign]
        side_effect=Exception("OpenAI endpoint unavailable")
    )

    # Mock native Ollama HTTP response
    mock_http_response = MagicMock(spec=httpx.Response)
    mock_http_response.status_code = 200
    mock_http_response.json.return_value = {
        "message": {
            "content": json.dumps(
                {
                    "thought": "Restarting DNS daemon",
                    "command": "systemctl restart systemd-resolved",
                    "timeout_seconds": 15,
                    "is_done": True,
                }
            )
        }
    }
    mock_http_response.raise_for_status = MagicMock()

    mock_post = AsyncMock(return_value=mock_http_response)
    client.http_client.post = mock_post  # type: ignore[method-assign]

    messages = [{"role": "user", "content": "Fix DNS"}]
    action, raw = await client.get_next_action(messages)

    assert action.thought == "Restarting DNS daemon"
    assert action.command == "systemctl restart systemd-resolved"
    assert action.is_done is True
    assert mock_post.called


@pytest.mark.asyncio
async def test_client_retry_and_error_feedback_loop() -> None:
    """Test automated retry loop with validation error reflection on malformed JSON payload."""
    cfg = LLMConfig(backend="ollama", max_retries=2)
    client = PolicyClient(cfg)

    # Attempt 1: Invalid JSON; Attempt 2: Correct JSON
    mock_bad = MagicMock()
    mock_bad.choices = [MagicMock(message=MagicMock(content="MALFORMED_OUTPUT_NOT_JSON"))]

    mock_good = MagicMock()
    mock_good.choices = [
        MagicMock(
            message=MagicMock(
                content=json.dumps(
                    {
                        "thought": "Clean JSON repair",
                        "command": "dpkg --configure -a",
                        "timeout_seconds": 15,
                        "is_done": True,
                    }
                )
            )
        )
    ]

    mock_create_retry = AsyncMock(side_effect=[mock_bad, mock_good])
    client.openai_client.chat.completions.create = mock_create_retry  # type: ignore[method-assign]

    messages = [{"role": "user", "content": "Fix package manager"}]
    action, raw = await client.get_next_action(messages)

    assert action.thought == "Clean JSON repair"
    assert action.command == "dpkg --configure -a"
    assert action.is_done is True
    assert mock_create_retry.call_count == 2


def test_mock_policy_heuristics() -> None:
    """Test offline heuristic mock policy for development without reachable endpoints."""
    cfg = LLMConfig(mock_mode=True)
    client = PolicyClient(cfg)

    dns_action = client._mock_action([{"role": "user", "content": "systemd-resolved broken DNS"}])
    assert "dns" in dns_action.thought.lower() or "resolved" in dns_action.thought.lower()

    route_action = client._mock_action([{"role": "user", "content": "network routing corrupted"}])
    assert "route" in route_action.thought.lower() or "default" in route_action.command

    pkg_action = client._mock_action([{"role": "user", "content": "dpkg lockfile held"}])
    assert "lock" in pkg_action.command or "dpkg" in pkg_action.command
