"""Unit tests for Ollama deployer, Modelfile generation, and remote API streaming."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from unittest.mock import MagicMock, patch

import httpx
import pytest

from engine.deployer import OllamaDeployer, generate_modelfile_content


def test_generate_modelfile_structure() -> None:
    """Test generating Modelfile with stop tokens and custom temperatures."""
    content = generate_modelfile_content(
        base_model_or_gguf="/data/models/qwen.gguf",
        temperature=0.3,
        top_p=0.85,
        stop_tokens=["<|end|>", "</s>"],
    )

    assert "FROM /data/models/qwen.gguf" in content
    assert "PARAMETER temperature 0.3" in content
    assert "PARAMETER top_p 0.85" in content
    assert 'PARAMETER stop "<|end|>"' in content
    assert 'PARAMETER stop "</s>"' in content


@pytest.mark.asyncio
async def test_ollama_list_models() -> None:
    """Test querying and parsing registered models from Ollama /api/tags."""
    deployer = OllamaDeployer(base_url="http://10.0.0.25:11434")

    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "models": [
            {"name": "qwen2.5-coder:7b"},
            {"name": "os-fixer:v1"},
        ]
    }
    mock_resp.raise_for_status = MagicMock()

    with patch("httpx.AsyncClient.get", return_value=mock_resp):
        models = await deployer.list_models()
        assert "qwen2.5-coder:7b" in models
        assert "os-fixer:v1" in models


@pytest.mark.asyncio
async def test_ollama_deploy_model_streaming() -> None:
    """Test model deployment via HTTP stream to /api/create."""
    deployer = OllamaDeployer(base_url="http://10.0.0.25:11434")

    events = [
        {"status": "reading model metadata"},
        {"status": "processing layers"},
        {"status": "success"},
    ]

    async def mock_stream(*args: object, **kwargs: object) -> AsyncGenerator[dict[str, str], None]:
        for ev in events:
            yield ev

    with patch.object(deployer, "create_model_stream", side_effect=mock_stream):
        with patch.object(deployer, "list_models", return_value=["os-fixer:v1"]):
            ok = await deployer.deploy_model(
                model_name="os-fixer:v1",
                base_model_or_gguf="qwen2.5-coder:7b",
            )
            assert ok is True


@pytest.mark.asyncio
async def test_ollama_delete_model() -> None:
    """Test deleting model tag via /api/delete."""
    deployer = OllamaDeployer(base_url="http://10.0.0.25:11434")

    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = 200

    with patch("httpx.AsyncClient.request", return_value=mock_resp):
        deleted = await deployer.delete_model("old-model:tag")
        assert deleted is True
