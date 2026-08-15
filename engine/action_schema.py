"""Structured action schemas and JSON parsing utilities for agent policies."""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import BaseModel, Field


class AgentAction(BaseModel):
    """Strict structured action emitted by an OS troubleshooting policy."""

    thought: str = Field(
        ...,
        description="Chain-of-thought analysis of current system state, root cause hypothesis, and planned next step.",
        min_length=1,
    )
    command: str = Field(
        default="",
        description="Non-interactive shell command to execute in the guest environment. Empty if done.",
    )
    timeout_seconds: int = Field(
        default=15,
        ge=1,
        le=60,
        description="Timeout in seconds for this command execution (default: 15s).",
    )
    is_done: bool = Field(
        default=False,
        description="Set to True when you believe the fault is completely resolved and ready for verification.",
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Subjective confidence score (0.0 to 1.0) in current hypothesis or fix.",
    )

    @classmethod
    def get_json_schema(cls) -> dict[str, Any]:
        """Return the strict JSON Schema for OpenAI / Ollama structured decoding."""
        return cls.model_json_schema()


def extract_json_block(text: str) -> str:
    """Extract JSON string from potential markdown code fences or raw string."""
    text = text.strip()

    # Match ```json ... ``` or ``` ... ```
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence_match:
        return fence_match.group(1).strip()

    # Match raw JSON object spanning the string
    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        return text[first_brace : last_brace + 1].strip()

    return text


def parse_action_response(response_text: str) -> AgentAction:
    """Parse and validate LLM output into an AgentAction model.

    Raises:
        ValueError: If JSON is unparseable.
        ValidationError: If JSON does not conform to AgentAction schema.
    """
    cleaned = extract_json_block(response_text)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValueError(f"Malformed JSON response: {e}\nRaw output:\n{response_text}") from e

    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object, received {type(data).__name__}")

    return AgentAction.model_validate(data)
