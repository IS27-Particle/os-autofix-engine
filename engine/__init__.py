"""Engine package for structured schemas, multi-backend LLM clients, orchestration, benchmark reporting, and continuous learning."""

from engine.action_schema import (
    AgentAction,
    extract_json_block,
    parse_action_response,
)
from engine.client import LLMClientError, PolicyClient
from engine.continuous_loop import ContinuousFeedbackLoop, LoopIterationResult
from engine.deployer import OllamaDeployer, generate_modelfile_content
from engine.orchestrator import Orchestrator
from engine.reporter import BenchmarkReporter

__all__ = [
    "AgentAction",
    "extract_json_block",
    "parse_action_response",
    "PolicyClient",
    "LLMClientError",
    "Orchestrator",
    "OllamaDeployer",
    "generate_modelfile_content",
    "ContinuousFeedbackLoop",
    "LoopIterationResult",
    "BenchmarkReporter",
]
