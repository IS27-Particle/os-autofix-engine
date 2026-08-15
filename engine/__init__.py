"""Engine package for structured schemas, multi-backend LLM clients, orchestration, benchmark reporting, and continuous learning."""

from engine.action_schema import (
    AgentAction,
    extract_json_block,
    parse_action_response,
)
from engine.cascading_fuzzer import CascadingFaultFuzzer, CompoundFaultResult
from engine.causal_tracer import CausalGraph, CausalTracer
from engine.client import LLMClientError, PolicyClient
from engine.continuous_loop import ContinuousFeedbackLoop, LoopIterationResult
from engine.criu_state_preserver import CRIUStatePreserver, ProcessCheckpointResult
from engine.deployer import OllamaDeployer, generate_modelfile_content
from engine.fleet_orchestrator import (
    FleetRolloutOrchestrator,
    FleetRolloutResult,
    TierExecutionSummary,
)
from engine.orchestrator import Orchestrator
from engine.reporter import BenchmarkReporter
from engine.shadow_evaluator import (
    DifferentialMetrics,
    DifferentialStateReport,
    ShadowEvaluator,
)

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
    "CausalGraph",
    "CausalTracer",
    "CascadingFaultFuzzer",
    "CompoundFaultResult",
    "ShadowEvaluator",
    "DifferentialStateReport",
    "DifferentialMetrics",
    "CRIUStatePreserver",
    "ProcessCheckpointResult",
    "FleetRolloutOrchestrator",
    "FleetRolloutResult",
    "TierExecutionSummary",
]
