"""Autonomous Chaos Engineering Daemon running randomized fault injection experiments across Incus sandboxes."""

from __future__ import annotations

import asyncio
import logging
import random
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from config.settings import EngineConfig, get_default_config
from engine.agents.coordinator import SwarmCoordinator
from monitoring.metrics import (
    CHAOS_INJECTIONS_TOTAL,
    CHAOS_MTTR_SECONDS,
    CHAOS_SAFETY_VIOLATIONS,
)
from sandbox.incus_sandbox import IncusSandbox
from scenarios.base_scenario import BaseScenario
from scenarios.registry import get_all_scenarios
from security.ebpf_auditor import SyscallSecurityAuditor

logger = logging.getLogger("os_autofix.engine.chaos_daemon")


@dataclass
class ChaosExperimentResult:
    """Outcome of an autonomous chaos experiment."""

    experiment_id: str
    scenario_name: str
    instance_id: str
    injected: bool
    recovered: bool
    mttr_seconds: float
    safety_score: float
    safety_violations: list[str] = field(default_factory=list)


class ChaosDaemon:
    """Autonomous chaos worker injecting faults into canary sandboxes and measuring autonomous recovery MTTR."""

    def __init__(
        self,
        config: EngineConfig | None = None,
        fleet_size: int = 3,
        rate_minutes: float = 1.0,
        duration_hours: float = 1.0,
        instance_type: str = "container",
        sandbox_factory: Any | None = None,
    ) -> None:
        self.config = config or get_default_config()
        self.fleet_size = fleet_size
        self.rate_minutes = rate_minutes
        self.duration_hours = duration_hours
        self.instance_type = instance_type
        self.sandbox_factory = sandbox_factory
        self.auditor = SyscallSecurityAuditor(safety_threshold=0.7)
        self.coordinator = SwarmCoordinator(self.config, max_cycles=2)
        self.history: list[ChaosExperimentResult] = []
        self._running = False

    async def run_single_experiment(
        self,
        scenario: BaseScenario | None = None,
    ) -> ChaosExperimentResult:
        """Execute a single randomized chaos injection and remediation cycle on a canary sandbox."""
        if not scenario:
            scenarios = get_all_scenarios()
            scenario = random.choice(scenarios)

        exp_id = f"chaos-{uuid.uuid4().hex[:8]}"
        instance_name = f"canary-{exp_id[:12]}"
        logger.info(
            "Chaos Daemon: Launching experiment '%s' with scenario '%s'...", exp_id, scenario.name
        )

        CHAOS_INJECTIONS_TOTAL.inc(scenario=scenario.name)

        sb = (
            self.sandbox_factory(instance_name)
            if self.sandbox_factory
            else IncusSandbox(
                instance_name=instance_name,
                is_vm=(self.instance_type.lower() == "vm"),
            )
        )

        recovered = False
        mttr = 0.0
        safety_score = 1.0
        violations: list[str] = []

        try:
            await sb.setup()
            await scenario.setup(sb)
            await scenario.inject_fault(sb)

            # Attempt autonomous resolution via Swarm Coordinator
            swarm_res = await self.coordinator.run(scenario=scenario, sandbox=sb)
            recovered = swarm_res.success
            mttr = swarm_res.duration_seconds

            # Security audit on executed commands
            if swarm_res.remediation_result:
                for cmd in swarm_res.remediation_result.executed_commands:
                    sec_rep = self.auditor.inspect_command(cmd)
                    if not sec_rep.is_safe:
                        violations.append(sec_rep.rejection_reason)
                        safety_score = min(safety_score, sec_rep.safety_score)
                        CHAOS_SAFETY_VIOLATIONS.inc(violation_type="destructive_command")

            if recovered:
                CHAOS_MTTR_SECONDS.observe(mttr, scenario=scenario.name)
                logger.info("Chaos experiment '%s' recovered in %.2fs (MTTR)", exp_id, mttr)
            else:
                logger.warning("Chaos experiment '%s' failed to recover within limits.", exp_id)

        finally:
            await sb.cleanup()

        result = ChaosExperimentResult(
            experiment_id=exp_id,
            scenario_name=scenario.name,
            instance_id=instance_name,
            injected=True,
            recovered=recovered,
            mttr_seconds=mttr,
            safety_score=safety_score,
            safety_violations=violations,
        )
        self.history.append(result)
        return result

    async def run(self) -> list[ChaosExperimentResult]:
        """Main chaos daemon loop executing Poisson-distributed fault injections over the duration."""
        self._running = True
        end_time = time.monotonic() + (self.duration_hours * 3600)
        logger.info(
            "Chaos Daemon started (fleet_size=%d, rate=%.1fm, duration=%.1fh)...",
            self.fleet_size,
            self.rate_minutes,
            self.duration_hours,
        )

        while self._running and time.monotonic() < end_time:
            # Poisson-distributed interval (mean = rate_minutes * 60)
            wait_interval = random.expovariate(1.0 / max(1.0, self.rate_minutes * 60))
            logger.info("Chaos Daemon: Next injection scheduled in %.1f seconds...", wait_interval)
            await asyncio.sleep(min(wait_interval, 5.0))  # Capped for responsiveness

            try:
                await self.run_single_experiment()
            except Exception as e:
                logger.error("Error during chaos experiment execution: %s", e)

        logger.info("Chaos Daemon completed %d experiments.", len(self.history))
        return self.history

    def stop(self) -> None:
        """Signal the daemon loop to stop."""
        self._running = False

    def get_summary_metrics(self) -> dict[str, Any]:
        """Compute aggregate recovery metrics across completed chaos experiments."""
        if not self.history:
            return {"total_experiments": 0, "recovery_rate": 0.0, "mean_mttr_seconds": 0.0}

        recoveries = [h for h in self.history if h.recovered]
        mean_mttr = sum(h.mttr_seconds for h in recoveries) / len(recoveries) if recoveries else 0.0

        return {
            "total_experiments": len(self.history),
            "recoveries": len(recoveries),
            "recovery_rate": round(len(recoveries) / len(self.history), 3),
            "mean_mttr_seconds": round(mean_mttr, 2),
            "avg_safety_score": round(
                sum(h.safety_score for h in self.history) / len(self.history), 3
            ),
        }
