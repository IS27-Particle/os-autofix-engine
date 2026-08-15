"""Progressive Canary Fleet Rollout Manager.

Orchestrates multi-tier canary progression (Canary 10% -> Staging 50% -> Production 100%)
across an N-instance Incus fleet with real-time error rate evaluation and automatic rollback.
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any

from config.settings import EngineConfig
from engine.agents.coordinator import SwarmCoordinator
from sandbox.base import BaseSandbox
from sandbox.incus_sandbox import IncusSandbox
from scenarios.base_scenario import BaseScenario

logger = logging.getLogger("os_autofix.engine.fleet_orchestrator")


@dataclass
class TierExecutionSummary:
    """Summary of remediation performance within a specific fleet canary tier."""

    tier_name: str
    percentage: float
    nodes_count: int
    target_nodes: list[str]
    success_count: int
    failure_count: int
    error_rate: float
    mttr_seconds: float
    passed: bool


@dataclass
class FleetRolloutResult:
    """Consolidated outcome of a progressive canary fleet rollout."""

    rollout_id: str
    scenario_name: str
    total_fleet_size: int
    tiers_executed: list[TierExecutionSummary] = field(default_factory=list)
    final_status: str = "SUCCESS"  # SUCCESS, FROZEN_ROLLED_BACK, FAILED
    rolled_back_nodes: list[str] = field(default_factory=list)
    total_duration_seconds: float = 0.0
    error_threshold: float = 0.02

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class FleetRolloutOrchestrator:
    """Orchestrates tiered canary rollouts across a fleet of Incus sandboxes with auto-rollback."""

    DEFAULT_TIERS = [0.10, 0.50, 1.00]

    def __init__(
        self,
        config: EngineConfig | None = None,
        sandbox_factory: Callable[[str], BaseSandbox] | None = None,
        error_threshold: float = 0.02,
        tiers: list[float] | None = None,
    ) -> None:
        self.config = config or EngineConfig()
        self.sandbox_factory = sandbox_factory or (
            lambda name: IncusSandbox(name, self.config.incus)
        )
        self.error_threshold = error_threshold
        self.tiers = tiers or self.DEFAULT_TIERS
        self.coordinator = SwarmCoordinator(self.config)

    async def remediate_node(
        self,
        scenario: BaseScenario,
        sandbox: BaseSandbox,
        patch_command: str | None = None,
    ) -> tuple[bool, float]:
        """Apply remediation on a single node and return success status + duration."""
        start = time.monotonic()
        if patch_command:
            res = await sandbox.execute(patch_command)
            ok, _ = await scenario.verify(sandbox)
            success = (res.exit_code == 0) and ok
        else:
            swarm_res = await self.coordinator.run(scenario=scenario, sandbox=sandbox)
            success = swarm_res.success

        duration = time.monotonic() - start
        return success, duration

    async def execute_fleet_rollout(
        self,
        scenario: BaseScenario,
        fleet_size: int = 10,
        patch_command: str | None = None,
        fleet_prefix: str = "fleet-node",
    ) -> FleetRolloutResult:
        """Execute multi-tier progressive canary rollout across the fleet."""
        rollout_id = f"rollout-{int(time.time())}-{uuid.uuid4().hex[:4]}"
        overall_start = time.monotonic()

        logger.info(
            "Fleet Orchestrator [%s]: Initializing %d-node canary rollout for '%s' (Error Threshold: %.1f%%)",
            rollout_id,
            fleet_size,
            scenario.name,
            self.error_threshold * 100,
        )

        # 1. Instantiate fleet sandboxes
        node_names = [f"{fleet_prefix}-{i:02d}" for i in range(1, fleet_size + 1)]
        fleet: dict[str, BaseSandbox] = {name: self.sandbox_factory(name) for name in node_names}

        # 2. Setup baseline and inject faults across all nodes
        logger.info(
            "Fleet Orchestrator: Setting up baseline environment on %d instances...", fleet_size
        )
        snap_base = f"snap-fleet-base-{rollout_id}"

        for _name, sb in fleet.items():
            await sb.setup()
            await scenario.setup(sb)
            await scenario.inject_fault(sb)
            await sb.create_snapshot(snap_base)

        applied_nodes: list[str] = []
        rolled_back_nodes: list[str] = []
        tier_summaries: list[TierExecutionSummary] = []
        final_status = "SUCCESS"

        try:
            # 3. Progressive Tier Progression
            for idx, pct in enumerate(self.tiers, 1):
                tier_target_count = max(1, math.ceil(fleet_size * pct))
                tier_target_count = min(tier_target_count, fleet_size)
                tier_nodes = node_names[:tier_target_count]

                # Identify newly targeted nodes in this tier
                new_nodes = [n for n in tier_nodes if n not in applied_nodes]
                tier_name = (
                    "Canary" if pct <= 0.15 else ("Staging" if pct <= 0.60 else "Production Fleet")
                )
                logger.info(
                    "--- Tier %d/%d: %s (%.0f%% - %d/%d nodes) ---",
                    idx,
                    len(self.tiers),
                    tier_name,
                    pct * 100,
                    len(new_nodes),
                    fleet_size,
                )

                # Concurrently execute remediation on new nodes
                tasks = [self.remediate_node(scenario, fleet[n], patch_command) for n in new_nodes]
                results = await asyncio.gather(*tasks, return_exceptions=True)

                tier_successes = 0
                tier_failures = 0
                tier_durations: list[float] = []

                for node_name, res in zip(new_nodes, results, strict=False):
                    applied_nodes.append(node_name)
                    if isinstance(res, BaseException):
                        logger.error("Node %s failed with exception: %s", node_name, res)
                        tier_failures += 1
                    else:
                        ok, dur = res
                        tier_durations.append(dur)
                        if ok:
                            tier_successes += 1
                        else:
                            tier_failures += 1

                # Calculate tier error rate and MTTR
                error_rate = tier_failures / len(new_nodes) if new_nodes else 0.0
                avg_mttr = (
                    round(sum(tier_durations) / len(tier_durations), 2) if tier_durations else 0.0
                )
                tier_passed = error_rate <= self.error_threshold

                tier_summaries.append(
                    TierExecutionSummary(
                        tier_name=tier_name,
                        percentage=pct,
                        nodes_count=len(new_nodes),
                        target_nodes=new_nodes,
                        success_count=tier_successes,
                        failure_count=tier_failures,
                        error_rate=round(error_rate, 4),
                        mttr_seconds=avg_mttr,
                        passed=tier_passed,
                    )
                )

                # 4. Check Error Threshold & Trigger Atomic Rollback
                if not tier_passed:
                    logger.critical(
                        "Fleet Rollout [%s]: Tier %s breached error threshold (%.2f%% > %.2f%%). FREEZING ROLLOUT!",
                        rollout_id,
                        tier_name,
                        error_rate * 100,
                        self.error_threshold * 100,
                    )
                    final_status = "FROZEN_ROLLED_BACK"

                    # Rollback all applied nodes
                    logger.warning(
                        "Executing atomic rollback across %d affected nodes to snapshot '%s'...",
                        len(applied_nodes),
                        snap_base,
                    )
                    for n in applied_nodes:
                        try:
                            await fleet[n].revert(snap_base)
                            rolled_back_nodes.append(n)
                        except Exception as e:
                            logger.error("Rollback failed on node %s: %e", n, e)
                    break

            total_duration = round(time.monotonic() - overall_start, 2)
            return FleetRolloutResult(
                rollout_id=rollout_id,
                scenario_name=scenario.name,
                total_fleet_size=fleet_size,
                tiers_executed=tier_summaries,
                final_status=final_status,
                rolled_back_nodes=rolled_back_nodes,
                total_duration_seconds=total_duration,
                error_threshold=self.error_threshold,
            )

        finally:
            # Cleanup sandboxes
            for sb in fleet.values():
                try:
                    await sb.cleanup()
                except Exception as e:
                    logger.debug("Cleanup exception: %s", e)
