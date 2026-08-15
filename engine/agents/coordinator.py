"""Coordinator orchestrating multi-agent specialist handoffs (Triage -> Remediation -> Audit) in Incus sandboxes."""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass

from config.settings import EngineConfig
from engine.agents.audit_agent import AuditAgent, AuditReport
from engine.agents.remediation_agent import RemediationAgent, RemediationResult
from engine.agents.triage_agent import TriageAgent, TriageFinding
from engine.client import PolicyClient
from sandbox.base import BaseSandbox
from scenarios.base_scenario import BaseScenario

logger = logging.getLogger("os_autofix.engine.agents.coordinator")


@dataclass
class SwarmResult:
    """Consolidated outcome of the Tri-Agent specialist swarm execution."""

    scenario_name: str
    instance_id: str
    success: bool
    cycles_executed: int
    duration_seconds: float
    triage_finding: TriageFinding | None = None
    remediation_result: RemediationResult | None = None
    audit_report: AuditReport | None = None
    reverted_on_failure: bool = False
    details: str = ""


class SwarmCoordinator:
    """Coordinates specialist agents (Triage, Remediation, Audit) to resolve complex OS faults with rollback safety."""

    def __init__(
        self,
        config: EngineConfig,
        client: PolicyClient | None = None,
        max_cycles: int = 2,
    ) -> None:
        self.config = config
        self.client = client or PolicyClient(config.llm)
        self.max_cycles = max_cycles
        self.triage = TriageAgent(config, self.client)
        self.remediation = RemediationAgent(config, self.client)
        self.audit = AuditAgent(config, self.client)

    async def run(
        self,
        scenario: BaseScenario,
        sandbox: BaseSandbox,
    ) -> SwarmResult:
        """Execute the coordinated Tri-Agent lifecycle on the target sandbox."""
        logger.info(
            "Swarm Coordinator: Launching Tri-Agent swarm on scenario '%s' (sandbox: %s)...",
            scenario.name,
            getattr(sandbox, "instance_name", "sandbox"),
        )

        start_time = time.monotonic()
        baseline_snapshot = f"snap-swarm-base-{uuid.uuid4().hex[:6]}"
        await sandbox.create_snapshot(baseline_snapshot)

        latest_triage: TriageFinding | None = None
        latest_remediation: RemediationResult | None = None
        latest_audit: AuditReport | None = None
        reverted = False

        for cycle in range(1, self.max_cycles + 1):
            logger.info("--- Tri-Agent Swarm Cycle %d/%d ---", cycle, self.max_cycles)

            # Phase 1: Triage (Read-only inspection)
            latest_triage = await self.triage.diagnose(
                sandbox=sandbox,
                symptom_description=scenario.description,
            )

            # Phase 2: Remediation (Surgical state mutation)
            latest_remediation = await self.remediation.remediate(
                sandbox=sandbox,
                triage_finding=latest_triage,
            )

            # Phase 3: Audit (Verification & Collateral safety check)
            latest_audit = await self.audit.audit(
                sandbox=sandbox,
                scenario=scenario,
                triage_finding=latest_triage,
                remediation_result=latest_remediation,
            )

            if latest_audit.approved:
                logger.info("Tri-Agent Swarm: Audit approved remediation on cycle %d!", cycle)
                break

            if latest_audit.revert_recommended and cycle < self.max_cycles:
                logger.warning(
                    "Audit rejected changes. Reverting to baseline snapshot '%s'...",
                    baseline_snapshot,
                )
                await sandbox.revert(baseline_snapshot)
                reverted = True

        duration = time.monotonic() - start_time
        success = latest_audit.approved if latest_audit else False

        inst_id = getattr(sandbox, "instance_name", getattr(sandbox, "name", "sandbox"))
        return SwarmResult(
            scenario_name=scenario.name,
            instance_id=inst_id,
            success=success,
            cycles_executed=cycle,
            duration_seconds=duration,
            triage_finding=latest_triage,
            remediation_result=latest_remediation,
            audit_report=latest_audit,
            reverted_on_failure=reverted,
            details=latest_audit.notes if latest_audit else "No audit completed.",
        )
