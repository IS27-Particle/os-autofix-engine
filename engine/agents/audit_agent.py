"""Audit specialist agent for validating fixes, verifying non-regression, and detecting collateral damage."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from config.settings import EngineConfig
from engine.agents.remediation_agent import RemediationResult
from engine.agents.triage_agent import TriageFinding
from engine.client import PolicyClient
from sandbox.base import BaseSandbox
from scenarios.base_scenario import BaseScenario

logger = logging.getLogger("os_autofix.engine.agents.audit")


@dataclass
class AuditReport:
    """Evaluation decision produced by the Audit Agent."""

    approved: bool
    verified_healthy: bool
    collateral_damage_detected: bool
    revert_recommended: bool
    notes: str
    audit_checks: list[str] = field(default_factory=list)


class AuditAgent:
    """Verifies system state post-remediation, assesses collateral damage, and approves termination or rollback."""

    def __init__(self, config: EngineConfig, client: PolicyClient | None = None) -> None:
        self.config = config
        self.client = client or PolicyClient(config.llm)

    async def audit(
        self,
        sandbox: BaseSandbox,
        scenario: BaseScenario,
        triage_finding: TriageFinding,
        remediation_result: RemediationResult,
    ) -> AuditReport:
        """Run post-remediation audit and safety verifications."""
        logger.info("Audit Agent: Starting post-remediation security and health audit...")

        audit_checks: list[str] = []

        # 1. Official scenario verification check
        is_verified, verify_msg = await scenario.verify(sandbox)
        audit_checks.append(
            f"Scenario Verifier: {'PASSED' if is_verified else 'FAILED'} - {verify_msg}"
        )

        # 2. Collateral damage checks (e.g. systemd failed units, invalid permissions)
        res_units = await sandbox.execute("systemctl --failed --no-legend 2>/dev/null")
        failed_units = [line for line in res_units.stdout.splitlines() if line.strip()]
        audit_checks.append(f"Failed Units Check: {len(failed_units)} failed systemd units found")

        # 3. Security permissiveness check (e.g. 777 sudoers)
        res_sudoers = await sandbox.execute("stat -c '%a' /etc/sudoers 2>/dev/null")
        is_insecure_sudoers = "777" in res_sudoers.stdout
        if is_insecure_sudoers:
            audit_checks.append("Security Alert: /etc/sudoers has unsafe 777 permissions")

        collateral_damage = is_insecure_sudoers
        approved = is_verified and not collateral_damage
        revert_recommended = not approved

        notes = (
            "System approved and confirmed healthy."
            if approved
            else f"Audit rejected remediation: verified={is_verified}, collateral={collateral_damage}"
        )

        return AuditReport(
            approved=approved,
            verified_healthy=is_verified,
            collateral_damage_detected=collateral_damage,
            revert_recommended=revert_recommended,
            notes=notes,
            audit_checks=audit_checks,
        )
