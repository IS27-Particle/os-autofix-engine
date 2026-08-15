"""Unit tests for the Tri-Agent Specialist Swarm (Triage -> Remediation -> Audit)."""

from __future__ import annotations

import pytest

from config.settings import EngineConfig
from engine.agents.audit_agent import AuditAgent
from engine.agents.coordinator import SwarmCoordinator
from engine.agents.remediation_agent import RemediationAgent
from engine.agents.triage_agent import TriageAgent
from scenarios.systemd_dns import SystemdDNSScenario
from tests.conftest import MockSandbox


@pytest.mark.asyncio
async def test_triage_agent_read_only_investigation() -> None:
    """Test that TriageAgent performs read-only probes and isolates root causes."""
    cfg = EngineConfig()
    cfg.llm.mock_mode = True
    agent = TriageAgent(cfg)
    sandbox = MockSandbox("triage-test")

    finding = await agent.diagnose(sandbox, symptom_description="DNS resolution failure")
    assert finding is not None
    assert "systemd-resolved" in finding.affected_daemons
    assert len(finding.evidence) >= 1


@pytest.mark.asyncio
async def test_remediation_agent_execution() -> None:
    """Test that RemediationAgent executes targeted mutations from triage findings."""
    cfg = EngineConfig()
    cfg.llm.mock_mode = True
    agent = RemediationAgent(cfg)
    sandbox = MockSandbox("remed-test")
    triage_agent = TriageAgent(cfg)

    finding = await triage_agent.diagnose(sandbox, symptom_description="DNS resolution failure")
    result = await agent.remediate(sandbox, finding)

    assert result.success_attempted is True
    assert len(result.executed_commands) >= 1
    assert "systemctl restart" in result.executed_commands[0]


@pytest.mark.asyncio
async def test_audit_agent_approval() -> None:
    """Test that AuditAgent verifies system health and checks for collateral damage."""
    cfg = EngineConfig()
    cfg.llm.mock_mode = True
    audit = AuditAgent(cfg)
    sandbox = MockSandbox("audit-test")
    scenario = SystemdDNSScenario()

    await scenario.setup(sandbox)
    await scenario.inject_fault(sandbox)
    # Fix the fault
    await sandbox.execute("systemctl restart systemd-resolved")

    triage_agent = TriageAgent(cfg)
    remed_agent = RemediationAgent(cfg)
    finding = await triage_agent.diagnose(sandbox, scenario.description)
    remed_res = await remed_agent.remediate(sandbox, finding)

    report = await audit.audit(sandbox, scenario, finding, remed_res)
    assert report.verified_healthy is True
    assert report.approved is True
    assert report.revert_recommended is False


@pytest.mark.asyncio
async def test_swarm_coordinator_workflow() -> None:
    """Test full Tri-Agent handoff cycle through the SwarmCoordinator."""
    cfg = EngineConfig()
    cfg.llm.mock_mode = True
    sandbox = MockSandbox("coordinator-test")
    scenario = SystemdDNSScenario()

    await scenario.setup(sandbox)
    await scenario.inject_fault(sandbox)

    coordinator = SwarmCoordinator(config=cfg, max_cycles=2)
    res = await coordinator.run(scenario, sandbox)

    assert res.scenario_name == "systemd_dns"
    assert res.success is True
    assert res.cycles_executed >= 1
    assert res.triage_finding is not None
    assert res.remediation_result is not None
    assert res.audit_report is not None
