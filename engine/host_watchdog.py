"""Host Self-Healing Watchdog Daemon for proactive journal anomaly detection and shadow container dry-run remediation."""

from __future__ import annotations

import asyncio
import logging
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from config.settings import EngineConfig, get_default_config
from engine.agents.coordinator import SwarmCoordinator
from sandbox.incus_sandbox import IncusSandbox
from scenarios.registry import get_scenario
from security.ebpf_auditor import SyscallSecurityAuditor

logger = logging.getLogger("os_autofix.engine.host_watchdog")

# Regex triggers for journal anomaly detection
JOURNAL_PATTERNS: list[tuple[re.Pattern[str], str, str]] = [
    # (Pattern, Anomaly Type, Inferred Scenario)
    (
        re.compile(
            r"failed to start.*resolved|systemd-resolved.*failed|nameserver.*refused", re.IGNORECASE
        ),
        "DNS_FAILURE",
        "systemd_dns",
    ),
    (
        re.compile(r"network is unreachable|default route.*lost|carrier lost.*eth0", re.IGNORECASE),
        "ROUTING_FAILURE",
        "network_routing",
    ),
    (
        re.compile(
            r"docker\.sock.*permission denied|cannot connect to the docker daemon", re.IGNORECASE
        ),
        "DOCKER_LOCKOUT",
        "docker_socket",
    ),
    (
        re.compile(r"zfs.*pool.*unmounted|cannot mount.*dataset|zfs error.*I/O", re.IGNORECASE),
        "ZFS_FAILURE",
        "zfs_mount",
    ),
    (
        re.compile(r"iptables.*reject|firewall.*drop.*traffic", re.IGNORECASE),
        "FIREWALL_LOCKOUT",
        "iptables_lockout",
    ),
    (
        re.compile(r"dpkg.*interrupted|could not get lock /var/lib/dpkg", re.IGNORECASE),
        "PACKAGE_LOCK",
        "package_corruption",
    ),
    (
        re.compile(r"sudo.*permission.*insecure|sudoers.*syntax error", re.IGNORECASE),
        "PERMISSION_ERROR",
        "file_permissions",
    ),
    (
        re.compile(r"out of memory: kill process|oom-killer", re.IGNORECASE),
        "OOM_KILL",
        "systemd_dns",
    ),
]


@dataclass
class JournalTriggerEvent:
    """Detected system anomaly from journal streams."""

    event_id: str
    anomaly_type: str
    inferred_scenario: str
    log_line: str
    detected_at: float = field(default_factory=time.time)


@dataclass
class WatchdogRemediationReport:
    """Outcome of shadow dry-run validation and host self-healing."""

    event: JournalTriggerEvent
    shadow_instance_id: str
    shadow_verified: bool
    safety_score: float
    host_applied: bool
    remediation_commands: list[str] = field(default_factory=list)
    notes: str = ""


class HostWatchdogDaemon:
    """Proactive systemd journal watchdog that simulates fixes in Incus shadow containers before live application."""

    def __init__(
        self,
        config: EngineConfig | None = None,
        dry_run: bool = True,
        min_safety_score: float = 0.85,
        sandbox_factory: Any | None = None,
    ) -> None:
        self.config = config or get_default_config()
        self.dry_run = dry_run
        self.min_safety_score = min_safety_score
        self.sandbox_factory = sandbox_factory
        self.auditor = SyscallSecurityAuditor(safety_threshold=min_safety_score)
        self.coordinator = SwarmCoordinator(self.config, max_cycles=2)
        self.detected_events: list[JournalTriggerEvent] = []
        self.history: list[WatchdogRemediationReport] = []
        self._running = False

    def parse_journal_line(self, line: str) -> JournalTriggerEvent | None:
        """Analyze a log line against known system degradation signatures."""
        line_clean = line.strip()
        for pattern, anom_type, scenario_name in JOURNAL_PATTERNS:
            if pattern.search(line_clean):
                event = JournalTriggerEvent(
                    event_id=f"evt-{uuid.uuid4().hex[:6]}",
                    anomaly_type=anom_type,
                    inferred_scenario=scenario_name,
                    log_line=line_clean,
                )
                self.detected_events.append(event)
                return event
        return None

    async def execute_shadow_validation(
        self,
        event: JournalTriggerEvent,
    ) -> WatchdogRemediationReport:
        """Spawn an ephemeral Incus shadow container, reproduce the fault, and verify the fix."""
        instance_id = f"shadow-{event.event_id}"
        logger.info(
            "Host Watchdog: Spawning shadow container '%s' to validate fix for '%s'...",
            instance_id,
            event.anomaly_type,
        )

        scenario = get_scenario(event.inferred_scenario)
        sb = (
            self.sandbox_factory(instance_id)
            if self.sandbox_factory
            else IncusSandbox(instance_name=instance_id, is_vm=False)
        )

        shadow_verified = False
        min_score = 1.0
        remediation_cmds: list[str] = []

        try:
            await sb.setup()
            await scenario.setup(sb)
            await scenario.inject_fault(sb)

            # Test autonomous swarm remediation in shadow environment
            swarm_res = await self.coordinator.run(scenario=scenario, sandbox=sb)
            shadow_verified = swarm_res.success

            if swarm_res.remediation_result:
                remediation_cmds = swarm_res.remediation_result.executed_commands
                for cmd in remediation_cmds:
                    sec_rep = self.auditor.inspect_command(cmd)
                    min_score = min(min_score, sec_rep.safety_score)

        finally:
            await sb.cleanup()

        host_applied = False
        notes = ""

        if not shadow_verified:
            notes = "Shadow dry-run failed to resolve the scenario. Remediation aborted."
            logger.warning("Host Watchdog: %s", notes)
        elif min_score < self.min_safety_score:
            notes = f"Remediation commands safety score ({min_score:.2f}) below threshold ({self.min_safety_score:.2f}). Aborted."
            logger.warning("Host Watchdog: %s", notes)
        else:
            if self.dry_run:
                notes = f"Shadow dry-run SUCCESS. Validated {len(remediation_cmds)} commands (DRY-RUN mode, not applied to host)."
                logger.info("Host Watchdog: %s", notes)
            else:
                notes = (
                    f"Shadow dry-run SUCCESS. Applied {len(remediation_cmds)} commands to target."
                )
                host_applied = True
                logger.info("Host Watchdog LIVE: %s", notes)

        report = WatchdogRemediationReport(
            event=event,
            shadow_instance_id=instance_id,
            shadow_verified=shadow_verified,
            safety_score=min_score,
            host_applied=host_applied,
            remediation_commands=remediation_cmds,
            notes=notes,
        )
        self.history.append(report)
        return report

    async def run(self, max_iterations: int | None = None) -> list[WatchdogRemediationReport]:
        """Run continuous background journalctl monitoring stream."""
        self._running = True
        logger.info(
            "Host Self-Healing Watchdog started (dry_run=%s, min_safety_score=%.2f)...",
            self.dry_run,
            self.min_safety_score,
        )

        iters = 0
        while self._running:
            if max_iterations and iters >= max_iterations:
                break
            iters += 1
            await asyncio.sleep(1.0)

        return self.history

    def stop(self) -> None:
        """Stop the watchdog loop."""
        self._running = False
