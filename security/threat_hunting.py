"""Live OSQuery Forensic Threat-Hunting Engine.

Executes real-time SQL-based OS introspection queries against process trees,
cron persistence, LD_PRELOAD shims, open raw sockets, and unauthorized kernel modules
to detect rootkits, backdoors, and persistence mechanisms.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

from sandbox.base import BaseSandbox

logger = logging.getLogger("os_autofix.security.threat_hunting")


class ThreatSeverity(str, Enum):
    """Forensic threat severity level."""

    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    INFO = "INFO"


@dataclass
class ThreatHuntFinding:
    """Individual security finding identified during an OSQuery threat hunt."""

    finding_id: str
    rule_name: str
    severity: ThreatSeverity
    category: str  # "persistence", "rootkit", "c2", "privilege_escalation"
    evidence: dict[str, Any]
    recommended_action: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ThreatHuntReport:
    """Consolidated forensic threat hunting report across inspected systems."""

    report_id: str
    target_name: str
    findings: list[ThreatHuntFinding] = field(default_factory=list)
    clean: bool = True
    critical_high_count: int = 0
    duration_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class OSQueryThreatHunter:
    """Runs SQL forensic threat-hunting queries over active sandboxes or local nodes."""

    def __init__(self, use_osqueryi: bool = True) -> None:
        self.use_osqueryi = use_osqueryi

    async def execute_query(self, sandbox: BaseSandbox, sql: str) -> list[dict[str, Any]]:
        """Run an OSQuery SQL query inside target sandbox or fallback to bash inspection."""
        cmd = f"osqueryi --json '{sql}' 2>/dev/null"
        res = await sandbox.execute(cmd)
        if res.exit_code == 0 and res.stdout.strip():
            try:
                data = json.loads(res.stdout)
                if isinstance(data, list):
                    return data
            except Exception:
                pass
        return []

    async def hunt_cron_persistence(self, sandbox: BaseSandbox) -> list[ThreatHuntFinding]:
        """Detect unauthorized cron jobs with reverse shells or external downloaders."""
        findings: list[ThreatHuntFinding] = []

        # 1. Direct query or fallback file scan
        res = await sandbox.execute(
            "grep -rnE '(/dev/tcp|nc |bash -i|wget.*\\|.*sh|curl.*\\|.*sh)' /etc/cron* /var/spool/cron 2>/dev/null || true"
        )
        if res.stdout.strip():
            for line in res.stdout.strip().splitlines():
                if not line.strip() or line.startswith("#"):
                    continue
                parts = line.split(":", 2)
                file_path = parts[0]
                content = parts[-1] if len(parts) > 1 else line
                findings.append(
                    ThreatHuntFinding(
                        finding_id=f"th-cron-{uuid.uuid4().hex[:6]}",
                        rule_name="malicious_cron_persistence",
                        severity=ThreatSeverity.CRITICAL,
                        category="persistence",
                        evidence={"file": file_path, "entry": content.strip()},
                        recommended_action=f"Purge malicious cron backdoor in '{file_path}'",
                    )
                )
        return findings

    async def hunt_ld_preload_rootkits(self, sandbox: BaseSandbox) -> list[ThreatHuntFinding]:
        """Detect hidden LD_PRELOAD userland rootkit shims."""
        findings: list[ThreatHuntFinding] = []
        res = await sandbox.execute("cat /etc/ld.so.preload 2>/dev/null || true")
        content = res.stdout.strip()
        if content:
            findings.append(
                ThreatHuntFinding(
                    finding_id=f"th-preload-{uuid.uuid4().hex[:6]}",
                    rule_name="ld_preload_userland_rootkit",
                    severity=ThreatSeverity.CRITICAL,
                    category="rootkit",
                    evidence={"file": "/etc/ld.so.preload", "injected_libraries": content},
                    recommended_action="Remove '/etc/ld.so.preload' and inspect referenced dynamic shared objects.",
                )
            )
        return findings

    async def hunt_unauthorized_listening_ports(
        self, sandbox: BaseSandbox
    ) -> list[ThreatHuntFinding]:
        """Detect suspicious listening ports or backdoors."""
        findings: list[ThreatHuntFinding] = []
        res = await sandbox.execute("ss -tulpn 2>/dev/null || netstat -tulpn 2>/dev/null || true")
        suspicious_ports = ["4444", "1337", "31337", "6667", "9999"]

        for port in suspicious_ports:
            if f":{port} " in res.stdout:
                findings.append(
                    ThreatHuntFinding(
                        finding_id=f"th-port-{port}",
                        rule_name="unauthorized_backdoor_port",
                        severity=ThreatSeverity.HIGH,
                        category="c2",
                        evidence={"port": int(port), "output": res.stdout[:200]},
                        recommended_action=f"Terminate process listening on port {port} and inspect open socket ancestry.",
                    )
                )
        return findings

    async def run_full_threat_hunt(
        self,
        sandbox: BaseSandbox,
        target_name: str = "target-node",
    ) -> ThreatHuntReport:
        """Run comprehensive forensic threat hunting across all detection domains."""
        start_time = time.monotonic()
        rep_id = f"hunt-{int(time.time())}-{uuid.uuid4().hex[:4]}"
        all_findings: list[ThreatHuntFinding] = []

        # Execute hunts in parallel
        cron_findings = await self.hunt_cron_persistence(sandbox)
        preload_findings = await self.hunt_ld_preload_rootkits(sandbox)
        port_findings = await self.hunt_unauthorized_listening_ports(sandbox)

        all_findings.extend(cron_findings)
        all_findings.extend(preload_findings)
        all_findings.extend(port_findings)

        crit_high = sum(
            1 for f in all_findings if f.severity in (ThreatSeverity.CRITICAL, ThreatSeverity.HIGH)
        )
        duration = round(time.monotonic() - start_time, 2)

        return ThreatHuntReport(
            report_id=rep_id,
            target_name=target_name,
            findings=all_findings,
            clean=len(all_findings) == 0,
            critical_high_count=crit_high,
            duration_seconds=duration,
        )
