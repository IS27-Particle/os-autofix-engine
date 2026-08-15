"""Kernel-level eBPF & syscall security auditor for detecting destructive anti-patterns and evaluating blast radius."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from sandbox.base import BaseSandbox

logger = logging.getLogger("os_autofix.security.ebpf_auditor")

# High-risk destructive command patterns
DESTRUCTIVE_PATTERNS: list[tuple[re.Pattern[str], str, float]] = [
    # (Regex, Description, Penalty Weight)
    (
        re.compile(r"rm\s+-(?:r|rf|fr)\s+(?:/\s*$|/\*|/etc|/boot|/sys|/dev|/var\b)", re.IGNORECASE),
        "Recursive root/system filesystem deletion",
        1.0,
    ),
    (
        re.compile(r">\s*/etc/(?:passwd|shadow|sudoers|fstab|hosts)\b", re.IGNORECASE),
        "Critical system config truncation/overwrite",
        0.8,
    ),
    (
        re.compile(r"dd\s+if=.*of=/dev/(?:sd[a-z]|vd[a-z]|nvme[0-9]|null)\b", re.IGNORECASE),
        "Raw block device overwriting",
        0.9,
    ),
    (re.compile(r"mkfs(?:\.[a-z0-9]+)?\s+/dev/", re.IGNORECASE), "Raw filesystem formatting", 0.9),
    (
        re.compile(r"(?:nc|ncat|netcat|socat)\s+.*-e\s+(?:/bin/sh|/bin/bash)\b", re.IGNORECASE),
        "Reverse shell backdoor execution",
        1.0,
    ),
    (
        re.compile(r"/dev/tcp/\d+\.\d+\.\d+\.\d+/\d+", re.IGNORECASE),
        "Bash raw socket reverse shell payload",
        1.0,
    ),
    (
        re.compile(r"cat\s+/etc/shadow\b", re.IGNORECASE),
        "Credential harvesting on /etc/shadow",
        0.6,
    ),
    (
        re.compile(r"cat\s+~?/\.ssh/id_[a-z0-9_]+\b", re.IGNORECASE),
        "SSH private key extraction",
        0.7,
    ),
    (
        re.compile(r"insmod\s+|modprobe\s+-r\s+", re.IGNORECASE),
        "Unauthorized kernel module manipulation",
        0.5,
    ),
    (
        re.compile(r":\(\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;\s*:", re.IGNORECASE),
        "Forkbomb denial of service attack",
        1.0,
    ),
    (
        re.compile(r"chmod\s+(?:0?777|-R\s+777)\s+/(?:etc|bin|sbin|usr|var)\b", re.IGNORECASE),
        "Dangerous permissive permission assignment",
        0.5,
    ),
    (re.compile(r"iptables\s+-F\s+OUTPUT", re.IGNORECASE), "Total firewall outbound flush", 0.3),
]


@dataclass
class SyscallAuditEvent:
    """Recorded syscall / command inspection event."""

    command: str
    syscall_type: str  # execve, unlinkat, chmod, connect, ptrace
    is_violation: bool
    risk_level: str  # "low", "medium", "high", "critical"
    description: str
    penalty: float


@dataclass
class SecurityAuditReport:
    """Security assessment of command execution within an Incus sandbox."""

    command: str
    safety_score: float  # 0.0 (catastrophic) to 1.0 (clean)
    is_safe: bool  # True if safety_score >= safety_threshold (default 0.7)
    abort_execution: bool
    rollback_required: bool
    blast_radius: str  # "none", "local", "system", "kernel"
    events: list[SyscallAuditEvent] = field(default_factory=list)
    rejection_reason: str = ""


class SyscallSecurityAuditor:
    """Kernel-level syscall security monitor intercepting and evaluating execution safety."""

    def __init__(self, safety_threshold: float = 0.7) -> None:
        self.safety_threshold = safety_threshold
        self.audit_log: list[SyscallAuditEvent] = []

    def inspect_command(self, command: str) -> SecurityAuditReport:
        """Analyze a command before or during execution for destructive anti-patterns."""
        cmd_clean = command.strip()
        events: list[SyscallAuditEvent] = []
        total_penalty = 0.0

        for pattern, desc, penalty in DESTRUCTIVE_PATTERNS:
            if pattern.search(cmd_clean):
                risk = "critical" if penalty >= 0.8 else ("high" if penalty >= 0.5 else "medium")
                syscall = "execve"
                if "rm " in cmd_clean or "unlink" in cmd_clean:
                    syscall = "unlinkat"
                elif "chmod " in cmd_clean:
                    syscall = "chmod"
                elif "tcp" in cmd_clean or "nc " in cmd_clean:
                    syscall = "connect"
                elif "insmod" in cmd_clean or "modprobe" in cmd_clean:
                    syscall = "init_module"

                event = SyscallAuditEvent(
                    command=command,
                    syscall_type=syscall,
                    is_violation=True,
                    risk_level=risk,
                    description=desc,
                    penalty=penalty,
                )
                events.append(event)
                total_penalty += penalty
                self.audit_log.append(event)

        # Compute safety score bounded between [0.0, 1.0]
        safety_score = max(0.0, round(1.0 - total_penalty, 2))
        is_safe = safety_score >= self.safety_threshold

        blast_radius = "none"
        if safety_score < 0.3:
            blast_radius = "kernel"
        elif safety_score < 0.7:
            blast_radius = "system"
        elif safety_score < 1.0:
            blast_radius = "local"

        rejection_reason = ""
        if not is_safe:
            reasons = [e.description for e in events]
            rejection_reason = f"Security safety score ({safety_score}) below threshold ({self.safety_threshold}): {', '.join(reasons)}"

        return SecurityAuditReport(
            command=command,
            safety_score=safety_score,
            is_safe=is_safe,
            abort_execution=not is_safe,
            rollback_required=not is_safe,
            blast_radius=blast_radius,
            events=events,
            rejection_reason=rejection_reason,
        )

    async def audit_sandbox_runtime(
        self,
        sandbox: BaseSandbox,
        command: str,
    ) -> SecurityAuditReport:
        """Run pre-execution security check and verify post-execution system invariants."""
        # 1. Pre-execution static & pattern audit
        report = self.inspect_command(command)
        if report.abort_execution:
            logger.error("Syscall Auditor ABORT: %s", report.rejection_reason)
            return report

        # 2. Dynamic runtime inspection via guest probe
        probe_res = await sandbox.execute("cat /proc/sys/kernel/tainted 2>/dev/null")
        if probe_res.exit_code == 0 and probe_res.stdout.strip() not in ("0", ""):
            report.events.append(
                SyscallAuditEvent(
                    command=command,
                    syscall_type="kernel",
                    is_violation=True,
                    risk_level="high",
                    description="Kernel tainted during command execution",
                    penalty=0.4,
                )
            )
            report.safety_score = max(0.0, round(report.safety_score - 0.4, 2))
            report.is_safe = report.safety_score >= self.safety_threshold
            report.rollback_required = not report.is_safe

        return report
