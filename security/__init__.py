"""Security, eBPF syscall safety auditing, and approval gate package."""

from security.approval_gate import (
    GLOBAL_APPROVAL_GATE,
    ApprovalDecision,
    ApprovalRequest,
    ApprovalStatus,
    InteractiveApprovalGate,
)
from security.ebpf_auditor import (
    SecurityAuditReport,
    SyscallAuditEvent,
    SyscallSecurityAuditor,
)

__all__ = [
    "SyscallSecurityAuditor",
    "SecurityAuditReport",
    "SyscallAuditEvent",
    "InteractiveApprovalGate",
    "ApprovalRequest",
    "ApprovalDecision",
    "ApprovalStatus",
    "GLOBAL_APPROVAL_GATE",
]
