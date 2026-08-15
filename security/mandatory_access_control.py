"""Mandatory Access Control (MAC) Profile Synthesizer for AppArmor and SELinux."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

logger = logging.getLogger("os_autofix.security.mac")


class MacType(str, Enum):
    APPARMOR = "APPARMOR"
    SELINUX = "SELINUX"


@dataclass
class AppArmorRule:
    """AppArmor path permission or capability rule."""

    target: str  # File path glob or capability name
    permissions: str  # "r", "rw", "mrix", etc. or "capability"


@dataclass
class AppArmorProfile:
    """Synthesized AppArmor security profile."""

    profile_name: str
    binary_path: str
    rules: list[AppArmorRule] = field(default_factory=list)
    flags: list[str] = field(default_factory=lambda: ["attach_disconnected", "complain"])

    def render(self) -> str:
        """Render standard AppArmor profile format."""
        header_flags = f" flags=({', '.join(self.flags)})" if self.flags else ""
        lines = [
            f"# Auto-synthesized by OS-AutoFix MAC Engine for {self.binary_path}",
            f"profile {self.profile_name} {self.binary_path}{header_flags} {{",
            "  #include <tunables/global>",
            "",
            "  # Base POSIX file and library permissions",
            "  /lib/** mr,",
            "  /usr/lib/** mr,",
            "  /etc/ld.so.cache r,",
            "  /dev/null rw,",
            "  /dev/urandom r,",
        ]

        # Add explicit rules
        for rule in self.rules:
            if rule.permissions == "capability":
                lines.append(f"  capability {rule.target},")
            else:
                lines.append(f"  {rule.target} {rule.permissions},")

        lines.append("}")
        return "\n".join(lines) + "\n"


class MacProfileSynthesizer:
    """Passively profiles daemon executions and generates least-privilege MAC security policies."""

    def synthesize_apparmor(
        self,
        binary_path: str,
        profile_name: str | None = None,
        audit_logs: list[str] | None = None,
    ) -> AppArmorProfile:
        """Generate least-privilege AppArmor profile with required capabilities and file paths."""
        name = profile_name or Path(binary_path).name
        profile = AppArmorProfile(
            profile_name=name,
            binary_path=binary_path,
            flags=["attach_disconnected"],
        )

        # Baseline daemon capabilities
        profile.rules.append(AppArmorRule(target="net_bind_service", permissions="capability"))
        profile.rules.append(AppArmorRule(target="setuid", permissions="capability"))
        profile.rules.append(AppArmorRule(target="setgid", permissions="capability"))

        # Baseline paths
        profile.rules.append(AppArmorRule(target="/etc/resolv.conf", permissions="r"))
        profile.rules.append(AppArmorRule(target="/etc/hosts", permissions="r"))
        profile.rules.append(AppArmorRule(target="/etc/ssl/**", permissions="r"))
        profile.rules.append(AppArmorRule(target=f"/var/log/{name}/**", permissions="rw"))
        profile.rules.append(AppArmorRule(target=f"/run/{name}/**", permissions="rw"))

        # Parse audit denials if supplied
        if audit_logs:
            for log_line in audit_logs:
                if 'apparmor="DENIED"' in log_line or "denied" in log_line.lower():
                    # Extract requested_mask or name
                    if "name=" in log_line:
                        try:
                            start = log_line.index('name="') + 6
                            end = log_line.index('"', start)
                            denied_path = log_line[start:end]
                            profile.rules.append(AppArmorRule(target=denied_path, permissions="rw"))
                        except ValueError:
                            pass

        return profile

    def synthesize_selinux_te(
        self,
        module_name: str,
        daemon_name: str,
        allow_rules: list[str] | None = None,
    ) -> str:
        """Generate SELinux Type Enforcement (.te) policy module."""
        rules = allow_rules or [
            f"allow {daemon_name}_t etc_t:file {{ read open getattr }};",
            f"allow {daemon_name}_t net_conf_t:file {{ read open getattr }};",
            f"allow {daemon_name}_t self:capability {{ net_bind_service setuid setgid }};",
        ]

        lines = [
            f"module {module_name} 1.0;",
            "",
            "require {",
            f"    type {daemon_name}_t;",
            "    type etc_t;",
            "    type net_conf_t;",
            "    class file { read open getattr };",
            "    class capability { net_bind_service setuid setgid };",
            "}",
            "",
            "# Rule declarations",
        ]
        lines.extend(rules)
        return "\n".join(lines) + "\n"
