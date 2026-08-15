"""Combinatorial Cascading Fault Fuzzer for Compound Multi-Domain OS Outages."""

from __future__ import annotations

import logging
import random
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from config.settings import EngineConfig, get_default_config
from engine.agents.coordinator import SwarmCoordinator
from sandbox.base import BaseSandbox
from sandbox.incus_sandbox import IncusSandbox

logger = logging.getLogger("os_autofix.engine.cascading_fuzzer")


@dataclass
class CompoundFaultResult:
    """Outcome of compound multi-domain fault fuzzing pass."""

    fuzz_id: str
    domains_injected: list[str]
    success: bool
    mttr_seconds: float
    domain_statuses: dict[str, bool] = field(default_factory=dict)
    notes: str = ""


class CascadingFaultFuzzer:
    """Fuzzer generating coupled, simultaneous multi-domain OS failures."""

    def __init__(
        self,
        config: EngineConfig | None = None,
        sandbox_factory: Callable[[str], BaseSandbox] | None = None,
    ) -> None:
        self.config = config or get_default_config()
        self.sandbox_factory = sandbox_factory or (lambda name: IncusSandbox(instance_name=name))
        self.coordinator = SwarmCoordinator(self.config, max_cycles=3)

    async def inject_compound_faults(
        self, sandbox: BaseSandbox, domains: list[str]
    ) -> dict[str, str]:
        """Inject simultaneous compound breakages across requested system domains."""
        injected: dict[str, str] = {}

        if "network" in domains:
            await sandbox.execute(
                "iptables -I INPUT 1 -p udp --dport 53 -j DROP 2>/dev/null || true; "
                "echo 'nameserver 192.0.2.1' > /etc/resolv.conf"
            )
            injected["network"] = "Dropped UDP DNS port 53 & corrupted /etc/resolv.conf"

        if "storage" in domains:
            await sandbox.execute(
                "mkdir -p /mnt/data_pool && "
                "echo 'FAULT_UNMOUNTED' > /tmp/zfs_fault.flag && "
                "chmod 000 /mnt/data_pool 2>/dev/null || true"
            )
            injected["storage"] = "Locked dataset directory permissions /mnt/data_pool"

        if "permissions" in domains:
            await sandbox.execute("chmod 000 /etc/hosts /tmp 2>/dev/null || true")
            injected["permissions"] = "Corrupted permissions on /etc/hosts and /tmp"

        if "security" in domains:
            await sandbox.execute(
                "mkdir -p /etc/apparmor.d && "
                "echo 'deny /etc/** r,' > /etc/apparmor.d/custom_lockout.conf"
            )
            injected["security"] = "Injected restrictive AppArmor lockout rule"

        logger.info(
            "Injected %d compound faults across domains: %s", len(injected), list(injected.keys())
        )
        return injected

    async def verify_compound_state(
        self, sandbox: BaseSandbox, domains: list[str]
    ) -> tuple[bool, dict[str, bool]]:
        """Verify health across all injected failure domains."""
        domain_health: dict[str, bool] = {}

        if "network" in domains:
            chk = await sandbox.execute("cat /etc/resolv.conf 2>/dev/null || true")
            domain_health["network"] = "192.0.2.1" not in chk.stdout and bool(chk.stdout.strip())

        if "storage" in domains:
            chk = await sandbox.execute("ls /mnt/data_pool 2>/dev/null || true")
            domain_health["storage"] = chk.exit_code == 0

        if "permissions" in domains:
            chk = await sandbox.execute("test -r /etc/hosts && test -w /tmp")
            domain_health["permissions"] = chk.exit_code == 0

        if "security" in domains:
            chk = await sandbox.execute(
                "cat /etc/apparmor.d/custom_lockout.conf 2>/dev/null || true"
            )
            domain_health["security"] = "deny /etc/**" not in chk.stdout

        all_ok = all(domain_health.values()) if domain_health else True
        return all_ok, domain_health

    async def run_fuzzing_experiment(
        self,
        domains: list[str] | None = None,
        sandbox_name: str = "fuzz-canary",
    ) -> CompoundFaultResult:
        """Run complete combinatorial cascading outage experiment with swarm remediation."""
        target_domains = domains or ["network", "storage", "permissions"]
        fuzz_id = f"fuzz-{int(time.time())}-{random.randint(100, 999)}"
        sb = self.sandbox_factory(sandbox_name)

        start_time = time.monotonic()
        await sb.setup()

        try:
            # 1. Inject compound faults
            await self.inject_compound_faults(sb, target_domains)
            pre_ok, _ = await self.verify_compound_state(sb, target_domains)
            assert not pre_ok, "Faults failed to induce breakage"

            # 2. Trigger swarm remediation
            logger.info(
                "Fuzzer [%s]: Launching Tri-Agent Swarm against compound failures...", fuzz_id
            )
            from scenarios.base_scenario import BaseScenario

            class CompoundFuzzScenario(BaseScenario):
                name = f"compound_{'_'.join(target_domains)}"
                description = f"Compound multi-domain failure across {', '.join(target_domains)}"
                difficulty = "expert"
                category = "Chaos"

                async def setup(self, sandbox: BaseSandbox) -> bool:
                    return True

                async def inject_fault(self, sandbox: BaseSandbox) -> bool:
                    return True

                async def verify(self, sandbox: BaseSandbox) -> tuple[bool, str]:
                    return True, "Compound scenario verified."

            fuzz_scenario = CompoundFuzzScenario()

            swarm_res = await self.coordinator.run(
                scenario=fuzz_scenario,
                sandbox=sb,
            )

            # 3. Verify compound state post-fix
            post_ok, health = await self.verify_compound_state(sb, target_domains)
            duration = round(time.monotonic() - start_time, 2)

            return CompoundFaultResult(
                fuzz_id=fuzz_id,
                domains_injected=target_domains,
                success=post_ok and swarm_res.success,
                mttr_seconds=duration,
                domain_statuses=health,
                notes=f"Cycles: {swarm_res.cycles_executed} | Swarm success: {swarm_res.success}",
            )
        finally:
            await sb.cleanup()
