"""Self-Supervised Differential State Shadow Engine.

Executes twin sandboxes (Primary vs Shadow) to evaluate differential state changes,
asserting zero regression and computing State Equivalence Divergence Scores
prior to production promotion.
"""

from __future__ import annotations

import hashlib
import logging
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

logger = logging.getLogger("os_autofix.engine.shadow_evaluator")


@dataclass
class DifferentialMetrics:
    """Quantitative differential metrics comparing Primary and Shadow instances."""

    fs_hash_matches: int = 0
    fs_hash_divergences: int = 0
    socket_status_matches: int = 0
    socket_status_divergences: int = 0
    memory_rss_delta_mb: float = 0.0
    probed_endpoints_total: int = 0
    primary_success_rate: float = 1.0
    shadow_success_rate: float = 0.0


@dataclass
class DifferentialStateReport:
    """Consolidated report of a differential state evaluation."""

    evaluation_id: str
    scenario_name: str
    passed: bool
    divergence_score: float  # 0.0 = perfect, 1.0 = total divergence/regression
    primary_instance: str
    shadow_instance: str
    metrics: DifferentialMetrics
    divergent_files: list[str] = field(default_factory=list)
    promoted: bool = False
    duration_seconds: float = 0.0
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_mermaid(self) -> str:
        """Render a Mermaid diagram showing differential twin sandbox states."""
        p_stat = "Healthy (Promoted)" if self.promoted else "Failed/Diverged"
        return f"""```mermaid
graph TD
    Baseline[Baseline Snapshot] --> Primary["Primary Instance: {self.primary_instance}\\nState: {p_stat}"]
    Baseline --> Shadow["Shadow Control: {self.shadow_instance}\\nState: Faulted Baseline"]
    Primary --> Diff["Differential Divergence: {self.divergence_score:.2%}"]
    Shadow --> Diff
    Diff --> Decision{{"Promote Fix?"}}
    Decision -->|{"Yes" if self.promoted else "No"}| Action[{"Promote to Fleet" if self.promoted else "Hold & Rollback"}]
```"""


class ShadowEvaluator:
    """Evaluates differential state divergence between Primary and Shadow instances."""

    DEFAULT_PROBE_FILES = [
        "/etc/resolv.conf",
        "/etc/hosts",
        "/etc/sudoers",
        "/etc/ssh/sshd_config",
        "/etc/fstab",
    ]

    def __init__(
        self,
        config: EngineConfig | None = None,
        sandbox_factory: Callable[[str], BaseSandbox] | None = None,
        max_divergence_threshold: float = 0.05,
    ) -> None:
        self.config = config or EngineConfig()
        self.sandbox_factory = sandbox_factory or (
            lambda name: IncusSandbox(name, self.config.incus)
        )
        self.max_divergence_threshold = max_divergence_threshold
        self.coordinator = SwarmCoordinator(self.config)

    async def compute_file_hash(self, sandbox: BaseSandbox, path: str) -> str | None:
        """Compute SHA-256 hash of a target file inside the sandbox."""
        res = await sandbox.execute(f"sha256sum {path} 2>/dev/null || true")
        if res.exit_code == 0 and res.stdout.strip():
            parts = res.stdout.strip().split()
            if parts:
                return parts[0]
        # Fallback via cat
        cat_res = await sandbox.execute(f"cat {path} 2>/dev/null || true")
        if cat_res.stdout:
            return hashlib.sha256(cat_res.stdout.encode("utf-8")).hexdigest()
        return None

    async def probe_sockets(self, sandbox: BaseSandbox) -> set[str]:
        """Collect active listening sockets on the target sandbox."""
        res = await sandbox.execute("ss -tulpn 2>/dev/null || netstat -tulpn 2>/dev/null || true")
        sockets: set[str] = set()
        for line in res.stdout.splitlines():
            line_str = line.strip()
            if "LISTEN" in line_str or "udp" in line_str:
                parts = line_str.split()
                if len(parts) >= 5:
                    sockets.add(parts[4])
        return sockets

    async def get_rss_memory_mb(self, sandbox: BaseSandbox) -> float:
        """Extract total RSS memory consumption in MB."""
        res = await sandbox.execute("ps -eo rss --no-headers 2>/dev/null || true")
        total_kb = 0.0
        for line in res.stdout.splitlines():
            try:
                total_kb += float(line.strip())
            except ValueError:
                continue
        return round(total_kb / 1024.0, 2)

    async def evaluate_differential(
        self,
        scenario: BaseScenario,
        primary_sb: BaseSandbox,
        shadow_sb: BaseSandbox,
        probe_files: list[str] | None = None,
    ) -> DifferentialStateReport:
        """Execute twin differential state evaluation between Primary and Shadow sandboxes."""
        eval_id = f"diff-{int(time.time())}-{uuid.uuid4().hex[:4]}"
        start_time = time.monotonic()
        target_files = probe_files or self.DEFAULT_PROBE_FILES

        p_name = getattr(primary_sb, "instance_name", getattr(primary_sb, "name", "primary"))
        s_name = getattr(shadow_sb, "instance_name", getattr(shadow_sb, "name", "shadow"))

        logger.info(
            "Differential Shadow Engine [%s]: Initializing twin evaluation for '%s' (%s vs %s)",
            eval_id,
            scenario.name,
            p_name,
            s_name,
        )

        # 1. Setup baseline & fault injection on both instances
        await scenario.setup(primary_sb)
        await scenario.setup(shadow_sb)
        await scenario.inject_fault(primary_sb)
        await scenario.inject_fault(shadow_sb)

        # Create identical baseline snapshot
        snap_name = f"snap-shadow-base-{eval_id}"
        await primary_sb.create_snapshot(snap_name)
        await shadow_sb.create_snapshot(snap_name)

        # Verify both instances are broken at baseline
        p_pre_ok, _ = await scenario.verify(primary_sb)
        s_pre_ok, _ = await scenario.verify(shadow_sb)
        assert not p_pre_ok and not s_pre_ok, "Baseline fault injection failed on twin instances."

        # 2. Run Tri-Agent Swarm remediation ONLY on Primary
        logger.info("Differential Shadow Engine: Triggering remediation on Primary instance...")
        swarm_res = await self.coordinator.run(scenario=scenario, sandbox=primary_sb)

        # 3. Post-remediation verification
        p_post_ok, p_msg = await scenario.verify(primary_sb)
        s_post_ok, _ = await scenario.verify(shadow_sb)

        # Shadow control MUST remain faulted (no magic fix)
        # Primary MUST be healthy and verified
        primary_resolved = p_post_ok and swarm_res.success
        shadow_still_faulted = not s_post_ok

        # 4. Probe and compare differential states
        fs_matches = 0
        fs_divergences = 0
        divergent_files: list[str] = []

        for filepath in target_files:
            h_p = await self.compute_file_hash(primary_sb, filepath)
            h_s = await self.compute_file_hash(shadow_sb, filepath)
            if h_p == h_s:
                fs_matches += 1
            else:
                fs_divergences += 1
                divergent_files.append(filepath)

        # Probe sockets
        sock_p = await self.probe_sockets(primary_sb)
        sock_s = await self.probe_sockets(shadow_sb)
        sock_matches = len(sock_p.intersection(sock_s))
        sock_divergences = len(sock_p.symmetric_difference(sock_s))

        # Memory delta
        mem_p = await self.get_rss_memory_mb(primary_sb)
        mem_s = await self.get_rss_memory_mb(shadow_sb)
        mem_delta = round(abs(mem_p - mem_s), 2)

        # 5. Compute State Equivalence Divergence Score
        total_probes = len(target_files) + max(len(sock_p), 1)
        raw_divergence = 0.0
        if not primary_resolved:
            raw_divergence += 0.50
        if not shadow_still_faulted:
            raw_divergence += 0.30
        if mem_delta > 200.0:  # Excessive memory bloat
            raw_divergence += 0.20

        divergence_score = min(round(raw_divergence, 4), 1.0)
        passed = (
            primary_resolved
            and shadow_still_faulted
            and divergence_score <= self.max_divergence_threshold
        )
        promoted = passed

        duration = round(time.monotonic() - start_time, 2)
        metrics = DifferentialMetrics(
            fs_hash_matches=fs_matches,
            fs_hash_divergences=fs_divergences,
            socket_status_matches=sock_matches,
            socket_status_divergences=sock_divergences,
            memory_rss_delta_mb=mem_delta,
            probed_endpoints_total=total_probes,
            primary_success_rate=1.0 if primary_resolved else 0.0,
            shadow_success_rate=0.0 if shadow_still_faulted else 1.0,
        )

        return DifferentialStateReport(
            evaluation_id=eval_id,
            scenario_name=scenario.name,
            passed=passed,
            divergence_score=divergence_score,
            primary_instance=p_name,
            shadow_instance=s_name,
            metrics=metrics,
            divergent_files=divergent_files,
            promoted=promoted,
            duration_seconds=duration,
            notes=f"Primary verification: {p_msg} | Swarm success: {swarm_res.success}",
        )

    async def run_shadow_comparison(
        self,
        scenario: BaseScenario,
        primary_name: str = "canary-primary",
        shadow_name: str = "canary-shadow",
    ) -> DifferentialStateReport:
        """Manage full sandbox lifecycle and run differential shadow evaluation."""
        sb_primary = self.sandbox_factory(primary_name)
        sb_shadow = self.sandbox_factory(shadow_name)

        await sb_primary.setup()
        await sb_shadow.setup()

        try:
            return await self.evaluate_differential(
                scenario=scenario,
                primary_sb=sb_primary,
                shadow_sb=sb_shadow,
            )
        finally:
            await sb_primary.cleanup()
            await sb_shadow.cleanup()
