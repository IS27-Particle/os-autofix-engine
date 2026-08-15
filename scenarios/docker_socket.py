"""Diagnostic scenario for Docker daemon socket permission lockouts and service overrides."""

from __future__ import annotations

import logging

from sandbox.base import BaseSandbox
from scenarios.base_scenario import BaseScenario

logger = logging.getLogger("os_autofix.scenarios.docker_socket")


class DockerSocketScenario(BaseScenario):
    """Diagnose and resolve permission lockouts and service failures on /var/run/docker.sock."""

    name: str = "docker_socket"
    description: str = (
        "Docker daemon socket (/var/run/docker.sock) has corrupted permissions or is inaccessible. "
        "Container management commands report permission denied connecting to the Docker daemon."
    )
    category: str = "Containers / Docker"
    difficulty: str = "medium"
    max_steps: int = 6

    async def setup(self, sandbox: BaseSandbox) -> bool:
        """Setup baseline docker socket stub and group."""
        logger.info("Setting up baseline docker environment for %s...", self.name)
        cmds = [
            "groupadd -f docker",
            "mkdir -p /var/run /var/run/docker",
            "touch /var/run/docker.sock",
            "chmod 0660 /var/run/docker.sock",
            "chown root:docker /var/run/docker.sock 2>/dev/null || true",
        ]
        for cmd in cmds:
            await sandbox.execute(cmd)
        return True

    async def inject_fault(self, sandbox: BaseSandbox) -> bool:
        """Inject 0000 permission lockout on /var/run/docker.sock."""
        logger.info("Injecting permission lockout on /var/run/docker.sock...")
        cmds = [
            "chmod 0000 /var/run/docker.sock",
            "chown root:root /var/run/docker.sock",
        ]
        for cmd in cmds:
            await sandbox.execute(cmd)
        return True

    async def verify(self, sandbox: BaseSandbox) -> tuple[bool, str]:
        """Verify /var/run/docker.sock is accessible with read/write permissions."""
        res = await sandbox.execute(
            "test -e /var/run/docker.sock && test -r /var/run/docker.sock && test -w /var/run/docker.sock"
        )
        if res.exit_code == 0:
            return (
                True,
                "Docker daemon socket /var/run/docker.sock permissions verified accessible.",
            )

        perm_res = await sandbox.execute("stat -c '%a %U:%G' /var/run/docker.sock 2>/dev/null")
        return (
            False,
            f"Docker socket is not accessible. Current stat: '{perm_res.stdout.strip()}'",
        )
