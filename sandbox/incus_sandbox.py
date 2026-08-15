"""Production-grade Incus Sandbox driver for VM and container environments."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import shlex
import time
import uuid
from typing import Any

from config.settings import IncusConfig
from sandbox.base import BaseSandbox, ExecutionResult

logger = logging.getLogger("os_autofix.sandbox.incus")


class IncusSandboxError(Exception):
    """Base exception for Incus sandbox operations."""

    pass


class IncusAgentTimeoutError(IncusSandboxError):
    """Raised when incus-agent does not become responsive within timeout."""

    pass


class IncusSandbox(BaseSandbox):
    """Full Incus virtualization driver supporting both VM and container modes.

    Provides sub-second snapshotting, guest agent readiness verification,
    asynchronous execution with stream truncation and strict timeouts.
    """

    def __init__(
        self,
        instance_name: str | None = None,
        config: IncusConfig | None = None,
        image: str | None = None,
        is_vm: bool | None = None,
    ) -> None:
        self.config = config or IncusConfig()
        unique_suffix = uuid.uuid4().hex[:8]
        self.instance_name = instance_name or f"{self.config.instance_prefix}-{unique_suffix}"
        self.image = image or self.config.default_image
        self.is_vm = is_vm if is_vm is not None else self.config.is_vm
        self.project = self.config.project
        self.storage_pool = self.config.storage_pool
        self._is_ready = False
        self._snapshots: list[str] = []

    async def _run_incus_cli(
        self,
        args: list[str],
        timeout_seconds: float = 30.0,
        check: bool = True,
    ) -> tuple[int, str, str]:
        """Execute a local incus CLI command asynchronously."""
        cmd = ["incus", *args, "--project", self.project]
        cmd_str = " ".join(shlex.quote(arg) for arg in cmd)
        logger.debug("Executing host incus command: %s", cmd_str)

        try:
            proc = await asyncio.create_subprocess_exec(
                cmd[0],
                *cmd[1:],
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(),
                timeout=timeout_seconds,
            )
            stdout = stdout_bytes.decode("utf-8", errors="replace").strip()
            stderr = stderr_bytes.decode("utf-8", errors="replace").strip()
            exit_code = proc.returncode if proc.returncode is not None else -1

            if check and exit_code != 0:
                error_msg = (
                    f"Incus command failed (code {exit_code}): {cmd_str}\n"
                    f"Stderr: {stderr}\n"
                    f"Stdout: {stdout}"
                )
                logger.error(error_msg)
                raise IncusSandboxError(error_msg)

            return exit_code, stdout, stderr

        except asyncio.TimeoutError:
            try:
                proc.kill()
                await proc.wait()
            except Exception:
                pass
            raise IncusSandboxError(
                f"Incus CLI command timed out after {timeout_seconds}s: {cmd_str}"
            ) from None
        except Exception as e:
            if not isinstance(e, IncusSandboxError):
                raise IncusSandboxError(f"Failed to execute incus command '{cmd_str}': {e}") from e
            raise

    async def setup(self) -> None:
        """Launch the instance and wait for guest agent readiness."""
        logger.info(
            "Launching Incus instance '%s' (image=%s, vm=%s, pool=%s)...",
            self.instance_name,
            self.image,
            self.is_vm,
            self.storage_pool,
        )

        launch_args = ["launch", self.image, self.instance_name]
        if self.is_vm:
            launch_args.append("--vm")
        if self.storage_pool:
            launch_args.extend(["-s", self.storage_pool])
        if self.config.cpu_limit:
            launch_args.extend(["-c", f"limits.cpu={self.config.cpu_limit}"])
        if self.config.memory_limit:
            launch_args.extend(["-c", f"limits.memory={self.config.memory_limit}"])

        await self._run_incus_cli(launch_args, timeout_seconds=120.0)
        await self.wait_until_ready()
        self._is_ready = True
        logger.info("Incus instance '%s' is verified and ready for execution.", self.instance_name)

    async def wait_until_ready(self, timeout_seconds: int | None = None) -> None:
        """Poll and verify incus-agent / guest responsiveness."""
        timeout = timeout_seconds or self.config.agent_wait_timeout_seconds
        interval = self.config.agent_poll_interval_seconds
        start_time = time.monotonic()

        logger.debug(
            "Polling guest agent readiness for '%s' (timeout: %ds)...",
            self.instance_name,
            timeout,
        )

        while time.monotonic() - start_time < timeout:
            try:
                exit_code, stdout, _ = await self._run_incus_cli(
                    ["exec", self.instance_name, "--", "echo", "__READY__"],
                    timeout_seconds=5.0,
                    check=False,
                )
                if exit_code == 0 and "__READY__" in stdout:
                    elapsed = time.monotonic() - start_time
                    logger.debug(
                        "Instance '%s' guest agent responsive after %.2fs",
                        self.instance_name,
                        elapsed,
                    )
                    return
            except Exception as e:
                logger.debug("Agent poll failed on '%s': %s", self.instance_name, e)

            await asyncio.sleep(interval)

        raise IncusAgentTimeoutError(
            f"Guest agent on instance '{self.instance_name}' failed to become responsive "
            f"within {timeout} seconds."
        )

    async def create_snapshot(self, snapshot_name: str) -> None:
        """Create a point-in-time snapshot with zero-copy COW on ZFS/btrfs."""
        logger.info(
            "Creating snapshot '%s' for instance '%s'...", snapshot_name, self.instance_name
        )
        await self._run_incus_cli(["snapshot", "create", self.instance_name, snapshot_name])
        self._snapshots.append(snapshot_name)

    async def revert(self, snapshot_name: str) -> None:
        """Roll back the instance to a previously captured snapshot state."""
        logger.info(
            "Reverting instance '%s' to snapshot '%s'...", self.instance_name, snapshot_name
        )
        await self._run_incus_cli(["snapshot", "restore", self.instance_name, snapshot_name])
        await self.wait_until_ready(timeout_seconds=15)
        logger.info(
            "Revert of '%s' to '%s' completed successfully.", self.instance_name, snapshot_name
        )

    async def execute(
        self,
        command: str,
        timeout_seconds: int | None = None,
    ) -> ExecutionResult:
        """Execute a shell command inside the guest with truncation and timeout guarantees."""
        timeout = timeout_seconds or self.config.command_timeout_seconds
        max_chars = self.config.max_output_chars

        incus_cmd = [
            "incus",
            "exec",
            self.instance_name,
            "--project",
            self.project,
            "--",
            "/bin/bash",
            "-c",
            command,
        ]

        logger.debug("Guest exec on '%s' (timeout: %ds): %s", self.instance_name, timeout, command)
        start_time = time.monotonic()

        try:
            proc = await asyncio.create_subprocess_exec(
                incus_cmd[0],
                *incus_cmd[1:],
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    proc.communicate(),
                    timeout=float(timeout),
                )
                duration = time.monotonic() - start_time
                stdout_raw = stdout_bytes.decode("utf-8", errors="replace")
                stderr_raw = stderr_bytes.decode("utf-8", errors="replace")
                exit_code = proc.returncode if proc.returncode is not None else -1
                timed_out = False

            except asyncio.TimeoutError:
                duration = time.monotonic() - start_time
                logger.warning(
                    "Command timed out on '%s' after %.2fs: %s",
                    self.instance_name,
                    duration,
                    command,
                )
                try:
                    proc.kill()
                    await proc.wait()
                except Exception:
                    pass
                return ExecutionResult(
                    command=command,
                    exit_code=124,
                    stdout="",
                    stderr=f"[ERROR] Command timed out after {timeout} seconds.",
                    duration_seconds=duration,
                    truncated=False,
                    timed_out=True,
                )

            # Strict 2000-character truncation
            truncated = False
            stdout = stdout_raw
            stderr = stderr_raw

            if len(stdout) > max_chars:
                stdout = (
                    stdout[:max_chars]
                    + f"\n\n[... STDOUT TRUNCATED: Exceeded {max_chars} characters ...]"
                )
                truncated = True

            if len(stderr) > max_chars:
                stderr = (
                    stderr[:max_chars]
                    + f"\n\n[... STDERR TRUNCATED: Exceeded {max_chars} characters ...]"
                )
                truncated = True

            return ExecutionResult(
                command=command,
                exit_code=exit_code,
                stdout=stdout,
                stderr=stderr,
                duration_seconds=duration,
                truncated=truncated,
                timed_out=timed_out,
            )

        except Exception as e:
            duration = time.monotonic() - start_time
            logger.error("Execution error on '%s': %s", self.instance_name, e)
            return ExecutionResult(
                command=command,
                exit_code=1,
                stdout="",
                stderr=f"[FATAL HOST ERROR] Failed to execute in guest: {e}",
                duration_seconds=duration,
                truncated=False,
                timed_out=False,
            )

    async def get_state(self) -> dict[str, Any]:
        """Query and return instance status and network information."""
        exit_code, stdout, _ = await self._run_incus_cli(
            ["info", self.instance_name, "--format", "json"],
            check=False,
        )
        if exit_code == 0:
            try:
                info_data = json.loads(stdout)
                return {
                    "name": self.instance_name,
                    "status": info_data.get("status", "UNKNOWN"),
                    "type": info_data.get("type", "UNKNOWN"),
                    "architecture": info_data.get("architecture", "UNKNOWN"),
                    "created_at": info_data.get("created_at", ""),
                    "state": info_data.get("state", {}),
                    "snapshots": self._snapshots,
                }
            except Exception as e:
                logger.warning("Failed to parse instance info json: %s", e)

        return {
            "name": self.instance_name,
            "status": "RUNNING" if self._is_ready else "UNKNOWN",
            "is_vm": self.is_vm,
            "snapshots": self._snapshots,
        }

    async def read_file(self, remote_path: str) -> str:
        """Read content of a file from the guest filesystem."""
        result = await self.execute(f"cat {shlex.quote(remote_path)}")
        if not result.success:
            raise IncusSandboxError(f"Failed to read file '{remote_path}': {result.stderr}")
        return result.stdout

    async def write_file(self, remote_path: str, content: str, mode: str | None = None) -> None:
        """Write content to a file inside the guest filesystem."""
        b64_content = base64.b64encode(content.encode("utf-8")).decode("ascii")
        cmd = f"echo {shlex.quote(b64_content)} | base64 -d > {shlex.quote(remote_path)}"
        if mode:
            cmd += f" && chmod {shlex.quote(mode)} {shlex.quote(remote_path)}"

        result = await self.execute(cmd)
        if not result.success:
            raise IncusSandboxError(f"Failed to write file '{remote_path}': {result.stderr}")

    async def cleanup(self) -> None:
        """Terminate and permanently delete the instance."""
        logger.info("Cleaning up and deleting Incus instance '%s'...", self.instance_name)
        try:
            await self._run_incus_cli(
                ["delete", "--force", self.instance_name],
                timeout_seconds=30.0,
                check=False,
            )
            self._is_ready = False
            logger.info("Instance '%s' deleted successfully.", self.instance_name)
        except Exception as e:
            logger.warning("Error deleting instance '%s': %s", self.instance_name, e)
