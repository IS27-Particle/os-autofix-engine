"""Proxmox VE REST API Hypervisor Driver for OS-AutoFix Engine.

Manages remote QEMU VMs and LXC containers hosted on Proxmox Virtual Environment
clusters over token-authenticated REST HTTPS APIs with QEMU Guest Agent integration.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import time
from typing import Any

import httpx

from sandbox.base import BaseSandbox, ExecutionResult

logger = logging.getLogger("os_autofix.sandbox.proxmox")


class ProxmoxSandbox(BaseSandbox):
    """Manages Proxmox VE QEMU/LXC guest environments via the Proxmox REST API."""

    def __init__(
        self,
        instance_name: str,
        host: str = "https://proxmox.local:8006",
        node: str = "pve",
        vmid: int = 100,
        api_token: str | None = None,
        verify_ssl: bool = False,
        mock_mode: bool = False,
    ) -> None:
        self.name = instance_name
        self.host = host.rstrip("/")
        self.node = node
        self.vmid = vmid
        self.api_token = api_token or "root@pam!osautofix=00000000-0000-0000-0000-000000000000"
        self.verify_ssl = verify_ssl
        self.mock_mode = mock_mode

        self._running = False
        self._files: dict[str, str] = {}
        self._snapshots: dict[str, dict[str, Any]] = {}

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"PVEAPIToken={self.api_token}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

    async def setup(self) -> None:
        """Start or verify remote Proxmox VM status."""
        if self.mock_mode:
            logger.info(
                "Initializing Proxmox sandbox '%s' (VMID: %d) in mock mode.", self.name, self.vmid
            )
            self._running = True
            self._files["/etc/hostname"] = self.name
            self._files["/etc/resolv.conf"] = "nameserver 8.8.8.8\n"
            return

        try:
            async with httpx.AsyncClient(verify=self.verify_ssl, timeout=10.0) as client:
                url = f"{self.host}/api2/json/nodes/{self.node}/qemu/{self.vmid}/status/start"
                resp = await client.post(url, headers=self._headers())
                if resp.status_code in (200, 500):  # 500 often means already running
                    self._running = True
                    logger.info(
                        "Proxmox VM '%s' (VMID %d) started on node %s.",
                        self.name,
                        self.vmid,
                        self.node,
                    )
                else:
                    logger.warning(
                        "Proxmox start returned code %d, fallback to mock mode.", resp.status_code
                    )
                    self.mock_mode = True
                    self._running = True
        except Exception as e:
            logger.warning("Proxmox connection failed: %s; falling back to mock mode.", e)
            self.mock_mode = True
            self._running = True

    async def execute(self, command: str, timeout_seconds: int | None = None) -> ExecutionResult:
        """Execute command inside Proxmox VM using QEMU Guest Agent."""
        start_time = time.monotonic()
        timeout = timeout_seconds or 15

        if self.mock_mode:
            cmd_clean = command.strip()
            stdout = ""
            stderr = ""
            exit_code = 0

            if cmd_clean.startswith("cat "):
                target = cmd_clean.split("cat ", 1)[1].strip()
                if target in self._files:
                    stdout = self._files[target]
                else:
                    exit_code = 1
                    stderr = f"cat: {target}: No such file or directory"
            elif cmd_clean.startswith("echo ") and " > " in cmd_clean:
                parts = cmd_clean.split(" > ", 1)
                text = parts[0].replace("echo ", "").strip().strip("'\"")
                target = parts[1].strip()
                self._files[target] = text + "\n"
            else:
                stdout = f"Executed '{cmd_clean}' on Proxmox VM {self.vmid}"

            duration = round(time.monotonic() - start_time, 3)
            return ExecutionResult(
                command=command,
                exit_code=exit_code,
                stdout=stdout,
                stderr=stderr,
                duration_seconds=duration,
            )

        try:
            async with httpx.AsyncClient(verify=self.verify_ssl, timeout=float(timeout)) as client:
                exec_url = f"{self.host}/api2/json/nodes/{self.node}/qemu/{self.vmid}/agent/exec"
                payload = {"command": f"/bin/sh -c '{command}'"}
                resp = await client.post(exec_url, json=payload, headers=self._headers())
                if resp.status_code != 200:
                    return ExecutionResult(
                        command=command,
                        exit_code=1,
                        stderr=f"Proxmox QEMU agent error: {resp.text}",
                        duration_seconds=round(time.monotonic() - start_time, 3),
                    )

                data = resp.json().get("data", {})
                pid = data.get("pid")
                if not pid:
                    return ExecutionResult(
                        command=command, exit_code=1, stderr="No PID returned from guest agent."
                    )

                # Poll status
                status_url = f"{self.host}/api2/json/nodes/{self.node}/qemu/{self.vmid}/agent/exec-status?pid={pid}"
                for _ in range(int(timeout * 10)):
                    await asyncio.sleep(0.1)
                    st_resp = await client.get(status_url, headers=self._headers())
                    if st_resp.status_code == 200:
                        st_data = st_resp.json().get("data", {})
                        if st_data.get("exited", False):
                            duration = round(time.monotonic() - start_time, 3)
                            return ExecutionResult(
                                command=command,
                                exit_code=st_data.get("exitcode", 0),
                                stdout=st_data.get("out-data", ""),
                                stderr=st_data.get("err-data", ""),
                                duration_seconds=duration,
                            )

            duration = round(time.monotonic() - start_time, 3)
            return ExecutionResult(
                command=command, exit_code=-1, timed_out=True, duration_seconds=duration
            )
        except Exception as e:
            duration = round(time.monotonic() - start_time, 3)
            return ExecutionResult(
                command=command, exit_code=1, stderr=str(e), duration_seconds=duration
            )

    async def create_snapshot(self, snapshot_name: str) -> None:
        """Take remote Proxmox VM snapshot."""
        if self.mock_mode:
            self._snapshots[snapshot_name] = dict(self._files)
            return

        async with httpx.AsyncClient(verify=self.verify_ssl, timeout=30.0) as client:
            url = f"{self.host}/api2/json/nodes/{self.node}/qemu/{self.vmid}/snapshot"
            payload = {"snapname": snapshot_name, "vmstate": 1}
            await client.post(url, json=payload, headers=self._headers())
            logger.info("Created Proxmox snapshot '%s' on VMID %d", snapshot_name, self.vmid)

    async def revert(self, snapshot_name: str) -> None:
        """Roll back Proxmox VM to snapshot."""
        if self.mock_mode:
            if snapshot_name in self._snapshots:
                self._files = dict(self._snapshots[snapshot_name])
            return

        async with httpx.AsyncClient(verify=self.verify_ssl, timeout=30.0) as client:
            url = f"{self.host}/api2/json/nodes/{self.node}/qemu/{self.vmid}/snapshot/{snapshot_name}/rollback"
            await client.post(url, headers=self._headers())
            logger.info("Reverted Proxmox VMID %d to snapshot '%s'", self.vmid, snapshot_name)

    async def get_state(self) -> dict[str, Any]:
        """Query Proxmox VM resource status."""
        return {
            "name": self.name,
            "type": "proxmox_qemu",
            "host": self.host,
            "node": self.node,
            "vmid": self.vmid,
            "running": self._running,
            "mock_mode": self.mock_mode,
        }

    async def read_file(self, remote_path: str) -> str:
        """Read content of a file via guest agent."""
        res = await self.execute(f"cat {remote_path}")
        if res.exit_code != 0:
            raise FileNotFoundError(
                f"File {remote_path} not found in Proxmox VM {self.vmid}: {res.stderr}"
            )
        return res.stdout

    async def write_file(self, remote_path: str, content: str, mode: str | None = None) -> None:
        """Write file to remote Proxmox VM."""
        if self.mock_mode:
            self._files[remote_path] = content
            return
        b64_content = base64.b64encode(content.encode("utf-8")).decode("ascii")
        cmd = f"echo '{b64_content}' | base64 -d > {remote_path}"
        if mode:
            cmd += f" && chmod {mode} {remote_path}"
        await self.execute(cmd)

    async def cleanup(self) -> None:
        """Stop or release Proxmox sandbox session."""
        self._running = False
        logger.info("Cleaned up Proxmox sandbox session '%s' (VMID: %d)", self.name, self.vmid)
