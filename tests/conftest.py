"""Pytest configuration, fixtures, and mock sandbox implementations."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pytest

from config.settings import EngineConfig, IncusConfig, LLMConfig
from sandbox.base import BaseSandbox, ExecutionResult
from trainer.trajectory_buffer import TrajectoryBuffer


class MockSandbox(BaseSandbox):
    """In-memory mock sandbox simulating shell execution, files, and snapshot rollbacks."""

    def __init__(self, name: str = "mock-sandbox") -> None:
        self.name = name
        self.is_setup = False
        self.is_cleaned = False
        self.files: dict[str, str] = {
            "/etc/sudoers": "# sudoers default\nroot ALL=(ALL) ALL",
            "/etc/resolv.conf": "nameserver 1.1.1.1\nnameserver 8.8.8.8",
            "/etc/ssh/sshd_config": "# sshd config\nPort 22",
        }
        self.file_perms: dict[str, str] = {
            "/etc/sudoers": "440",
            "/etc/ssh/sshd_config": "600",
        }
        self.snapshots: dict[str, dict[str, Any]] = {}
        self.command_history: list[str] = []
        self.dns_working = True
        self.routing_working = True
        self.dpkg_working = True

    async def setup(self) -> None:
        self.is_setup = True

    async def create_snapshot(self, snapshot_name: str) -> None:
        self.snapshots[snapshot_name] = {
            "files": dict(self.files),
            "file_perms": dict(self.file_perms),
            "dns_working": self.dns_working,
            "routing_working": self.routing_working,
            "dpkg_working": self.dpkg_working,
        }

    async def revert(self, snapshot_name: str) -> None:
        if snapshot_name not in self.snapshots:
            raise KeyError(f"Snapshot '{snapshot_name}' does not exist")
        snap = self.snapshots[snapshot_name]
        self.files = dict(snap["files"])
        self.file_perms = dict(snap["file_perms"])
        self.dns_working = snap["dns_working"]
        self.routing_working = snap["routing_working"]
        self.dpkg_working = snap["dpkg_working"]

    async def execute(
        self,
        command: str,
        timeout_seconds: int | None = None,
    ) -> ExecutionResult:
        self.command_history.append(command)
        cmd_clean = command.strip()

        # 1. Verification scripts
        if "ROUTING_HEALTHY" in cmd_clean:
            if self.routing_working:
                return ExecutionResult(
                    command=command, exit_code=0, stdout="ROUTING_HEALTHY\n", stderr=""
                )
            return ExecutionResult(
                command=command, exit_code=1, stdout="NO_VALID_DEFAULT_ROUTE\n", stderr=""
            )

        if "DNS_RESOLVED_OK" in cmd_clean or "dns.google" in cmd_clean:
            if self.dns_working:
                return ExecutionResult(
                    command=command, exit_code=0, stdout="DNS_RESOLVED_OK\n", stderr=""
                )
            return ExecutionResult(
                command=command,
                exit_code=1,
                stdout="DNS_ERROR: Temporary failure in name resolution\n",
                stderr="",
            )

        if "PACKAGE_MANAGER_HEALTHY" in cmd_clean:
            if self.dpkg_working:
                return ExecutionResult(
                    command=command, exit_code=0, stdout="PACKAGE_MANAGER_HEALTHY\n", stderr=""
                )
            return ExecutionResult(
                command=command, exit_code=1, stdout="DPKG_AUDIT_FAILED\n", stderr=""
            )

        if "PERMISSIONS_VALID" in cmd_clean:
            current_perm = self.file_perms.get("/etc/sudoers", "440")
            if current_perm not in ("777", "666", "775"):
                return ExecutionResult(
                    command=command, exit_code=0, stdout="PERMISSIONS_VALID\n", stderr=""
                )
            return ExecutionResult(
                command=command, exit_code=1, stdout="SUDOERS_INSECURE_PERMS: 777\n", stderr=""
            )

        # 2. DNS operations
        if "127.0.0.99" in cmd_clean or "stop systemd-resolved" in cmd_clean:
            self.files["/etc/resolv.conf"] = "nameserver 127.0.0.99"
            self.dns_working = False
            return ExecutionResult(command=command, exit_code=0, stdout="", stderr="")

        if "systemctl restart systemd-resolved" in cmd_clean or "nameserver 1.1.1.1" in cmd_clean:
            self.files["/etc/resolv.conf"] = "nameserver 1.1.1.1"
            self.dns_working = True
            return ExecutionResult(
                command=command, exit_code=0, stdout="systemd-resolved restarted", stderr=""
            )

        if "getent ahosts" in cmd_clean or "one.one.one.one" in cmd_clean:
            if self.dns_working:
                return ExecutionResult(
                    command=command,
                    exit_code=0,
                    stdout="1.1.1.1 STREAM one.one.one.one\n",
                    stderr="",
                )
            return ExecutionResult(command=command, exit_code=2, stdout="", stderr="Unknown host")

        # 3. Network routing operations
        if "ip route del default" in cmd_clean or "192.0.2.254" in cmd_clean:
            self.routing_working = False
            return ExecutionResult(command=command, exit_code=0, stdout="", stderr="")

        if (
            "ip route replace default" in cmd_clean
            or "ip route add default" in cmd_clean
            or "dhclient" in cmd_clean
        ):
            self.routing_working = True
            return ExecutionResult(command=command, exit_code=0, stdout="", stderr="")

        # 4. Package manager operations
        if "touch /var/lib/dpkg/lock-frontend" in cmd_clean or "autofix-dummy" in cmd_clean:
            self.dpkg_working = False
            return ExecutionResult(command=command, exit_code=0, stdout="", stderr="")

        if "rm -f /var/lib/dpkg/lock" in cmd_clean or "dpkg --configure -a" in cmd_clean:
            self.dpkg_working = True
            return ExecutionResult(command=command, exit_code=0, stdout="", stderr="")

        # 5. Sudoers / file permissions operations
        if "chmod 0777 /etc/sudoers" in cmd_clean or "chmod 777 /etc/sudoers" in cmd_clean:
            self.file_perms["/etc/sudoers"] = "777"
            return ExecutionResult(command=command, exit_code=0, stdout="", stderr="")

        if "chmod 0440 /etc/sudoers" in cmd_clean or "chmod 440 /etc/sudoers" in cmd_clean:
            self.file_perms["/etc/sudoers"] = "440"
            return ExecutionResult(command=command, exit_code=0, stdout="", stderr="")

        # 6. ZFS / Filesystem mount operations
        if "umount /mnt/data" in cmd_clean or "umount -l /mnt/data" in cmd_clean:
            self.files.pop("/mnt/data/critical_dataset.txt", None)
            return ExecutionResult(command=command, exit_code=0, stdout="", stderr="")

        if "mount " in cmd_clean and "/mnt/data" in cmd_clean:
            self.files["/mnt/data/critical_dataset.txt"] = "CRITICAL_DATASET_PAYLOAD_VALID_2026\n"
            return ExecutionResult(command=command, exit_code=0, stdout="", stderr="")

        if "mountpoint -q /mnt/data" in cmd_clean:
            if "/mnt/data/critical_dataset.txt" in self.files:
                return ExecutionResult(command=command, exit_code=0, stdout="", stderr="")
            return ExecutionResult(command=command, exit_code=1, stdout="", stderr="")

        # 7. Docker socket operations
        if "chmod 0000 /var/run/docker.sock" in cmd_clean:
            self.file_perms["/var/run/docker.sock"] = "000"
            return ExecutionResult(command=command, exit_code=0, stdout="", stderr="")

        if (
            "chmod 0660 /var/run/docker.sock" in cmd_clean
            or "chmod 666 /var/run/docker.sock" in cmd_clean
        ):
            self.file_perms["/var/run/docker.sock"] = "660"
            return ExecutionResult(command=command, exit_code=0, stdout="", stderr="")

        if "test -e /var/run/docker.sock" in cmd_clean:
            perm = self.file_perms.get("/var/run/docker.sock", "660")
            if perm in ("660", "666", "777"):
                return ExecutionResult(command=command, exit_code=0, stdout="", stderr="")
            return ExecutionResult(command=command, exit_code=1, stdout="", stderr="")

        # 8. IPTables / Firewall operations
        if "iptables -I OUTPUT" in cmd_clean and "DROP" in cmd_clean:
            self.files["/etc/iptables/lock_status.conf"] = "FIREWALL_LOCKED=1"
            self.files["/etc/iptables/rules"] = "-A OUTPUT -p udp --dport 53 -j DROP"
            return ExecutionResult(command=command, exit_code=0, stdout="", stderr="")

        if "iptables -F" in cmd_clean or "iptables -D" in cmd_clean:
            self.files.pop("/etc/iptables/lock_status.conf", None)
            self.files.pop("/etc/iptables/rules", None)
            return ExecutionResult(command=command, exit_code=0, stdout="", stderr="")

        if "iptables -S OUTPUT" in cmd_clean or "iptables -L OUTPUT" in cmd_clean:
            if "/etc/iptables/rules" in self.files:
                return ExecutionResult(
                    command=command,
                    exit_code=0,
                    stdout="-P OUTPUT ACCEPT\n-A OUTPUT -p udp --dport 53 -j DROP\n",
                    stderr="",
                )
            return ExecutionResult(
                command=command, exit_code=0, stdout="-P OUTPUT ACCEPT\n", stderr=""
            )

        # 9. File reading
        if cmd_clean.startswith("cat "):
            path = cmd_clean.split("cat ", 1)[1].strip().strip("'").strip('"')
            path = path.split(" 2>")[0].strip()
            if path in self.files:
                return ExecutionResult(
                    command=command, exit_code=0, stdout=self.files[path], stderr=""
                )
            return ExecutionResult(
                command=command,
                exit_code=1,
                stdout="",
                stderr=f"cat: {path}: No such file or directory",
            )

        # 10. Fallback success
        return ExecutionResult(command=command, exit_code=0, stdout="OK", stderr="")

    async def get_state(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": "RUNNING" if self.is_setup and not self.is_cleaned else "STOPPED",
            "snapshots": list(self.snapshots.keys()),
        }

    async def read_file(self, remote_path: str) -> str:
        if remote_path not in self.files:
            raise FileNotFoundError(f"File '{remote_path}' not found in mock sandbox")
        return self.files[remote_path]

    async def write_file(self, remote_path: str, content: str, mode: str | None = None) -> None:
        self.files[remote_path] = content
        if mode:
            self.file_perms[remote_path] = mode

    async def cleanup(self) -> None:
        self.is_cleaned = True


@pytest.fixture
def mock_sandbox() -> MockSandbox:
    """Fixture providing an in-memory MockSandbox instance."""
    return MockSandbox()


@pytest.fixture
def engine_config(tmp_path: Path) -> EngineConfig:
    """Fixture providing an EngineConfig pointing to temporary paths."""
    base_dir = tmp_path / "test_engine"
    base_dir.mkdir(parents=True, exist_ok=True)
    return EngineConfig(
        workers=2,
        max_steps_per_episode=5,
        data_dir=base_dir / "data",
        logs_dir=base_dir / "logs",
        llm=LLMConfig(mock_mode=True),
        incus=IncusConfig(instance_prefix="pytest-test"),
    )


@pytest.fixture
def trajectory_buffer() -> TrajectoryBuffer:
    """Fixture providing a fresh TrajectoryBuffer."""
    return TrajectoryBuffer()


def is_incus_available() -> bool:
    """Check whether incus CLI is installed and responsive on the test host."""
    return shutil.which("incus") is not None


incus_live_mark = pytest.mark.skipif(
    not is_incus_available(),
    reason="Incus daemon/CLI not available on host",
)
