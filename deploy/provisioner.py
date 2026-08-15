"""Host pre-flight diagnostic provisioner and systemd service deployment automation."""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import httpx
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

logger = logging.getLogger("os_autofix.deploy.provisioner")
console = Console()


class HostProvisioner:
    """Pre-flight hardware, virtualization, hypervisor, and network diagnostics."""

    def __init__(
        self,
        ollama_url: str = "http://10.0.0.25:11434",
        open_webui_url: str = "https://ai.is27.duckdns.org/api",
    ) -> None:
        self.ollama_url = ollama_url.rstrip("/")
        self.open_webui_url = open_webui_url.rstrip("/")

    def check_kvm(self) -> dict[str, Any]:
        """Verify KVM hardware acceleration support for VM mode."""
        kvm_path = Path("/dev/kvm")
        if not kvm_path.exists():
            return {
                "name": "KVM Hardware Acceleration",
                "status": "WARN",
                "details": "/dev/kvm not found. Container virtualization recommended.",
            }
        is_writable = os.access(kvm_path, os.R_OK | os.W_OK)
        return {
            "name": "KVM Hardware Acceleration",
            "status": "PASS" if is_writable else "WARN",
            "details": "Hardware virtualization active (/dev/kvm accessible)"
            if is_writable
            else "Present but lacking read/write permissions for current user",
        }

    def check_incus(self) -> dict[str, Any]:
        """Check Incus hypervisor CLI and daemon availability."""
        if shutil.which("incus") is None:
            return {
                "name": "Incus Hypervisor CLI",
                "status": "FAIL",
                "details": "Command 'incus' not found in system PATH.",
            }
        try:
            ver = subprocess.check_output(["incus", "--version"], text=True).strip()
            return {
                "name": "Incus Hypervisor CLI",
                "status": "PASS",
                "details": f"Version: {ver}",
            }
        except Exception as e:
            return {
                "name": "Incus Hypervisor CLI",
                "status": "FAIL",
                "details": f"Failed communicating with Incus daemon: {e}",
            }

    def check_storage_pools(self) -> dict[str, Any]:
        """Verify available Incus storage pools (ZFS, Btrfs, dir)."""
        try:
            raw = subprocess.check_output(
                ["incus", "storage", "list", "--format", "json"], text=True
            )
            pools = json.loads(raw)
            if not pools:
                return {
                    "name": "Incus Storage Pools",
                    "status": "WARN",
                    "details": "No storage pools configured in Incus project.",
                }
            pool_names = [f"{p['name']} ({p.get('driver', 'unknown')})" for p in pools]
            return {
                "name": "Incus Storage Pools",
                "status": "PASS",
                "details": f"Configured pools: {', '.join(pool_names)}",
            }
        except Exception as e:
            return {
                "name": "Incus Storage Pools",
                "status": "FAIL",
                "details": f"Storage query failed: {e}",
            }

    def check_network_bridge(self) -> dict[str, Any]:
        """Validate incusbr0 or network bridge routing."""
        try:
            raw = subprocess.check_output(
                ["incus", "network", "list", "--format", "json"], text=True
            )
            nets = json.loads(raw)
            managed = [n["name"] for n in nets if n.get("managed", False)]
            return {
                "name": "Incus Network Bridges",
                "status": "PASS" if managed else "WARN",
                "details": f"Active managed bridges: {', '.join(managed) if managed else 'none'}",
            }
        except Exception as e:
            return {
                "name": "Incus Network Bridges",
                "status": "WARN",
                "details": f"Network bridge query failed: {e}",
            }

    def check_ollama(self) -> dict[str, Any]:
        """Check Ollama REST API reachability."""
        url = f"{self.ollama_url}/api/version"
        try:
            with httpx.Client(timeout=3.0) as client:
                resp = client.get(url)
                if resp.status_code == 200:
                    v = resp.json().get("version", "unknown")
                    return {
                        "name": f"Ollama ({self.ollama_url})",
                        "status": "PASS",
                        "details": f"Online (version {v})",
                    }
                return {
                    "name": f"Ollama ({self.ollama_url})",
                    "status": "WARN",
                    "details": f"HTTP {resp.status_code}",
                }
        except Exception as e:
            return {
                "name": f"Ollama ({self.ollama_url})",
                "status": "FAIL",
                "details": f"Unreachable: {e}",
            }

    def check_open_webui(self) -> dict[str, Any]:
        """Check Open-WebUI endpoint reachability."""
        try:
            with httpx.Client(timeout=3.0, verify=False) as client:
                resp = client.get(self.open_webui_url)
                return {
                    "name": f"Open-WebUI ({self.open_webui_url})",
                    "status": "PASS" if resp.status_code < 500 else "WARN",
                    "details": f"Reachable (status {resp.status_code})",
                }
        except Exception as e:
            return {
                "name": f"Open-WebUI ({self.open_webui_url})",
                "status": "WARN",
                "details": f"Connection check: {e}",
            }

    def run_doctor(self) -> bool:
        """Execute full suite of pre-flight checks and render diagnostic table."""
        console.print(
            Panel.fit(
                "[bold magenta]OS-AutoFix Host & Network Pre-Flight Diagnostics[/bold magenta]"
            )
        )

        checks = [
            self.check_kvm(),
            self.check_incus(),
            self.check_storage_pools(),
            self.check_network_bridge(),
            self.check_ollama(),
            self.check_open_webui(),
        ]

        table = Table(
            title="Pre-Flight Diagnostic Results",
            show_header=True,
            header_style="bold green",
        )
        table.add_column("Component", style="bold cyan")
        table.add_column("Status", justify="center")
        table.add_column("Details")

        all_ok = True
        for c in checks:
            status_color = (
                "green" if c["status"] == "PASS" else "yellow" if c["status"] == "WARN" else "red"
            )
            if c["status"] == "FAIL":
                all_ok = False
            table.add_row(
                c["name"],
                f"[{status_color}]{c['status']}[/{status_color}]",
                c["details"],
            )

        console.print(table)
        if all_ok:
            console.print(
                "[bold green]All essential host and network pre-flight checks passed![/bold green]"
            )
        else:
            console.print(
                "[bold red]One or more pre-flight checks failed! Resolve issues before continuous loop execution.[/bold red]"
            )
        return all_ok

    def install_systemd_services(
        self,
        target_dir: Path | str = "/etc/systemd/system",
        enable_services: bool = False,
    ) -> bool:
        """Install systemd service units to target directory."""
        dest_dir = Path(target_dir)
        source_dir = Path(__file__).parent / "systemd"

        if not source_dir.exists():
            raise FileNotFoundError(f"Systemd source directory not found: {source_dir}")

        dest_dir.mkdir(parents=True, exist_ok=True)
        installed_files: list[Path] = []

        for unit_file in source_dir.glob("*.service"):
            dest_file = dest_dir / unit_file.name
            shutil.copyfile(unit_file, dest_file)
            installed_files.append(dest_file)
            console.print(f"  • Installed [green]{dest_file}[/green]")

        if enable_services and shutil.which("systemctl"):
            try:
                subprocess.run(["systemctl", "daemon-reload"], check=True)
                for f in installed_files:
                    subprocess.run(["systemctl", "enable", f.name], check=True)
                console.print("[bold green]Systemd units enabled successfully![/bold green]")
            except Exception as e:
                logger.warning("Failed running systemctl commands: %s", e)
                return False

        return True
