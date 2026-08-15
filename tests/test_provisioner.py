"""Unit tests for host provisioner pre-flight checks and systemd installation."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx

from deploy.provisioner import HostProvisioner


def test_host_provisioner_kvm_check() -> None:
    """Test KVM detection logic."""
    provisioner = HostProvisioner()
    res = provisioner.check_kvm()
    assert res["status"] in ("PASS", "WARN", "FAIL")
    assert "KVM" in res["name"]


def test_host_provisioner_incus_check() -> None:
    """Test Incus CLI version parsing."""
    provisioner = HostProvisioner()
    with patch("subprocess.check_output", return_value="6.23\n"):
        with patch("shutil.which", return_value="/usr/bin/incus"):
            res = provisioner.check_incus()
            assert res["status"] == "PASS"
            assert "6.23" in res["details"]


def test_host_provisioner_storage_pools() -> None:
    """Test storage pools query and driver detection."""
    provisioner = HostProvisioner()
    mock_pools = [
        {"name": "default", "driver": "zfs"},
        {"name": "custom", "driver": "btrfs"},
    ]
    with patch("subprocess.check_output", return_value=json.dumps(mock_pools)):
        res = provisioner.check_storage_pools()
        assert res["status"] == "PASS"
        assert "default (zfs)" in res["details"]


def test_host_provisioner_network_bridges() -> None:
    """Test Incus network bridge list parsing."""
    provisioner = HostProvisioner()
    mock_nets = [
        {"name": "incusbr0", "managed": True},
        {"name": "eth0", "managed": False},
    ]
    with patch("subprocess.check_output", return_value=json.dumps(mock_nets)):
        res = provisioner.check_network_bridge()
        assert res["status"] == "PASS"
        assert "incusbr0" in res["details"]


def test_host_provisioner_ollama_check() -> None:
    """Test Ollama version query check."""
    provisioner = HostProvisioner(ollama_url="http://10.0.0.25:11434")
    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"version": "0.32.9"}

    with patch("httpx.Client.get", return_value=mock_resp):
        res = provisioner.check_ollama()
        assert res["status"] == "PASS"
        assert "0.32.9" in res["details"]


def test_systemd_installation(tmp_path: Path) -> None:
    """Test copying systemd unit files to target directory."""
    provisioner = HostProvisioner()
    target_dir = tmp_path / "systemd"

    ok = provisioner.install_systemd_services(target_dir=target_dir, enable_services=False)
    assert ok is True
    assert (target_dir / "os-autofix.service").exists()
    assert (target_dir / "os-autofix-metrics.service").exists()

    content = (target_dir / "os-autofix.service").read_text(encoding="utf-8")
    assert "ExecStart" in content
