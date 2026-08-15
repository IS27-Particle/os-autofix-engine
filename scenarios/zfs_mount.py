"""Diagnostic scenario for broken or unmounted filesystem datasets."""

from __future__ import annotations

import logging

from sandbox.base import BaseSandbox
from scenarios.base_scenario import BaseScenario

logger = logging.getLogger("os_autofix.scenarios.zfs_mount")

VERIFICATION_PAYLOAD = "CRITICAL_DATASET_PAYLOAD_VALID_2026"


class ZFSMountScenario(BaseScenario):
    """Diagnose and restore corrupted or unmounted filesystem dataset mountpoints."""

    name: str = "zfs_mount"
    description: str = (
        "Critical storage dataset mountpoint (/mnt/data) is unmounted or inaccessible. "
        "Production applications report missing database storage."
    )
    category: str = "Storage / Filesystems"
    difficulty: str = "medium"
    max_steps: int = 8

    async def setup(self, sandbox: BaseSandbox) -> bool:
        """Create test storage backing file, mountpoint, and dataset payload."""
        logger.info("Setting up baseline dataset for %s...", self.name)
        cmds = [
            "mkdir -p /mnt/data /var/lib/storage_backing",
            "dd if=/dev/zero of=/var/lib/storage_backing/disk.img bs=1M count=10 2>/dev/null",
            "mkfs.ext4 -F /var/lib/storage_backing/disk.img >/dev/null 2>&1 || true",
            "mount -o loop /var/lib/storage_backing/disk.img /mnt/data 2>/dev/null || true",
            f"echo '{VERIFICATION_PAYLOAD}' > /mnt/data/critical_dataset.txt",
        ]
        for cmd in cmds:
            await sandbox.execute(cmd)
        return True

    async def inject_fault(self, sandbox: BaseSandbox) -> bool:
        """Unmount dataset and inject fstab / mount corruption."""
        logger.info("Injecting mount fault into /mnt/data...")
        cmds = [
            "umount /mnt/data 2>/dev/null || true",
            "umount -l /mnt/data 2>/dev/null || true",
            "mkdir -p /mnt/data",
            "echo 'MOUNT_FAILURE_PLACEHOLDER' > /mnt/data/.corrupted_placeholder",
        ]
        for cmd in cmds:
            await sandbox.execute(cmd)
        return True

    async def verify(self, sandbox: BaseSandbox) -> tuple[bool, str]:
        """Verify /mnt/data is mounted and critical dataset payload is readable."""
        res_read = await sandbox.execute("cat /mnt/data/critical_dataset.txt 2>/dev/null")
        if VERIFICATION_PAYLOAD in res_read.stdout:
            return True, "Filesystem dataset /mnt/data mounted and verified intact."

        res_mount = await sandbox.execute(
            "mountpoint -q /mnt/data || grep -qs '/mnt/data ' /proc/mounts"
        )
        if res_mount.exit_code != 0:
            return (
                False,
                f"Mount verification failed: /mnt/data is not mounted. Exit code: {res_mount.exit_code}",
            )

        return (
            False,
            f"Mountpoint active but critical data unreadable: '{res_read.stdout.strip()}'",
        )
