# ZFS Storage & Dataset Mount Recovery Runbook

## Overview
ZFS storage pools and datasets may fail to mount automatically on boot or become unmounted due to missing cache files, corrupted properties, or export states.

## Common Root Causes
1. Datasets have `canmount=off` or `mountpoint=none` set erroneously.
2. The pool was exported and not imported on boot (`zpool import -a`).
3. Underlying loop or block device path changed.

## Diagnostic Steps
```bash
# Check pool status and health
zpool status -x
zpool list

# List all datasets and their mountpoints
zfs list -o name,mountpoint,mounted,canmount
```

## Remediation Commands
```bash
# 1. Force import degraded or unimported pools
zpool import -f -a

# 2. Fix dataset mountpoint and canmount properties
zfs set canmount=on <pool>/<dataset>
zfs set mountpoint=/mnt/data <pool>/<dataset>

# 3. Mount all datasets
zfs mount -a
```
