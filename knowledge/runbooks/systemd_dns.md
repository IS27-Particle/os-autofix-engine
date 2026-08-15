# Systemd-Resolved & DNS Recovery Runbook

## Overview
When domain name resolution fails on Linux systems using `systemd-resolved`, network utilities like `curl`, `apt`, and `ping` fail to resolve hostnames.

## Common Root Causes
1. `systemd-resolved` service is stopped or in a failed state.
2. `/etc/resolv.conf` is misconfigured or not pointing to the systemd-resolved stub listener (`/run/systemd/resolve/stub-resolv.conf`).
3. No valid upstream nameservers configured in `/etc/systemd/resolved.conf`.

## Diagnostic Steps
```bash
# Check service status
systemctl status systemd-resolved

# Check DNS query resolution
resolvectl status
resolvectl query google.com

# Verify symlink
ls -la /etc/resolv.conf
```

## Remediation Commands
```bash
# 1. Ensure symlink points to stub listener
ln -sf /run/systemd/resolve/stub-resolv.conf /etc/resolv.conf

# 2. Add fallback nameserver if needed
mkdir -p /etc/systemd/resolved.conf.d
echo -e "[Resolve]\nDNS=1.1.1.1 8.8.8.8" > /etc/systemd/resolved.conf.d/fallback.conf

# 3. Restart daemon
systemctl restart systemd-resolved
```
