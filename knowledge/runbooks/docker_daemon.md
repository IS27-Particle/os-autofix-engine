# Docker Daemon & Unix Socket Lockout Recovery Runbook

## Overview
Docker client operations (`docker ps`, `docker run`) fail when the Docker daemon (`dockerd`) crashes, the `/var/run/docker.sock` Unix domain socket has incorrect permissions, or invalid JSON exists in `/etc/docker/daemon.json`.

## Common Root Causes
1. Permission denied on `/var/run/docker.sock` (requires `0660` with group `docker`).
2. Syntax error in `/etc/docker/daemon.json`.
3. Systemd service `docker.service` or `docker.socket` is masked or inactive.

## Diagnostic Steps
```bash
# Check docker service and socket status
systemctl status docker.socket docker.service

# Check socket file ownership and permissions
ls -la /var/run/docker.sock

# Validate JSON configuration
cat /etc/docker/daemon.json | jq .
```

## Remediation Commands
```bash
# 1. Correct socket permissions
chmod 0660 /var/run/docker.sock
chown root:docker /var/run/docker.sock

# 2. Fix JSON configuration syntax if corrupted
echo "{}" > /etc/docker/daemon.json

# 3. Restart Docker socket and daemon
systemctl restart docker.socket docker.service
```
