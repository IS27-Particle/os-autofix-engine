# PAM Authentication & Account Lockout Recovery Runbook

## Overview
Pluggable Authentication Module (PAM) failures prevent user logins via SSH, console, or `sudo`. Common issues include max authentication attempt lockouts (`pam_faillock`, `pam_tally2`) or syntax errors in `/etc/pam.d/common-auth`.

## Common Root Causes
1. Account locked out due to consecutive failed password attempts.
2. Syntax errors or misordered control flags in `/etc/pam.d/` configuration files.
3. Missing or corrupted `/etc/security/faillock.conf`.

## Diagnostic Steps
```bash
# Check faillock status for a specific user
faillock --user <username>

# Check PAM journal logs
journalctl -u ssh -u systemd-logind -n 50 --no-pager
```

## Remediation Commands
```bash
# 1. Reset user failed attempt counter
faillock --user <username> --reset

# 2. Verify PAM common-auth structure
pam-auth-update --force
```
