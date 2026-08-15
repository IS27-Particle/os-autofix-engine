# Linux IP Routing & Default Gateway Recovery Runbook

## Overview
Loss of outbound network connectivity often results from missing default gateway routes, downed network interfaces, or broken Netplan/systemd-networkd configurations.

## Common Root Causes
1. Default route missing from the kernel routing table (`ip route`).
2. Primary network interface (`eth0` or `enp0s3`) is in the `DOWN` administrative state.
3. IP address assignment conflict or DHCP client timeout.

## Diagnostic Steps
```bash
# Check routing table
ip route show

# Check link status and IP addresses
ip -br link show
ip -br addr show

# Test gateway reachability
ping -c 2 -W 2 <gateway_ip>
```

## Remediation Commands
```bash
# 1. Bring up primary network interface
ip link set eth0 up

# 2. Add default gateway route
ip route add default via 10.0.0.1 dev eth0

# 3. Apply Netplan configuration persistently
netplan apply
```
