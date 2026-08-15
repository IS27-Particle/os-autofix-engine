# OS-AutoFix Engine 🚀

[![CI](https://github.com/Antigravity/os-autofix-engine/actions/workflows/ci.yml/badge.svg)](https://github.com/Antigravity/os-autofix-engine/actions/workflows/ci.yml)
[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Virtualization: Incus](https://img.shields.io/badge/Virtualization-Incus%20VM%2FContainer-orange.svg)](https://linuxcontainers.org/incus/)

**os-autofix-engine** is an autonomous operating system-level policy training, evaluation, quantization, deployment, and closed-loop self-improvement harness designed for Large Language Model (LLM) Site Reliability Engineers.

It features **first-class support for Incus virtual machines and containers**, sub-second zero-copy COW snapshot rollbacks, structured JSON action enforcement, **4-bit LoRA (Unsloth / TRL)** and **GRPO alignment**, automatic **GGUF export**, remote **Ollama model deployment**, real-time **Prometheus telemetry & live TUI monitoring**, and continuous self-improving feedback loops.

---

## 🏛️ Architecture Overview

```mermaid
flowchart TD
    subgraph Host ["Host Infrastructure"]
        CLI["CLI (main.py)"] --> Orch["Async Orchestrator (Pool: N Workers)"]
        Orch --> LLM["Policy Client (Ollama / Open-WebUI)"]
        Orch --> IncusDriver["Incus Sandbox Driver"]
        Orch --> TBuffer["Trajectory Buffer (Thread-Safe)"]
        Orch --> Loop["Continuous Feedback Loop"]
        Orch --> Telemetry["Prometheus Metrics & Live TUI"]
    end

    subgraph Observability ["Telemetry & Monitoring Stack"]
        Telemetry --> PromServer["Prometheus Exporter (:9100/metrics)"]
        PromServer --> Prometheus["Prometheus (:9090)"]
        Prometheus --> Grafana["Grafana Dashboard (:3000)"]
    end

    subgraph Training_Deployment ["Training & Deployment Pipeline"]
        TBuffer --> SFT["SFT 4-bit LoRA (Unsloth / PEFT)"]
        TBuffer --> GRPO["GRPO Alignment (TRL)"]
        SFT --> GGUF["GGUF Quantization (q4_k_m)"]
        GGUF --> Deployer["Ollama Deployer (/api/create)"]
        GRPO --> Deployer
        Deployer --> OllamaTag["Ollama Model Tag (os-fixer:vN)"]
    end

    subgraph LLM_Endpoints ["Self-Hosted LLM Infrastructure"]
        LLM -->|REST API| Ollama["Ollama (http://10.0.0.25:11434)"]
        LLM -->|Bearer Auth| OpenWebUI["Open-WebUI (https://ai.is27.duckdns.org/api)"]
    end

    subgraph Incus ["Incus Hypervisor (ZFS / Btrfs)"]
        IncusDriver -->|launch --vm| VM["Isolated VM / Container Guest"]
        VM -->|incus-agent| Agent["incus-agent (Handshake)"]
        IncusDriver -->|incus snapshot create| SnapBase["Baseline Snapshot"]
        IncusDriver -->|Fault Injection| FaultState["Broken State"]
        IncusDriver -->|incus snapshot create| SnapFault["Fault Injected Snapshot"]
        IncusDriver -->|incus exec / non-interactive| Exec["Bash Execution (15s Timeout)"]
        IncusDriver -->|incus snapshot restore| Revert["Sub-second Rollback"]
    end
```

---

## 📋 Table of Contents
1. [Key Features](#-key-features)
2. [Prerequisites & Incus Setup](#-prerequisites--incus-setup)
3. [Endpoint Configuration](#-endpoint-configuration)
4. [Installation](#-installation)
5. [CLI Usage](#-cli-usage)
   - [Documentation & Runbook Indexer (`index-docs`)](#1-documentation--runbook-indexer-index-docs)
   - [Knowledge Base Query (`query-knowledge`)](#2-knowledge-base-query-query-knowledge)
   - [Host Self-Healing Watchdog Daemon (`watchdog`)](#3-host-self-healing-watchdog-daemon-watchdog)
   - [Kernel Syscall Security Auditor (`audit-security`)](#4-kernel-syscall-security-auditor-audit-security)
   - [Autonomous Chaos Engineering Daemon (`chaos`)](#5-autonomous-chaos-engineering-daemon-chaos)
   - [Multi-Node Distributed Topology Benchmark (`bench-distributed`)](#6-multi-node-distributed-topology-benchmark-bench-distributed)
   - [Model Context Protocol (MCP) Server (`mcp`)](#7-model-context-protocol-mcp-server-mcp)
   - [Tri-Agent Specialist Swarm (`swarm`)](#8-tri-agent-specialist-swarm-swarm)
   - [Model Arena ELO Tournament (`arena`)](#9-model-arena-elo-tournament-arena)
   - [Open-WebUI Pipeline Bundle Export (`export-webui`)](#10-open-webui-pipeline-bundle-export-export-webui)
   - [Monte Carlo Tree Search Trajectory Collection (`mcts-collect`)](#11-monte-carlo-tree-search-trajectory-collection-mcts-collect)
   - [Synthetic Scenario Synthesizer (`synthesize-scenario`)](#12-synthetic-scenario-synthesizer-synthesize-scenario)
   - [Host Pre-Flight Doctor (`doctor`)](#13-host-pre-flight-doctor-doctor)
   - [Real-Time Monitoring & Telemetry (`monitor`)](#14-real-time-monitoring--telemetry-monitor)
   - [Environment Health Check (`test-env`)](#15-environment-health-check-test-env)
   - [Scenario Benchmarking (`bench`)](#16-scenario-benchmarking-bench)
   - [Dataset Collection (`collect`)](#17-dataset-collection-collect)
   - [SFT 4-bit LoRA Training (`train-sft`)](#18-sft-4-bit-lora-training-train-sft)
   - [GRPO Policy Optimization (`train-grpo`)](#19-grpo-policy-optimization-train-grpo)
   - [Model Packaging & Ollama Deployment (`deploy`)](#20-model-packaging--ollama-deployment-deploy)
   - [Continuous Self-Improvement Loop (`loop`)](#21-continuous-self-improvement-loop-loop)
   - [Production Systemd Daemon Deployment (`deploy-daemon`)](#22-production-systemd-daemon-deployment-deploy-daemon)
   - [Automated GitHub Repo Setup (`git-init`)](#23-automated-github-repo-setup-git-init)
6. [Offline Hybrid Documentation & Runbook Retriever](#-offline-hybrid-documentation--runbook-retriever)
7. [Human-in-the-Loop Interactive Webhook Approval Gate](#-human-in-the-loop-interactive-webhook-approval-gate)
8. [Host Self-Healing Watchdog Daemon](#-host-self-healing-watchdog-daemon)
9. [Kernel Syscall Security Auditor & eBPF Inspection](#-kernel-syscall-security-auditor--ebpf-inspection)
10. [Multi-Node Distributed Scenarios](#-multi-node-distributed-scenarios)
11. [Autonomous Chaos Engineering Daemon](#-autonomous-chaos-engineering-daemon)
12. [Model Context Protocol (MCP) Integration](#-model-context-protocol-mcp-integration)
13. [Open-WebUI Pipeline & Tools](#-open-webui-pipeline--tools)
14. [Model Arena & ELO Rating System](#-model-arena--elo-rating-system)
15. [Supported Diagnostic Scenarios](#-supported-diagnostic-scenarios)
16. [Prometheus Metrics & Grafana](#-prometheus-metrics--grafana)
17. [Training Data Export Formats](#-training-data-export-formats)
18. [Testing & CI](#-testing--ci)

---

## ✨ Key Features

- **Offline Hybrid Documentation & Runbook Retriever**: Embedded BM25 + vector search engine parsing Linux manpages and markdown runbooks for systemd, ZFS, networking, Docker, and PAM authentication.
- **Human-in-the-Loop Interactive Webhook Approval Gate**: Intercepts high-blast-radius actions (safety score 0.70 - 0.85) with actionable Discord/Slack notifications and callback approvals.
- **Host Self-Healing Watchdog Daemon**: Continuously analyzes `journalctl` log streams, detects anomalies, and validates remediation inside ephemeral Incus shadow containers before live application.
- **Kernel-Level Syscall Security Auditor**: Intercepts `execve`, `unlinkat`, `chmod`, `connect`, `init_module` to catch recursive wiping (`rm -rf /`), reverse shells, and credential theft with automatic rollback if safety score drops below 0.70.
- **Multi-Node Distributed Topology Scenarios**: Orchestrates multi-instance Incus overlay topologies (`wireguard_mesh`, `etcd_split_brain`, `reverse_proxy_ha`) with automated cross-node fault injection and raft quorum verifiers.
- **Autonomous Chaos Engineering Daemon**: Continuously injects Poisson-distributed faults across a fleet of canary sandboxes, measuring Mean Time to Resolution (MTTR) and safety score distributions in real-time.
- **Tri-Agent Specialist Swarm (`Triage` $\to$ `Remediation` $\to$ `Audit`)**: Structured multi-turn state machine handoffs with read-only inspection boundaries, surgical state mutation, and automatic rollback on collateral damage.
- **Model Arena ELO Rating System**: Paired A/B tournament testing between model checkpoints on identical cloned snapshots with standard ELO update calculations ($K=32$) and multi-tier victory scoring.
- **Interactive Open-WebUI Pipeline & Tool Integration**: Real-time streaming pipeline filter yielding thoughts, bash commands, and diagnostics directly into `https://ai.is27.duckdns.org`.
- **Monte Carlo Tree Search (MCTS) Trajectory Collector**: Explores OS action spaces with snapshot branching, UCT selection, loop pruning heuristics, and shortest-path extraction into `data/dataset_mcts_optimal.jsonl`.
- **LLM-Driven Synthetic Scenario Synthesizer**: Prompts teacher models to generate novel Linux failure scenarios, dynamically compiles Python code, validates in 3-phase sandbox pre-flight checks, and registers them automatically.
- **Model Context Protocol (MCP) Server**: Exposes sandbox creation, fault injection, command execution, and benchmark metrics over stdio / SSE to Claude Desktop, Open-WebUI, and AI agent frameworks.
- **Native Incus Hypervisor Virtualization**: Ephemeral VM (`--vm`) or container isolation.
- **Zero-Copy Sub-Second Rollbacks**: Instant ZFS/Btrfs snapshotting before and after fault injection.
- **Guest Agent Polling**: Automatic `incus-agent` readiness detection with exponential backoff.
- **Context Protection**: Strict 2000-character stdout/stderr truncation and 15-second command timeouts.
- **Multi-Backend Inference**: Built-in integration with:
  - Local/Remote **Ollama** (`http://10.0.0.25:11434`)
  - **Open-WebUI** (`https://ai.is27.duckdns.org/api`) with Bearer token authentication.
  - OpenAI-compatible endpoints (`vLLM`, `LocalAI`, `DeepSeek`).
- **Real-Time Telemetry & Prometheus Exporter**: Native metrics exposition on `:9100/metrics` tracking sandboxes, tasks, steps, LLM latency, and snapshot revert performance.
- **Live Terminal UI (TUI) Dashboard**: Rich interactive dashboard tracking workers, agent thoughts, executed commands, and rolling pass rates in real-time.
- **4-bit LoRA Fine-Tuning**: `trainer/train_sft.py` targeting all linear attention/MLP layers (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`) with GGUF export.
- **GRPO Alignment**: `trainer/train_grpo.py` using multi-component reward functions (terminal verification, step efficiency penalty, JSON schema compliance bonus).
- **Automated Ollama Deployer**: Dynamic Modelfile construction, REST streaming deployment (`/api/create`), and version tagging (`os-fixer:v1`, `os-fixer:v2`).
- **Continuous Closed-Loop Engine**: Full autonomous self-improvement loop: Benchmark $\to$ Collect $\to$ Filter $\to$ Train $\to$ Deploy $\to$ Verify with automatic regression rollback.

---

## ⚙️ Prerequisites & Incus Setup

### 1. Incus Hypervisor & Storage Pool
Ensure Incus is installed with a ZFS or Btrfs storage pool for zero-copy snapshots:

```bash
# Verify Incus daemon version
incus version

# Verify storage pools (ZFS recommended)
incus storage list

# Verify network bridges
incus network list
```

### 2. KVM Acceleration (for VM mode)
To run in VM mode (`--vm`), ensure hardware virtualization is accessible:
```bash
ls -l /dev/kvm
# Ensure your user is in the 'incus-admin' group
groups | grep incus-admin
```

---

## 🌐 Endpoint Configuration

The engine reads target endpoints from environment variables with production defaults:

| Variable | Default Value | Description |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://10.0.0.25:11434/v1` | Ollama OpenAI-compatible or native `/api/chat` URL |
| `OPEN_WEBUI_BASE_URL` | `https://ai.is27.duckdns.org/api` | Open-WebUI REST API URL |
| `OPEN_WEBUI_API_KEY` | *(None)* | Bearer authentication token for Open-WebUI |
| `MODEL_NAME` | `qwen2.5-coder:7b` | Target LLM model name |
| `WORKER_COUNT` | `4` | Number of parallel Incus sandboxes |
| `TIMEOUT_SECONDS` | `15` | Default command execution timeout |
| `INCUS_PROJECT` | `default` | Incus project name |
| `INCUS_STORAGE_POOL` | `default` | Incus storage pool for snapshot copy-on-write |
| `INSTANCE_TYPE` | `vm` | `vm` for hardware isolation or `container` for lightweight speed |

---

## 📦 Installation

```bash
git clone https://github.com/Antigravity/os-autofix-engine.git
cd os-autofix-engine

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

---

## 💻 CLI Usage

### 1. Documentation & Runbook Indexer (`index-docs`)
Indexes offline Linux troubleshooting runbooks and manpage documentation into the hybrid search engine:

```bash
python3 main.py index-docs --runbooks-dir knowledge/runbooks --output reports/knowledge_index.json
```

### 2. Knowledge Base Query (`query-knowledge`)
Queries the offline hybrid BM25 / vector knowledge base for diagnostic runbooks:

```bash
python3 main.py query-knowledge --query "DNS resolution failure in systemd" --top-k 3
```

### 3. Host Self-Healing Watchdog Daemon (`watchdog`)
Runs the proactive journal stream watchdog with shadow container dry-run verification:

```bash
# Run in safe dry-run mode (validates fixes in Incus without modifying host)
python3 main.py watchdog --dry-run --min-safety-score 0.85

# Run in live mode (applies validated fixes to host)
python3 main.py watchdog --live --min-safety-score 0.90
```

### 4. Kernel Syscall Security Auditor (`audit-security`)
Inspects arbitrary commands or remediation scripts for destructive anti-patterns, reverse shells, and credential theft:

```bash
# Audit a high-risk destructive command
python3 main.py audit-security --command "rm -rf /"

# Audit a safe remediation command
python3 main.py audit-security --command "systemctl restart systemd-resolved"
```

### 5. Autonomous Chaos Engineering Daemon (`chaos`)
Runs continuous randomized Poisson-distributed fault injection experiments across a fleet of canary sandboxes:

```bash
python3 main.py chaos \
  --rate-minutes 1.0 \
  --fleet-size 3 \
  --duration-hours 1.0 \
  --type container
```

### 6. Multi-Node Distributed Topology Benchmark (`bench-distributed`)
Runs multi-node cluster verifications across WireGuard mesh, etcd Raft consensus, and Keepalived HA reverse proxies:

```bash
python3 main.py bench-distributed --scenario all --type container
```

### 7. Model Context Protocol (MCP) Server (`mcp`)
Starts the MCP server over standard I/O (stdio) or Server-Sent Events (SSE) for Claude Desktop, Open-WebUI, or external AI agents:

```bash
# Run MCP server on stdio (for Claude Desktop / IDE agents)
python3 main.py mcp

# Run MCP server on SSE (for remote network agents)
python3 main.py mcp --transport sse --port 8080
```

### 8. Tri-Agent Specialist Swarm (`swarm`)
Executes coordinated multi-turn handoffs between read-only Triage, surgical Remediation, and collateral safety Audit agents:

```bash
python3 main.py swarm \
  --scenario systemd_dns \
  --max-cycles 2 \
  --type container \
  --model qwen2.5-coder:7b
```

### 9. Model Arena ELO Tournament (`arena`)
Runs head-to-head A/B tournament matches between baseline and challenger models across identical sandbox snapshots with persistent ELO tracking:

```bash
python3 main.py arena \
  --model-a qwen2.5-coder:7b \
  --model-b os-fixer:v1 \
  --scenarios all \
  --rounds 2 \
  --ratings-file reports/arena_ratings.json
```

### 10. Open-WebUI Pipeline Bundle Export (`export-webui`)
Exports the ready-to-import Open-WebUI pipeline code and tool function calling schema:

```bash
python3 main.py export-webui --output dist/open_webui_bundle.json
```

### 11. Monte Carlo Tree Search Trajectory Collection (`mcts-collect`)
Explores the OS action space using snapshot-branching Monte Carlo Tree Search and extracts the optimal shortest-path resolution trajectory:

```bash
python3 main.py mcts-collect \
  --scenario systemd_dns \
  --simulations 15 \
  --exploration-constant 1.414 \
  --max-depth 6 \
  --branch-factor 3 \
  --output-file data/dataset_mcts_optimal.jsonl \
  --backend ollama \
  --model qwen2.5-coder:7b
```

### 12. Synthetic Scenario Synthesizer (`synthesize-scenario`)
Uses teacher LLMs to synthesize novel Linux diagnostic scenarios, validates them inside Incus sandboxes across 3 verification phases, and automatically registers them:

```bash
python3 main.py synthesize-scenario \
  --count 3 \
  --topic "PAM security lockout and file descriptor limits" \
  --output-dir scenarios/synthetic \
  --validate \
  --backend ollama \
  --model qwen2.5-coder:7b
```

### 13. Host Pre-Flight Doctor (`doctor`)
Runs pre-flight diagnostics for KVM virtualization, Incus CLI/storage/bridges, and remote LLM endpoints:

```bash
python3 main.py doctor
```

### 14. Real-Time Monitoring & Telemetry (`monitor`)
Launch the live Rich terminal dashboard or start the standalone Prometheus metrics exporter:

```bash
# Launch interactive terminal TUI dashboard (with background metrics exporter)
python3 main.py monitor

# Run standalone HTTP Prometheus metrics server
python3 main.py monitor --port 9100 --server-only
```

### 15. Environment Health Check (`test-env`)
Validates Incus hypervisor, KVM acceleration, Ollama / Open-WebUI connectivity, and executes a live ephemeral sandbox snapshot rollback test:

```bash
python3 main.py test-env --type container
```

### 16. Scenario Benchmarking (`bench`)
Evaluates model performance across diagnostic fault scenarios in parallel:

```bash
python3 main.py bench \
  --backend ollama \
  --endpoint http://10.0.0.25:11434/v1 \
  --model qwen2.5-coder:7b \
  --workers 4 \
  --type vm
```

### 17. Dataset Collection (`collect`)
Generates exploration rollouts and exports positive trajectories for training:

```bash
python3 main.py collect \
  --scenarios all \
  --samples 20 \
  --workers 4 \
  --backend ollama \
  --format all \
  --output-dir data
```

### 18. SFT 4-bit LoRA Training (`train-sft`)
Fine-tunes base models on ShareGPT-formatted trajectory datasets with automatic GGUF quantization:

```bash
python3 main.py train-sft \
  --dataset data/dataset_unsloth_sharegpt.jsonl \
  --model qwen2.5-coder:7b \
  --output-dir outputs/sft_adapter \
  --epochs 3 \
  --batch-size 2 \
  --lora-r 16 \
  --export-gguf
```

### 19. GRPO Policy Optimization (`train-grpo`)
Runs Group Relative Policy Optimization using multi-component reward functions:

```bash
python3 main.py train-grpo \
  --dataset data/dataset_trl_grpo.jsonl \
  --model qwen2.5-coder:7b \
  --output-dir outputs/grpo_adapter \
  --epochs 1 \
  --generations 4
```

### 20. Model Packaging & Ollama Deployment (`deploy`)
Generates Modelfiles with structured system prompts and registers new model tags with Ollama:

```bash
python3 main.py deploy \
  --model-tag os-fixer:v1 \
  --base qwen2.5-coder:7b \
  --ollama-url http://10.0.0.25:11434
```

### 21. Continuous Self-Improvement Loop (`loop`)
Executes an autonomous closed-loop cycle (Benchmark $\to$ Collect $\to$ Filter $\to$ Train $\to$ Deploy $\to$ Verify) with automatic rollback protection:

```bash
python3 main.py loop \
  --iterations 3 \
  --samples 6 \
  --training-type sft \
  --model qwen2.5-coder:7b \
  --prefix os-fixer \
  --workers 4 \
  --type container \
  --ollama-url http://10.0.0.25:11434
```

### 22. Production Systemd Daemon Deployment (`deploy-daemon`)
Installs and enables the `os-autofix.service` and `os-autofix-metrics.service` unit files:

```bash
python3 main.py deploy-daemon --systemd-dir /etc/systemd/system --enable
```

### 23. Automated GitHub Repo Setup (`git-init`)
Initializes the git repository, stages all code, creates the remote repo, and pushes initial commits:

```bash
python3 main.py git-init --name os-autofix-engine --public
```

---

## 📚 Offline Hybrid Documentation & Runbook Retriever

The **Hybrid Retriever** ([`knowledge/retriever.py`](file:///Docs/Programming/GitHub/os-autofix-engine/knowledge/retriever.py)) embeds local Linux troubleshooting runbooks and manpage lookups for zero-network environments:

- **Hybrid Search Engine**: BM25 keyword matching + normalized term vector similarity with title and section boost.
- **Pre-Populated Troubleshooting Runbooks**:
  - `knowledge/runbooks/systemd_dns.md`: DNS resolution, stub listener symlinks, and fallback configuration.
  - `knowledge/runbooks/zfs_storage.md`: Storage pool recovery, dataset mounting, and canmount properties.
  - `knowledge/runbooks/networking_routing.md`: Kernel routing tables, gateway recovery, and Netplan.
  - `knowledge/runbooks/docker_daemon.md`: Socket permissions, daemon JSON syntax, and service recovery.
  - `knowledge/runbooks/pam_security.md`: Authentication lockout clearance and PAM configuration.

---

## 🚦 Human-in-the-Loop Interactive Webhook Approval Gate

The **Approval Gate** ([`security/approval_gate.py`](file:///Docs/Programming/GitHub/os-autofix-engine/security/approval_gate.py)) manages human authorization for high-blast-radius actions:

- **Review Trigger Window**: Intercepts actions with safety scores between **0.70** and **0.85**.
- **Actionable Notifications**: Dispatches Discord/Slack webhook embeds containing proposed command diffs, blast radius, and approval/rejection callback endpoints.
- **State Machine**: Supports `PENDING`, `APPROVED`, `REJECTED`, and `TIMED_OUT` (default 60s timeout automatically commanding rollback).

---

## 🐕 Host Self-Healing Watchdog Daemon

The **Host Watchdog Daemon** ([`engine/host_watchdog.py`](file:///Docs/Programming/GitHub/os-autofix-engine/engine/host_watchdog.py)) provides proactive anomaly detection and shadow container dry-run remediation:

- **Journal Stream Monitoring**: Continuously parses systemd logs for failed services, OOM kills, network interface drops, or locked package managers.
- **Shadow Dry-Run Verification**: Spawns an isolated Incus container, reproduces the fault, and proves the fix works *before* applying changes to the live host.
- **Safety Enforcement**: Rejects fixes if the evaluated security score is below `--min-safety-score` (default 0.85).

---

## 🛡️ Kernel Syscall Security Auditor & eBPF Inspection

The **Syscall Security Auditor** ([`security/ebpf_auditor.py`](file:///Docs/Programming/GitHub/os-autofix-engine/security/ebpf_auditor.py)) evaluates guest commands and system state against destructive anti-patterns:

- **Intercepted Syscalls**: `execve`, `unlinkat`, `chmod`, `connect`, `init_module`, `ptrace`.
- **Destructive Heuristics**:
  - Recursive root wiping (`rm -rf /`, `rm -rf /etc`, `rm -rf /boot`)
  - Reverse shell backdoors (`nc -e /bin/bash`, `/dev/tcp/IP/PORT`, `socat`)
  - Credential harvesting (`cat /etc/shadow`, `cat ~/.ssh/id_rsa`)
  - Dangerous permissions (`chmod -R 777 /etc`)
  - Raw disk destruction (`dd if=/dev/zero of=/dev/sda`, `mkfs.ext4 /dev/vda`)
- **Blast Radius & Safety Score**: Evaluates execution safety ($0.0 - 1.0$). If safety score drops below $0.70$, execution is immediately aborted and the Audit Agent commands an instant Incus snapshot rollback.

---

## 🌐 Multi-Node Distributed Scenarios

The **Distributed Topology Suite** ([`scenarios/distributed/`](file:///Docs/Programming/GitHub/os-autofix-engine/scenarios/distributed/)) tests multi-instance cluster troubleshooting:

| Scenario | Nodes | Topology | Fault Injected |
|---|---|---|---|
| [`wireguard_mesh`](file:///Docs/Programming/GitHub/os-autofix-engine/scenarios/distributed/wireguard_mesh.py) | `node-1`, `node-2`, `node-3` | 3-node full mesh overlay (`10.0.99.0/24`) | MTU mismatch (1200) on node-1 and corrupted cryptographic peer public keys on node-2 |
| [`etcd_split_brain`](file:///Docs/Programming/GitHub/os-autofix-engine/scenarios/distributed/etcd_split_brain.py) | `etcd-1`, `etcd-2`, `etcd-3` | 3-node Raft consensus cluster | `iptables` DROP partition on peer heartbeat port 2380 causing quorum loss |
| [`reverse_proxy_ha`](file:///Docs/Programming/GitHub/os-autofix-engine/scenarios/distributed/reverse_proxy_ha.py) | `lb-1`, `lb-2`, `backend-1`, `backend-2` | Dual Keepalived VRRP + HAProxy with VIP (`10.0.0.200`) | Dead master VRRP daemon and VIP takeover priority lockout |

---

## 🌪️ Autonomous Chaos Engineering Daemon

The **Chaos Daemon** ([`engine/chaos_daemon.py`](file:///Docs/Programming/GitHub/os-autofix-engine/engine/chaos_daemon.py)) runs Poisson-distributed fault injection experiments against running Incus canary sandboxes:

- **Automated Fleet Canary Management**: Spawns isolated canary instances, prepares scenarios, injects randomized faults, and triggers autonomous remediation.
- **Continuous Reliability Metrics**:
  - **MTTR (Mean Time to Resolution)**: Recorded via Prometheus histogram `os_autofix_mttr_seconds`.
  - **Recovery Success Rate**: Tracks percentage of successful recoveries without human intervention.
  - **Safety Score Distribution**: Monitored via Prometheus counter `os_autofix_safety_violations_total`.

---

---

## 🔌 Model Context Protocol (MCP) Integration

The **os-autofix-engine** includes a native **Model Context Protocol (MCP)** server enabling Claude Desktop, Open-WebUI, Cursor, and agent frameworks to directly control Incus sandboxes, inject faults, execute bash commands, and run evaluation benchmarks.

### Registering with Claude Desktop

Run the registration helper to generate configuration snippets:
```bash
python3 scripts/register_mcp.py --write-claude
```

Or manually add to `~/.config/Claude/claude_desktop_config.json`:
```json
{
  "mcpServers": {
    "os-autofix": {
      "command": "python3",
      "args": ["/Docs/Programming/GitHub/os-autofix-engine/main.py", "mcp"],
      "env": {
        "OLLAMA_BASE_URL": "http://10.0.0.25:11434/v1",
        "OPEN_WEBUI_BASE_URL": "https://ai.is27.duckdns.org/api"
      }
    }
  }
}
```

### Exposed MCP Tools

| MCP Tool Name | Parameters | Description |
|---|---|---|
| `list_scenarios` | None | Lists all 7 available diagnostic scenarios and descriptions. |
| `create_sandbox` | `instance_type`, `image` | Spawns an isolated Incus container or VM and creates baseline snapshot. |
| `inject_fault` | `instance_id`, `scenario_name` | Sets up baseline scenario state, injects fault, and verifies initial breakage. |
| `exec_command` | `instance_id`, `command`, `timeout_seconds` | Executes non-interactive shell command with 2000-character truncation and timeout protection. |
| `verify_fix` | `instance_id`, `scenario_name` | Executes scenario verifier assertions and returns boolean pass/fail status. |
| `revert_sandbox` | `instance_id`, `snapshot_name` | Instant CoW rollback to `snap-baseline` or specified snapshot. |
| `destroy_sandbox` | `instance_id` | Deletes ephemeral container/VM and releases host resources. |
| `run_benchmark` | `scenarios`, `workers`, `iterations`, `model` | Executes parallel multi-worker benchmark and returns aggregated metrics. |

### Exposed MCP Resources

- `report://benchmark/latest`: Latest evaluation benchmark report in Markdown.
- `status://cluster`: Live host hardware, active sandboxes count, storage pools, and Ollama endpoint health.

---

## 🌐 Open-WebUI Pipeline & Tools

Integrate the **os-autofix-engine** directly with Open-WebUI running at `https://ai.is27.duckdns.org` or local deployments.

### 1. Export Bundle
```bash
python3 main.py export-webui --output dist/open_webui_bundle.json
```

### 2. Features
- **Streaming Execution Markdown**: Streams agent thoughts, bash commands, output drawers, and terminal verification status in real-time.
- **Valves Configuration**: Modify Ollama endpoint (`http://10.0.0.25:11434/v1`), model tags, and sandbox types (`container` / `vm`) directly from the Admin UI.
- **Tool Calling**: Function calling schemas for sandbox control and execution inside [`integrations/open_webui/tool_def.json`](file:///Docs/Programming/GitHub/os-autofix-engine/integrations/open_webui/tool_def.json).

---

## ⚔️ Model Arena & ELO Rating System

The **Model Arena** ([`engine/arena.py`](file:///Docs/Programming/GitHub/os-autofix-engine/engine/arena.py)) executes paired A/B tournaments between model checkpoints (e.g., baseline `qwen2.5-coder:7b` vs fine-tuned `os-fixer:v1`):

- **Snapshot-Cloned Isolation**: Both models are evaluated against exact identical snapshot states.
- **Multi-Tier Victory Criteria**:
  1. Terminal verification success (1.0 vs 0.0).
  2. Step count efficiency (fewer steps win).
  3. Execution duration (lower latency wins).
  4. Draw / Tie (0.5 vs 0.5) if performance is equivalent.
- **Standard Logistic ELO Tracking ($K=32$)**:
  $$E_A = \frac{1}{1 + 10^{(R_B - R_A)/400}}, \quad R_A' = R_A + K(S_A - E_A)$$
- **Persistent Leaderboard**: Tracks rating changes across iterations in [`reports/arena_ratings.json`](file:///Docs/Programming/GitHub/os-autofix-engine/reports/arena_ratings.json).

---

## 📈 Prometheus Metrics & Grafana

### Exported Prometheus Metrics

| Metric Name | Type | Description |
|---|---|---|
| `os_autofix_sandboxes_active` | Gauge | Active concurrent running Incus instances |
| `os_autofix_tasks_total` | Counter | Completed tasks partitioned by `scenario`, `model_tag`, `status` |
| `os_autofix_episode_steps` | Histogram | Steps per scenario resolution |
| `os_autofix_llm_latency_seconds` | Histogram | LLM inference latency per model request |
| `os_autofix_sandbox_revert_seconds` | Histogram | ZFS/Btrfs CoW snapshot restore latency |
| `os_autofix_model_pass_rate` | Gauge | Current benchmark generation pass rate accuracy |

### Launching Prometheus & Grafana Stack

```bash
cd deploy/
docker compose up -d
```

- Prometheus UI: `http://localhost:9090`
- Grafana UI: `http://localhost:3000` (Pre-loaded dashboard in [`monitoring/grafana_dashboard.json`](file:///Docs/Programming/GitHub/os-autofix-engine/monitoring/grafana_dashboard.json))

---

## 🎯 Supported Diagnostic Scenarios

| Scenario | Category | Difficulty | Max Steps | Description |
|---|---|---|---|---|
| [`systemd_dns`](file:///Docs/Programming/GitHub/os-autofix-engine/scenarios/systemd_dns.py) | Networking / DNS | Medium | 8 | Domain name resolution failure caused by broken systemd-resolved and symlinks. |
| [`network_routing`](file:///Docs/Programming/GitHub/os-autofix-engine/scenarios/network_routing.py) | Networking / Routing | Medium | 8 | Corrupted default gateway route breaking outbound IP packet routing. |
| [`package_corruption`](file:///Docs/Programming/GitHub/os-autofix-engine/scenarios/package_corruption.py) | Package Management | Easy | 6 | Locked or interrupted APT/dpkg state preventing package operations. |
| [`file_permissions`](file:///Docs/Programming/GitHub/os-autofix-engine/scenarios/file_permissions.py) | Security / Permissions | Easy | 6 | Critical system configuration files (`/etc/sudoers`, `/etc/ssh/sshd_config`) with corrupted permissions. |
| [`zfs_mount`](file:///Docs/Programming/GitHub/os-autofix-engine/scenarios/zfs_mount.py) | Storage / Filesystems | Medium | 8 | Unmounted or inaccessible filesystem dataset mountpoints (`/mnt/data`). |
| [`docker_socket`](file:///Docs/Programming/GitHub/os-autofix-engine/scenarios/docker_socket.py) | Containers / Docker | Medium | 6 | Permission lockouts and service failures on `/var/run/docker.sock`. |
| [`iptables_lockout`](file:///Docs/Programming/GitHub/os-autofix-engine/scenarios/iptables_lockout.py) | Networking / Security | Medium | 8 | Restrictive outbound firewall DROP rules blocking DNS and HTTP/HTTPS traffic. |

---

## 🔔 Webhook Alerts & Notification Dispatcher

The [`WebhookAlertDispatcher`](file:///Docs/Programming/GitHub/os-autofix-engine/monitoring/alerts.py) supports Discord, Slack, and generic JSON webhook endpoints:

- **Environment Variable**: `WEBHOOK_ALERT_URL="https://discord.com/api/webhooks/..."`
- **Real-Time Notification Triggers**:
  - `model_promoted`: Fired when a newly trained model generation beats baseline thresholds and is promoted.
  - `model_regression`: Fired when an evaluated model generation regresses below threshold, detailing automatic rollback.
  - `worker_failure`: Fired on Incus guest agent handshake timeouts or sandbox execution crashes.

---

## 📊 Training Data Export Formats

The [`TrajectoryBuffer`](file:///Docs/Programming/GitHub/os-autofix-engine/trainer/trajectory_buffer.py#L51) exports:

1. **Hugging Face TRL GRPO** (`data/dataset_trl_grpo.jsonl`):
   ```json
   {
     "prompt": "Diagnose and resolve the following OS issue: systemd_dns",
     "completion": "{\"thought\": \"Restarting resolver\", \"command\": \"systemctl restart systemd-resolved\", \"is_done\": true}",
     "reward": 1.15,
     "success": true,
     "scenario": "systemd_dns",
     "steps_count": 1
   }
   ```

2. **Hugging Face TRL DPO** (`data/dataset_trl_dpo.jsonl`):
   ```json
   {
     "prompt": "Diagnose and resolve the following OS issue: systemd_dns",
     "chosen": "{\"thought\": \"Fix symlink\", \"command\": \"ln -sf ... /etc/resolv.conf\", \"is_done\": true}",
     "rejected": "{\"thought\": \"Unknown\", \"command\": \"reboot\", \"is_done\": true}",
     "scenario": "systemd_dns"
   }
   ```

3. **Unsloth / ShareGPT** (`data/dataset_unsloth_sharegpt.jsonl`):
   ```json
   {
     "id": "episode-uuid",
     "conversations": [
       {"from": "human", "value": "Troubleshoot and resolve this OS issue:\nScenario: systemd_dns"},
       {"from": "gpt", "value": "{\"thought\": \"Fix DNS\", \"command\": \"systemctl restart systemd-resolved\", \"is_done\": true}"}
     ]
   }
   ```

---

## 🧪 Testing & CI

```bash
# Run linter and formatter checks
ruff check . && ruff format --check .

# Run static type checking
mypy config sandbox engine scenarios trainer monitoring deploy tests

# Run all 45 unit and integration tests
pytest -v tests/
```

GitHub Actions workflow ([`.github/workflows/ci.yml`](file:///Docs/Programming/GitHub/os-autofix-engine/.github/workflows/ci.yml)) automatically runs the test matrix across Python 3.10, 3.11, 3.12, and 3.13 on every pull request.
