"""Configuration settings and environment variable bindings for os-autofix-engine."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


@dataclass
class IncusConfig:
    """Configuration for Incus hypervisor and virtual sandbox instances."""

    project: str = "default"
    storage_pool: str = "default"
    default_image: str = "images:ubuntu/24.04"
    instance_type: Literal["vm", "container"] = "vm"
    instance_prefix: str = "autofix"
    agent_wait_timeout_seconds: int = 60
    agent_poll_interval_seconds: float = 1.0
    command_timeout_seconds: int = 15
    max_output_chars: int = 2000
    cpu_limit: str = "2"
    memory_limit: str = "2GiB"
    network_name: str = "lxdbr0"

    @property
    def is_vm(self) -> bool:
        """Return True if the configured instance type is VM."""
        return self.instance_type.lower() == "vm"


@dataclass
class LLMConfig:
    """Configuration for local Ollama, Open-WebUI, or OpenAI-compatible endpoints."""

    backend: Literal["ollama", "open-webui", "openai", "mock"] = "ollama"
    ollama_base_url: str = "http://10.0.0.25:11434/v1"
    open_webui_base_url: str = "https://ai.is27.duckdns.org/api"
    open_webui_api_key: str = ""
    model_name: str = "qwen2.5-coder:7b"
    temperature: float = 0.2
    max_tokens: int = 1024
    top_p: float = 0.95
    timeout_seconds: float = 30.0
    max_retries: int = 3
    mock_mode: bool = False

    @property
    def active_endpoint(self) -> str:
        """Resolve base URL depending on configured backend."""
        if self.backend == "open-webui":
            return self.open_webui_base_url.rstrip("/")
        return self.ollama_base_url.rstrip("/")

    @property
    def active_api_key(self) -> str:
        """Resolve API authentication token."""
        if self.backend == "open-webui" and self.open_webui_api_key:
            return self.open_webui_api_key
        return "ollama"


@dataclass
class GitHubConfig:
    """Configuration for GitHub repository integration and synchronization."""

    repo_name: str = "os-autofix-engine"
    remote_name: str = "origin"
    default_branch: str = "main"
    private: bool = False
    organization: str | None = None
    auto_push: bool = True


@dataclass
class EngineConfig:
    """Central configuration for the os-autofix-engine orchestrator and trainer."""

    workers: int = 4
    max_steps_per_episode: int = 10
    step_penalty: float = 0.05
    success_reward: float = 1.0
    failure_reward: float = 0.0
    webhook_alert_url: str = ""
    data_dir: Path = field(default_factory=lambda: Path("data"))
    logs_dir: Path = field(default_factory=lambda: Path("logs"))
    incus: IncusConfig = field(default_factory=IncusConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    github: GitHubConfig = field(default_factory=GitHubConfig)

    def __post_init__(self) -> None:
        """Ensure runtime data and log directories exist."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)


def get_default_config() -> EngineConfig:
    """Construct engine configuration with environment variable overrides."""
    instance_type_str = os.getenv(
        "INSTANCE_TYPE", os.getenv("OS_AUTOFIX_INSTANCE_TYPE", "vm")
    ).lower()
    instance_type: Literal["vm", "container"] = (
        "container" if instance_type_str == "container" else "vm"
    )

    backend_str = os.getenv("LLM_BACKEND", os.getenv("OS_AUTOFIX_BACKEND", "ollama")).lower()
    backend_val: Literal["ollama", "open-webui", "openai", "mock"]
    if backend_str in ("open-webui", "openwebui", "webui"):
        backend_val = "open-webui"
    elif backend_str == "openai":
        backend_val = "openai"
    elif backend_str == "mock":
        backend_val = "mock"
    else:
        backend_val = "ollama"

    incus_cfg = IncusConfig(
        project=os.getenv("INCUS_PROJECT", "default"),
        storage_pool=os.getenv("INCUS_STORAGE_POOL", "default"),
        default_image=os.getenv("INCUS_IMAGE", "images:ubuntu/24.04"),
        instance_type=instance_type,
        command_timeout_seconds=int(
            os.getenv("TIMEOUT_SECONDS", os.getenv("OS_AUTOFIX_TIMEOUT", "15"))
        ),
    )

    llm_cfg = LLMConfig(
        backend=backend_val,
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://10.0.0.25:11434/v1"),
        open_webui_base_url=os.getenv("OPEN_WEBUI_BASE_URL", "https://ai.is27.duckdns.org/api"),
        open_webui_api_key=os.getenv("OPEN_WEBUI_API_KEY", ""),
        model_name=os.getenv("MODEL_NAME", "qwen2.5-coder:7b"),
        timeout_seconds=float(os.getenv("LLM_TIMEOUT", "30.0")),
        mock_mode=os.getenv("MOCK_LLM", "false").lower() in ("1", "true", "yes")
        or backend_val == "mock",
    )

    github_cfg = GitHubConfig(
        repo_name=os.getenv("GITHUB_REPO", "os-autofix-engine"),
        remote_name=os.getenv("GITHUB_REMOTE", "origin"),
        default_branch=os.getenv("GITHUB_BRANCH", "main"),
        private=os.getenv("GITHUB_PRIVATE", "false").lower() in ("1", "true", "yes"),
    )

    return EngineConfig(
        workers=int(os.getenv("WORKER_COUNT", "4")),
        max_steps_per_episode=int(os.getenv("MAX_STEPS", "10")),
        webhook_alert_url=os.getenv("WEBHOOK_ALERT_URL", ""),
        incus=incus_cfg,
        llm=llm_cfg,
        github=github_cfg,
    )
