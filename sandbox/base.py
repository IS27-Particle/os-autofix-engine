"""Abstract base classes and execution result structures for sandboxes."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ExecutionResult:
    """Represents the outcome of a command executed inside a sandbox environment."""

    command: str
    exit_code: int
    stdout: str = ""
    stderr: str = ""
    duration_seconds: float = 0.0
    truncated: bool = False
    timed_out: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        """Return True if command succeeded with exit code 0 and did not time out."""
        return self.exit_code == 0 and not self.timed_out

    @property
    def combined_output(self) -> str:
        """Return combined stdout and stderr representation."""
        parts: list[str] = []
        if self.stdout:
            parts.append(self.stdout)
        if self.stderr:
            parts.append(f"[STDERR]\n{self.stderr}")
        if self.timed_out:
            parts.append("[ERROR: Command timed out]")
        return "\n".join(parts) if parts else "[No output]"


class BaseSandbox(ABC):
    """Abstract interface defining required sandbox lifecycle and execution primitives."""

    @abstractmethod
    async def setup(self) -> None:
        """Initialize, launch, and verify readiness of the sandbox instance."""
        pass

    @abstractmethod
    async def create_snapshot(self, snapshot_name: str) -> None:
        """Create a point-in-time snapshot of the instance state."""
        pass

    @abstractmethod
    async def revert(self, snapshot_name: str) -> None:
        """Roll back the instance to a previously created snapshot."""
        pass

    @abstractmethod
    async def execute(
        self,
        command: str,
        timeout_seconds: int | None = None,
    ) -> ExecutionResult:
        """Execute a shell command inside the guest instance asynchronously."""
        pass

    @abstractmethod
    async def get_state(self) -> dict[str, Any]:
        """Query and return instance status, network info, and metrics."""
        pass

    @abstractmethod
    async def cleanup(self) -> None:
        """Terminate and permanently delete the instance and ephemeral resources."""
        pass

    @abstractmethod
    async def read_file(self, remote_path: str) -> str:
        """Read text content of a file from the guest filesystem."""
        pass

    @abstractmethod
    async def write_file(self, remote_path: str, content: str, mode: str | None = None) -> None:
        """Write content to a file inside the guest filesystem."""
        pass

    async def __aenter__(self) -> BaseSandbox:
        await self.setup()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        await self.cleanup()
