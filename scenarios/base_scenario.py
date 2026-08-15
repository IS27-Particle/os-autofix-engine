"""Abstract base class for environment fault injection and resolution verification scenarios."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from sandbox.base import BaseSandbox


class BaseScenario(ABC):
    """Abstract class defining the lifecycle of a troubleshooting scenario."""

    name: str = "base_scenario"
    description: str = "Base scenario description"
    category: str = "General"
    difficulty: str = "medium"
    max_steps: int = 10

    @abstractmethod
    async def setup(self, sandbox: BaseSandbox) -> bool:
        """Prepare baseline software packages, services, or configurations inside the guest."""
        pass

    @abstractmethod
    async def inject_fault(self, sandbox: BaseSandbox) -> bool:
        """Introduce the specific operating system failure or misconfiguration."""
        pass

    @abstractmethod
    async def verify(self, sandbox: BaseSandbox) -> tuple[bool, str]:
        """Assert whether the system state has been successfully restored and functional.

        Returns:
            Tuple[bool, str]: (is_resolved, explanation_message)
        """
        pass

    def get_prompt(self) -> str:
        """Construct the prompt presented to the agent describing the symptom."""
        return (
            f"Scenario: {self.name}\n"
            f"Category: {self.category} | Difficulty: {self.difficulty}\n"
            f"Description: {self.description}\n"
            f"Task: Investigate the root cause and restore the system to full operational health."
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize scenario metadata."""
        return {
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "difficulty": self.difficulty,
            "max_steps": self.max_steps,
        }
