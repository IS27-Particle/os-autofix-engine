"""Unit tests for synthetic scenario synthesizer and pre-flight validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from config.settings import EngineConfig
from scenarios.base_scenario import BaseScenario
from scenarios.registry import get_scenario
from scenarios.synthesizer import ScenarioSynthesizer
from tests.conftest import MockSandbox


def test_extract_python_code() -> None:
    """Test extracting clean python code from LLM responses with markdown code blocks."""
    synthesizer = ScenarioSynthesizer(config=EngineConfig())
    raw_response = (
        "Here is the generated scenario:\n"
        "```python\n"
        "class MyScenario(BaseScenario):\n"
        "    name = 'my_test'\n"
        "```\n"
        "Hope this helps!"
    )
    code = synthesizer.extract_python_code(raw_response)
    assert code.startswith("class MyScenario")
    assert code.endswith("name = 'my_test'")


def test_compile_scenario_class() -> None:
    """Test dynamic compilation of synthetic scenario code string into a BaseScenario subclass."""
    synthesizer = ScenarioSynthesizer(config=EngineConfig())
    code = """from scenarios.base_scenario import BaseScenario
from sandbox.base import BaseSandbox

class TestPamLockoutScenario(BaseScenario):
    name = "synthetic_pam_test"
    description = "Test PAM lockout"
    category = "Security"
    difficulty = "easy"
    max_steps = 5

    async def setup(self, sandbox: BaseSandbox) -> bool:
        return True

    async def inject_fault(self, sandbox: BaseSandbox) -> bool:
        return True

    async def verify(self, sandbox: BaseSandbox) -> tuple[bool, str]:
        return True, "Resolved"
"""
    cls = synthesizer.compile_scenario_class(code)
    assert issubclass(cls, BaseScenario)
    instance = cls()
    assert instance.name == "synthetic_pam_test"


@pytest.mark.asyncio
async def test_validate_preflight_and_synthesis(tmp_path: Path) -> None:
    """Test full scenario generation, pre-flight sandbox validation, and registration."""
    cfg = EngineConfig()
    cfg.llm.mock_mode = True
    out_dir = tmp_path / "synthetic_scenarios"
    synthesizer = ScenarioSynthesizer(config=cfg, output_dir=out_dir)

    def mock_factory(name: str) -> MockSandbox:
        return MockSandbox(name)

    results = await synthesizer.synthesize(
        count=1,
        topic="limits_nofile",
        sandbox_factory=mock_factory,
    )

    assert len(results) == 1
    assert results[0]["valid"] is True
    sc_name = results[0]["name"]
    assert "limits" in sc_name

    # Check file exists on disk
    sc_file = out_dir / f"{sc_name}.py"
    assert sc_file.exists()

    # Check scenario is dynamically registered
    sc_instance = get_scenario(sc_name)
    assert sc_instance.name == sc_name
