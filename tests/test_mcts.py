"""Unit tests for Monte Carlo Tree Search (MCTS) trajectory collection."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from config.settings import EngineConfig
from engine.action_schema import AgentAction
from scenarios.systemd_dns import SystemdDNSScenario
from tests.conftest import MockSandbox
from trainer.mcts_search import MCTSNode, MCTSSearchEngine


def test_mcts_node_uct_and_q_value() -> None:
    """Test MCTS node Q-value and UCT formula computation."""
    root = MCTSNode(node_id="root", visit_count=10, total_reward=5.0)
    child1 = MCTSNode(node_id="c1", parent=root, visit_count=5, total_reward=4.0)
    child2 = MCTSNode(node_id="c2", parent=root, visit_count=2, total_reward=1.0)
    unvisited = MCTSNode(node_id="c3", parent=root, visit_count=0, total_reward=0.0)

    assert child1.q_value == 0.8
    assert child2.q_value == 0.5
    assert unvisited.uct_score() == float("inf")
    assert child1.uct_score(1.414) > child1.q_value


def test_mcts_loop_detection() -> None:
    """Test pruning heuristic for repetitive command loops."""
    engine = MCTSSearchEngine(config=EngineConfig())
    root = MCTSNode(node_id="root")
    node1 = MCTSNode(
        node_id="n1",
        parent=root,
        action=AgentAction(thought="check", command="systemctl status"),
    )
    node2 = MCTSNode(
        node_id="n2",
        parent=node1,
        action=AgentAction(thought="check again", command="systemctl status"),
    )

    # Identical command repeated 2 times
    repeated_action = AgentAction(thought="repeat", command="systemctl status")
    assert engine._detect_loop_or_fatal(node2, repeated_action) is True

    # Novel command
    novel_action = AgentAction(thought="fix", command="systemctl restart systemd-resolved")
    assert engine._detect_loop_or_fatal(node2, novel_action) is False


@pytest.mark.asyncio
async def test_mcts_search_workflow(engine_config: EngineConfig, tmp_path: Path) -> None:
    """Test end-to-end MCTS search execution and optimal trajectory extraction with MockSandbox."""
    engine_config.llm.mock_mode = True
    sandbox = MockSandbox("mcts-testbox")
    scenario = SystemdDNSScenario()

    search_engine = MCTSSearchEngine(
        config=engine_config,
        exploration_constant=1.414,
        max_depth=4,
        branch_factor=2,
    )

    traj = await search_engine.run_search(
        scenario=scenario,
        sandbox=sandbox,
        max_simulations=6,
    )

    assert traj is not None
    assert traj.scenario_name == "systemd_dns"
    assert len(traj.steps) >= 1
    assert traj.success is True

    out_file = tmp_path / "dataset_mcts.jsonl"
    saved_path = search_engine.save_optimal_trajectory(traj, output_file=out_file)
    assert saved_path.exists()

    record = json.loads(saved_path.read_text(encoding="utf-8").strip().splitlines()[0])
    assert record["scenario"] == "systemd_dns"
    assert record["success"] is True
    assert record["steps_count"] == len(traj.steps)
