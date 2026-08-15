"""Monte Carlo Tree Search (MCTS) trajectory collection engine over Incus snapshot branches."""

from __future__ import annotations

import json
import logging
import math
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from config.settings import EngineConfig
from engine.action_schema import AgentAction
from engine.client import LLMClientError, PolicyClient
from sandbox.base import BaseSandbox
from scenarios.base_scenario import BaseScenario
from trainer.trajectory_buffer import EpisodeTrajectory, TrajectoryStep

logger = logging.getLogger("os_autofix.trainer.mcts")


@dataclass
class MCTSNode:
    """Node in the Monte Carlo search tree representing an OS state snapshot."""

    node_id: str = field(default_factory=lambda: f"node-{uuid.uuid4().hex[:8]}")
    snapshot_id: str = ""
    parent: MCTSNode | None = None
    action: AgentAction | None = None
    observation: str = ""
    depth: int = 0
    visit_count: int = 0
    total_reward: float = 0.0
    children: list[MCTSNode] = field(default_factory=list)
    is_terminal: bool = False
    is_verified: bool = False
    is_pruned: bool = False
    raw_completion: str = ""
    verification_message: str = ""
    exit_code: int = 0

    @property
    def q_value(self) -> float:
        """Average reward / state-action value (Q)."""
        if self.visit_count == 0:
            return 0.0
        return self.total_reward / self.visit_count

    def uct_score(self, exploration_constant: float = 1.414) -> float:
        """Calculate Upper Confidence Bound for Trees (UCT)."""
        if self.visit_count == 0:
            return float("inf")
        if not self.parent or self.parent.visit_count == 0:
            return self.q_value

        exploration = exploration_constant * math.sqrt(
            math.log(self.parent.visit_count) / self.visit_count
        )
        return self.q_value + exploration


class MCTSSearchEngine:
    """Autonomous MCTS engine searching the shortest-path command trajectory to resolve an OS fault."""

    def __init__(
        self,
        config: EngineConfig,
        client: PolicyClient | None = None,
        exploration_constant: float = 1.414,
        max_depth: int = 6,
        branch_factor: int = 3,
        step_penalty: float = 0.05,
    ) -> None:
        self.config = config
        self.client = client or PolicyClient(config.llm)
        self.exploration_constant = exploration_constant
        self.max_depth = max_depth
        self.branch_factor = branch_factor
        self.step_penalty = step_penalty

    def _select_child(self, node: MCTSNode) -> MCTSNode:
        """Select best child according to UCT score, skipping pruned branches."""
        active_children = [c for c in node.children if not c.is_pruned]
        if not active_children:
            return node
        return max(active_children, key=lambda c: c.uct_score(self.exploration_constant))

    def _build_conversation(
        self,
        scenario: BaseScenario,
        node: MCTSNode,
    ) -> list[dict[str, str]]:
        """Construct the prompt conversation leading up to the given tree node."""
        path: list[MCTSNode] = []
        curr: MCTSNode | None = node
        while curr and curr.parent:
            path.append(curr)
            curr = curr.parent
        path.reverse()

        messages: list[dict[str, str]] = [
            {"role": "system", "content": scenario.get_prompt()},
            {
                "role": "user",
                "content": f"OBSERVATION:\nSystem fault injected: {scenario.description}",
            },
        ]

        for step_node in path:
            if step_node.action:
                completion = step_node.raw_completion or json.dumps(
                    {
                        "thought": step_node.action.thought,
                        "command": step_node.action.command,
                        "is_done": step_node.action.is_done,
                        "confidence": step_node.action.confidence,
                    }
                )
                messages.append({"role": "assistant", "content": completion})
                messages.append(
                    {
                        "role": "user",
                        "content": f"[EXIT CODE: {step_node.exit_code}]\nSTDOUT:\n{step_node.observation or '[No output]'}",
                    }
                )

        return messages

    async def _generate_candidate_actions(
        self,
        messages: list[dict[str, str]],
        count: int,
    ) -> list[tuple[AgentAction, str]]:
        """Generate K diverse candidate actions using policy LLM with temperature sampling."""
        candidates: list[tuple[AgentAction, str]] = []
        seen_commands: set[str] = set()

        for _ in range(count):
            try:
                action, raw = await self.client.get_next_action(messages)
                cmd_norm = action.command.strip()
                if cmd_norm and cmd_norm not in seen_commands:
                    seen_commands.add(cmd_norm)
                    candidates.append((action, raw))
                elif not candidates:
                    candidates.append((action, raw))
            except LLMClientError as e:
                logger.warning("LLM client error during candidate generation: %s", e)
                break

        return candidates

    def _detect_loop_or_fatal(self, node: MCTSNode, new_action: AgentAction) -> bool:
        """Pruning heuristic: detect command repetition loops or empty commands."""
        if not new_action.command.strip():
            return True

        # Check ancestors for identical command
        curr: MCTSNode | None = node
        repeat_count = 0
        while curr:
            if curr.action and curr.action.command.strip() == new_action.command.strip():
                repeat_count += 1
            curr = curr.parent

        return repeat_count >= 2

    async def run_search(
        self,
        scenario: BaseScenario,
        sandbox: BaseSandbox,
        max_simulations: int = 12,
    ) -> EpisodeTrajectory | None:
        """Execute Monte Carlo Tree Search to discover the optimal solution trajectory."""
        logger.info(
            "Starting MCTS trajectory search for scenario '%s' (max_simulations=%d, max_depth=%d)...",
            scenario.name,
            max_simulations,
            self.max_depth,
        )

        # 1. Environment initialization & baseline snapshots
        await sandbox.setup()
        await scenario.setup(sandbox)
        root_snapshot = "snap-mcts-root"
        await sandbox.create_snapshot(root_snapshot)

        await scenario.inject_fault(sandbox)
        fault_snapshot = "snap-mcts-fault"
        await sandbox.create_snapshot(fault_snapshot)

        root = MCTSNode(
            node_id="root",
            snapshot_id=fault_snapshot,
            observation=f"Fault injected: {scenario.description}",
            depth=0,
        )

        best_terminal_node: MCTSNode | None = None
        start_time = time.monotonic()

        # 2. Main Simulation Loop
        for sim_idx in range(1, max_simulations + 1):
            logger.debug("--- MCTS Simulation %d/%d ---", sim_idx, max_simulations)

            # A. Selection: Traverse down the tree using UCT
            current = root
            while current.children and not current.is_terminal:
                current = self._select_child(current)

            # B. Expansion & Execution: If current node is non-terminal, expand candidate branches
            simulation_reward = 0.0
            if not current.is_terminal and current.depth < self.max_depth:
                conv = self._build_conversation(scenario, current)
                candidates = await self._generate_candidate_actions(conv, self.branch_factor)

                for action, raw_comp in candidates:
                    if self._detect_loop_or_fatal(current, action):
                        pruned_child = MCTSNode(
                            parent=current,
                            action=action,
                            depth=current.depth + 1,
                            is_pruned=True,
                            raw_completion=raw_comp,
                        )
                        current.children.append(pruned_child)
                        continue

                    # Restore state to current parent snapshot
                    await sandbox.revert(current.snapshot_id)

                    # Execute candidate command
                    exec_res = await sandbox.execute(
                        action.command,
                        timeout_seconds=action.timeout_seconds
                        or self.config.incus.command_timeout_seconds,
                    )

                    # Check resolution
                    is_resolved, verify_msg = await scenario.verify(sandbox)
                    child_depth = current.depth + 1
                    child_snapshot = f"snap-mcts-d{child_depth}-{uuid.uuid4().hex[:6]}"
                    await sandbox.create_snapshot(child_snapshot)

                    is_terminal = is_resolved or action.is_done or (child_depth >= self.max_depth)
                    reward = (
                        1.0 - (child_depth * self.step_penalty)
                        if is_resolved
                        else -self.step_penalty
                    )

                    child_node = MCTSNode(
                        snapshot_id=child_snapshot,
                        parent=current,
                        action=action,
                        observation=exec_res.combined_output,
                        depth=child_depth,
                        is_terminal=is_terminal,
                        is_verified=is_resolved,
                        raw_completion=raw_comp,
                        verification_message=verify_msg,
                        exit_code=exec_res.exit_code,
                        total_reward=reward,
                        visit_count=1,
                    )
                    current.children.append(child_node)

                    if is_resolved and (
                        not best_terminal_node or child_node.depth < best_terminal_node.depth
                    ):
                        best_terminal_node = child_node
                        logger.info(
                            "MCTS found verified solution at depth %d in simulation %d!",
                            child_depth,
                            sim_idx,
                        )

                    simulation_reward = max(simulation_reward, reward)
            else:
                simulation_reward = current.total_reward

            # C. Backpropagation: Propagate simulation outcome up to root
            back_node: MCTSNode | None = current
            while back_node:
                back_node.visit_count += 1
                back_node.total_reward += simulation_reward
                back_node = back_node.parent

        # 3. Path Extraction: Extract optimal trajectory
        duration = time.monotonic() - start_time
        target_leaf = best_terminal_node

        if not target_leaf:
            # Fallback to the most visited leaf node
            curr = root
            while curr.children:
                curr = max(curr.children, key=lambda c: c.visit_count)
            target_leaf = curr

        optimal_steps: list[TrajectoryStep] = []
        path_nodes: list[MCTSNode] = []
        curr_node: MCTSNode | None = target_leaf
        while curr_node and curr_node.parent:
            path_nodes.append(curr_node)
            curr_node = curr_node.parent
        path_nodes.reverse()

        step_idx = 1
        cum_reward = 0.0
        for n in path_nodes:
            if n.action:
                cum_reward += n.total_reward
                optimal_steps.append(
                    TrajectoryStep(
                        step_index=step_idx,
                        state_observation=n.observation,
                        thought=n.action.thought,
                        command=n.action.command,
                        timeout_seconds=n.action.timeout_seconds,
                        stdout=n.observation,
                        stderr="",
                        exit_code=n.exit_code,
                        reward=n.total_reward,
                        done=n.is_terminal,
                        raw_model_completion=n.raw_completion,
                    )
                )
                step_idx += 1

        inst_id = getattr(sandbox, "instance_name", getattr(sandbox, "name", "mcts-sandbox"))
        trajectory = EpisodeTrajectory(
            scenario_name=scenario.name,
            instance_id=inst_id,
            steps=optimal_steps,
            success=target_leaf.is_verified,
            total_reward=cum_reward,
            duration_seconds=duration,
            verification_message=target_leaf.verification_message,
        )

        return trajectory

    def save_optimal_trajectory(
        self,
        trajectory: EpisodeTrajectory,
        output_file: Path | str = "data/dataset_mcts_optimal.jsonl",
    ) -> Path:
        """Append the optimal extracted MCTS trajectory to a JSONL dataset file."""
        out_path = Path(output_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        record = {
            "scenario": trajectory.scenario_name,
            "success": trajectory.success,
            "total_reward": trajectory.total_reward,
            "steps_count": len(trajectory.steps),
            "duration_seconds": trajectory.duration_seconds,
            "trajectory": [
                {
                    "step": s.step_index,
                    "thought": s.thought,
                    "command": s.command,
                    "observation": s.state_observation,
                    "reward": s.reward,
                }
                for s in trajectory.steps
            ],
        }

        with open(out_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

        return out_path
