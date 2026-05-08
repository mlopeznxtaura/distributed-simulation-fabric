"""
Gymnasium-compatible multi-agent environment.
100-10000 agents running in parallel via Ray or GPU batch.
Each agent has position, velocity, goal, and a simple physics model.
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field


@dataclass
class AgentState:
    id: int
    position: np.ndarray    # (2,) x, y
    velocity: np.ndarray    # (2,)
    goal: np.ndarray        # (2,) target position
    energy: float = 1.0
    alive: bool = True
    step_count: int = 0


class MultiAgentGridEnv(gym.Env):
    """
    N-agent 2D grid environment. Each agent navigates toward a goal
    while avoiding collisions. Scales from 100 to 100k agents.

    Observation per agent: [pos(2), vel(2), goal_rel(2), nearest_neighbor_rel(2)] = 8-dim
    Action per agent: continuous velocity delta (2,)
    Reward: -dist_to_goal + collision_penalty + step_penalty
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        n_agents: int = 100,
        world_size: float = 100.0,
        max_speed: float = 2.0,
        collision_radius: float = 1.0,
        max_steps: int = 500,
        render_mode: Optional[str] = None,
    ):
        super().__init__()
        self.n_agents = n_agents
        self.world_size = world_size
        self.max_speed = max_speed
        self.collision_radius = collision_radius
        self.max_steps = max_steps
        self.render_mode = render_mode

        obs_dim = 8
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(n_agents, obs_dim), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(n_agents, 2), dtype=np.float32
        )

        self.positions = np.zeros((n_agents, 2), dtype=np.float32)
        self.velocities = np.zeros((n_agents, 2), dtype=np.float32)
        self.goals = np.zeros((n_agents, 2), dtype=np.float32)
        self.alive = np.ones(n_agents, dtype=bool)
        self._step_count = 0

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        rng = np.random.default_rng(seed)

        self.positions = rng.uniform(0, self.world_size, size=(self.n_agents, 2)).astype(np.float32)
        self.velocities = np.zeros((self.n_agents, 2), dtype=np.float32)
        self.goals = rng.uniform(0, self.world_size, size=(self.n_agents, 2)).astype(np.float32)
        self.alive = np.ones(self.n_agents, dtype=bool)
        self._step_count = 0

        return self._get_obs(), {}

    def _get_obs(self) -> np.ndarray:
        # Nearest neighbor for each agent
        nn_rel = np.zeros((self.n_agents, 2), dtype=np.float32)
        for i in range(self.n_agents):
            diffs = self.positions - self.positions[i]
            dists = np.linalg.norm(diffs, axis=1)
            dists[i] = np.inf
            nn_idx = np.argmin(dists)
            nn_rel[i] = diffs[nn_idx]

        goal_rel = self.goals - self.positions
        obs = np.concatenate([
            self.positions / self.world_size,
            self.velocities / self.max_speed,
            goal_rel / self.world_size,
            nn_rel / self.world_size,
        ], axis=1).astype(np.float32)
        return obs

    def step(self, actions: np.ndarray):
        actions = np.clip(actions, -1.0, 1.0) * self.max_speed
        self.velocities = np.clip(self.velocities + actions * 0.1, -self.max_speed, self.max_speed)
        self.positions = np.clip(self.positions + self.velocities * 0.1, 0, self.world_size)
        self._step_count += 1

        # Rewards
        dist_to_goal = np.linalg.norm(self.goals - self.positions, axis=1)
        rewards = -dist_to_goal / self.world_size

        # Collision penalty
        for i in range(self.n_agents):
            dists = np.linalg.norm(self.positions - self.positions[i], axis=1)
            dists[i] = np.inf
            if np.any(dists < self.collision_radius):
                rewards[i] -= 1.0

        # Success bonus
        reached = dist_to_goal < self.collision_radius
        rewards[reached] += 10.0
        if reached.any():
            rng = np.random.default_rng()
            self.goals[reached] = rng.uniform(0, self.world_size, size=(reached.sum(), 2))

        terminated = np.zeros(self.n_agents, dtype=bool)
        truncated = self._step_count >= self.max_steps
        obs = self._get_obs()

        info = {
            "n_reached": int(reached.sum()),
            "mean_dist_to_goal": float(dist_to_goal.mean()),
            "step": self._step_count,
        }
        return obs, rewards, terminated, truncated, info

    def render(self):
        if self.render_mode != "rgb_array":
            return
        import cv2
        size = 600
        frame = np.zeros((size, size, 3), dtype=np.uint8)
        scale = size / self.world_size
        for i in range(self.n_agents):
            px = int(self.positions[i, 0] * scale)
            py = int(self.positions[i, 1] * scale)
            gx = int(self.goals[i, 0] * scale)
            gy = int(self.goals[i, 1] * scale)
            cv2.circle(frame, (px, py), 3, (100, 200, 255), -1)
            cv2.line(frame, (px, py), (gx, gy), (60, 60, 60), 1)
        return frame
