"""
Ray distributed agent actors.
Each RayAgent runs as an independent remote actor, steps its local env,
and reports state back to a central coordinator.
SDKs: Ray, LangGraph, Gymnasium
"""
import ray
import numpy as np
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, asdict
import time


@dataclass
class AgentReport:
    agent_id: int
    step: int
    obs: List[float]
    reward: float
    done: bool
    info: Dict[str, Any]
    timestamp: float


@ray.remote(num_cpus=0.1)
class RaySimAgent:
    """
    Remote Ray actor running an independent agent in a local env slice.
    Communicates results back to coordinator via return values (not shared memory).
    """

    def __init__(self, agent_id: int, n_agents: int = 10, seed: int = 0):
        from envs.multi_agent_env import MultiAgentGridEnv
        self.agent_id = agent_id
        self.env = MultiAgentGridEnv(n_agents=n_agents, max_steps=500)
        self.obs, _ = self.env.reset(seed=seed + agent_id * 100)
        self.step_count = 0
        self.total_reward = 0.0

    def step(self, actions: Optional[np.ndarray] = None) -> AgentReport:
        """Step environment. Uses random policy if no actions provided."""
        if actions is None:
            actions = self.env.action_space.sample()

        obs, rewards, terminated, truncated, info = self.env.step(actions)
        self.obs = obs
        self.step_count += 1
        mean_reward = float(np.mean(rewards))
        self.total_reward += mean_reward

        done = bool(truncated) or bool(np.any(terminated))
        if done:
            self.obs, _ = self.env.reset()

        return AgentReport(
            agent_id=self.agent_id,
            step=self.step_count,
            obs=obs[0].tolist(),
            reward=mean_reward,
            done=done,
            info=info,
            timestamp=time.time(),
        )

    def get_stats(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "step_count": self.step_count,
            "total_reward": self.total_reward,
            "mean_reward": self.total_reward / max(self.step_count, 1),
        }

    def reset(self, seed: Optional[int] = None):
        self.obs, _ = self.env.reset(seed=seed)
        self.step_count = 0
        self.total_reward = 0.0


class RaySimCoordinator:
    """
    Manages a pool of RaySimAgents and runs parallel simulation steps.
    """

    def __init__(self, n_actors: int = 100, agents_per_actor: int = 10, seed: int = 42):
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)

        self.n_actors = n_actors
        print(f"[Ray] Spawning {n_actors} agent actors...")
        self.actors = [
            RaySimAgent.remote(i, agents_per_actor, seed)
            for i in range(n_actors)
        ]
        print(f"[Ray] {n_actors} actors ready ({n_actors * agents_per_actor} total agents)")

    def step_all(self, actions_list: Optional[List] = None) -> List[AgentReport]:
        """Step all actors in parallel. Returns list of AgentReports."""
        if actions_list is None:
            futures = [actor.step.remote() for actor in self.actors]
        else:
            futures = [
                actor.step.remote(actions_list[i] if i < len(actions_list) else None)
                for i, actor in enumerate(self.actors)
            ]
        return ray.get(futures)

    def run(self, n_steps: int = 1000, log_every: int = 100) -> Dict[str, Any]:
        """Run full simulation for n_steps, return summary stats."""
        all_rewards = []
        start = time.time()

        for step in range(n_steps):
            reports = self.step_all()
            step_rewards = [r.reward for r in reports]
            all_rewards.extend(step_rewards)

            if (step + 1) % log_every == 0:
                elapsed = time.time() - start
                sps = (step + 1) * self.n_actors / elapsed
                print(f"[Ray] Step {step+1}/{n_steps} | mean_reward={np.mean(step_rewards):.3f} | {sps:.0f} steps/sec")

        stats_futures = [actor.get_stats.remote() for actor in self.actors]
        actor_stats = ray.get(stats_futures)

        return {
            "n_actors": self.n_actors,
            "n_steps": n_steps,
            "total_steps": n_steps * self.n_actors,
            "mean_reward": float(np.mean(all_rewards)),
            "elapsed_sec": time.time() - start,
            "actor_stats": actor_stats,
        }

    def shutdown(self):
        ray.shutdown()
        print("[Ray] Coordinator shut down")
