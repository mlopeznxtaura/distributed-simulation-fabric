"""
Prometheus metrics for simulation observability.
Tracks agent throughput, reward distributions, Kafka lag, and GPU utilization.
SDKs: prometheus-client, Grafana SDK
"""
import time
import threading
from typing import Optional
from prometheus_client import (
    Counter, Gauge, Histogram, Summary,
    start_http_server, CollectorRegistry, REGISTRY,
)

# Simulation metrics
SIM_AGENT_STEPS = Counter(
    "sim_agent_steps_total",
    "Total agent-environment steps across all simulations",
    ["sim_id", "mode"],
)

SIM_AGENT_REWARD = Histogram(
    "sim_agent_reward",
    "Per-step agent reward distribution",
    ["sim_id"],
    buckets=[-10, -5, -2, -1, -0.5, -0.1, 0, 0.5, 1, 5, 10],
)

SIM_AGENTS_ACTIVE = Gauge(
    "sim_agents_active",
    "Number of currently active agents",
    ["sim_id"],
)

SIM_GOALS_REACHED = Counter(
    "sim_goals_reached_total",
    "Total goals reached by agents",
    ["sim_id"],
)

SIM_STEP_LATENCY = Histogram(
    "sim_step_latency_seconds",
    "Wall-clock time per simulation step",
    ["mode"],
    buckets=[0.0001, 0.001, 0.01, 0.05, 0.1, 0.5, 1.0],
)

KAFKA_EVENTS_PUBLISHED = Counter(
    "kafka_events_published_total",
    "Total events published to Kafka",
    ["topic"],
)

RAY_ACTORS_ACTIVE = Gauge(
    "ray_actors_active",
    "Number of active Ray simulation actors",
)

GPU_AGENT_STEPS_PER_SEC = Gauge(
    "gpu_agent_steps_per_sec",
    "GPU agent-steps per second (CuPy/Warp backend)",
)

DUCKDB_RECORDS_WRITTEN = Counter(
    "duckdb_records_written_total",
    "Total agent-step records written to DuckDB/Parquet",
    ["sim_id"],
)


class SimMetricsCollector:
    """
    Central metrics collector. Call update_* methods from simulation loop.
    Exposes /metrics endpoint for Prometheus scraping.
    """

    def __init__(self, port: int = 9090, sim_id: str = "sim_01", mode: str = "ray"):
        self.sim_id = sim_id
        self.mode = mode
        self.port = port
        self._server_started = False

    def start_server(self):
        """Start Prometheus metrics HTTP server."""
        if not self._server_started:
            start_http_server(self.port)
            self._server_started = True
            print(f"[Prometheus] Metrics server on :{self.port}/metrics")

    def record_step(self, n_agents: int, mean_reward: float, n_goals: int, latency: float):
        SIM_AGENT_STEPS.labels(sim_id=self.sim_id, mode=self.mode).inc(n_agents)
        SIM_AGENT_REWARD.labels(sim_id=self.sim_id).observe(mean_reward)
        SIM_AGENTS_ACTIVE.labels(sim_id=self.sim_id).set(n_agents)
        SIM_GOALS_REACHED.labels(sim_id=self.sim_id).inc(n_goals)
        SIM_STEP_LATENCY.labels(mode=self.mode).observe(latency)

    def record_kafka_publish(self, topic: str, n_events: int = 1):
        KAFKA_EVENTS_PUBLISHED.labels(topic=topic).inc(n_events)

    def set_ray_actors(self, n: int):
        RAY_ACTORS_ACTIVE.set(n)

    def set_gpu_throughput(self, agent_steps_per_sec: float):
        GPU_AGENT_STEPS_PER_SEC.set(agent_steps_per_sec)

    def record_duckdb_write(self, n_records: int):
        DUCKDB_RECORDS_WRITTEN.labels(sim_id=self.sim_id).inc(n_records)
