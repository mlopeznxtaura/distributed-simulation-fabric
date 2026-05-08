"""
Kafka event streaming for simulation state.
Every agent action, state change, and milestone is an event on a Kafka topic.
SDKs: kafka-python, Apache Arrow
"""
import json
import time
import threading
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timezone

import numpy as np
import pyarrow as pa

try:
    from kafka import KafkaProducer, KafkaConsumer
    from kafka.errors import NoBrokersAvailable
    KAFKA_AVAILABLE = True
except ImportError:
    KAFKA_AVAILABLE = False
    print("Warning: kafka-python not available. Install: pip install kafka-python")


@dataclass
class SimEvent:
    event_type: str      # "agent_step", "agent_collision", "goal_reached", "sim_tick"
    sim_id: str
    step: int
    timestamp: float
    payload: Dict[str, Any]

    def to_json(self) -> bytes:
        d = asdict(self)
        d["timestamp_iso"] = datetime.fromtimestamp(self.timestamp, tz=timezone.utc).isoformat()
        return json.dumps(d).encode()


# Arrow schema for batch state snapshots
AGENT_STATE_SCHEMA = pa.schema([
    pa.field("sim_id", pa.string()),
    pa.field("step", pa.int32()),
    pa.field("agent_id", pa.int32()),
    pa.field("pos_x", pa.float32()),
    pa.field("pos_y", pa.float32()),
    pa.field("vel_x", pa.float32()),
    pa.field("vel_y", pa.float32()),
    pa.field("reward", pa.float32()),
    pa.field("timestamp", pa.float64()),
])


class SimEventProducer:
    """Publish simulation events to Kafka topics."""

    TOPICS = {
        "steps": "sim.agent.steps",
        "collisions": "sim.agent.collisions",
        "goals": "sim.agent.goals",
        "ticks": "sim.ticks",
        "snapshots": "sim.snapshots",
    }

    def __init__(self, bootstrap_servers: str = "localhost:9092", sim_id: str = "sim_01"):
        self.sim_id = sim_id
        self._queue: List[SimEvent] = []
        self.producer = None

        if KAFKA_AVAILABLE:
            try:
                self.producer = KafkaProducer(
                    bootstrap_servers=bootstrap_servers,
                    value_serializer=lambda v: v,
                    acks=1,
                    linger_ms=5,
                    batch_size=65536,
                )
                print(f"[Kafka] Producer connected: {bootstrap_servers}")
            except NoBrokersAvailable:
                print(f"[Kafka] No brokers at {bootstrap_servers}. Using in-memory queue.")

    def emit(self, event_type: str, step: int, payload: Dict):
        event = SimEvent(
            event_type=event_type,
            sim_id=self.sim_id,
            step=step,
            timestamp=time.time(),
            payload=payload,
        )
        topic = self.TOPICS.get(event_type.split("_")[0], "sim.misc")
        if self.producer:
            self.producer.send(topic, value=event.to_json())
        else:
            self._queue.append(event)

    def emit_batch_snapshot(
        self,
        step: int,
        positions: np.ndarray,
        velocities: np.ndarray,
        rewards: np.ndarray,
    ):
        """Publish a full agent state snapshot as Arrow IPC for efficiency."""
        n = len(positions)
        table = pa.table({
            "sim_id": [self.sim_id] * n,
            "step": np.full(n, step, dtype=np.int32),
            "agent_id": np.arange(n, dtype=np.int32),
            "pos_x": positions[:, 0].astype(np.float32),
            "pos_y": positions[:, 1].astype(np.float32),
            "vel_x": velocities[:, 0].astype(np.float32),
            "vel_y": velocities[:, 1].astype(np.float32),
            "reward": rewards.astype(np.float32),
            "timestamp": np.full(n, time.time(), dtype=np.float64),
        }, schema=AGENT_STATE_SCHEMA)

        sink = pa.BufferOutputStream()
        with pa.ipc.new_stream(sink, table.schema) as writer:
            writer.write_table(table)
        buf = sink.getvalue().to_pybytes()

        if self.producer:
            self.producer.send(self.TOPICS["snapshots"], value=buf)

    def flush(self):
        if self.producer:
            self.producer.flush()

    def close(self):
        if self.producer:
            self.producer.close()

    def drain_queue(self) -> List[SimEvent]:
        """Return and clear in-memory fallback queue."""
        q = self._queue.copy()
        self._queue.clear()
        return q


class SimEventConsumer:
    """Consume and process simulation events from Kafka."""

    def __init__(
        self,
        bootstrap_servers: str = "localhost:9092",
        topics: Optional[List[str]] = None,
        group_id: str = "sim-analytics",
    ):
        self.topics = topics or list(SimEventProducer.TOPICS.values())
        self.consumer = None

        if KAFKA_AVAILABLE:
            try:
                self.consumer = KafkaConsumer(
                    *self.topics,
                    bootstrap_servers=bootstrap_servers,
                    group_id=group_id,
                    auto_offset_reset="latest",
                    value_deserializer=lambda v: json.loads(v.decode()),
                    consumer_timeout_ms=1000,
                )
                print(f"[Kafka] Consumer subscribed to {self.topics}")
            except NoBrokersAvailable:
                print("[Kafka] Consumer: no brokers available")

    def consume(self, handler: Callable[[dict], None], max_messages: int = 1000):
        """Consume messages and call handler for each."""
        if not self.consumer:
            return
        count = 0
        for msg in self.consumer:
            handler(msg.value)
            count += 1
            if count >= max_messages:
                break

    def close(self):
        if self.consumer:
            self.consumer.close()
