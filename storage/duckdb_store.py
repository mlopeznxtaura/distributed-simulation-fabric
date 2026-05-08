"""
DuckDB + Apache Arrow + Polars simulation history storage.
Query millions of agent-step records with SQL in milliseconds.
SDKs: DuckDB, PyArrow, Polars
"""
import os
import time
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import polars as pl
import duckdb
from pathlib import Path
from typing import Optional, List, Dict, Any


class SimHistoryStore:
    """
    Persistent simulation history using DuckDB over Parquet files.
    Write path: Arrow IPC -> Parquet. Query path: DuckDB SQL over Parquet.
    Handles hundreds of millions of agent-step records efficiently.
    """

    def __init__(self, data_dir: str = "./sim_data", db_path: str = ":memory:"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.db = duckdb.connect(db_path)
        self._setup_schema()
        self._buffer: List[dict] = []
        self._buffer_size = 100_000
        print(f"[DuckDB] Store initialized at {data_dir}")

    def _setup_schema(self):
        self.db.execute("""
            CREATE TABLE IF NOT EXISTS sim_runs (
                sim_id VARCHAR,
                n_agents INTEGER,
                n_steps INTEGER,
                mean_reward DOUBLE,
                elapsed_sec DOUBLE,
                started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self.db.execute("""
            CREATE TABLE IF NOT EXISTS agent_steps (
                sim_id VARCHAR,
                step INTEGER,
                agent_id INTEGER,
                pos_x FLOAT,
                pos_y FLOAT,
                vel_x FLOAT,
                vel_y FLOAT,
                reward FLOAT,
                ts DOUBLE
            )
        """)

    def write_snapshot(
        self,
        sim_id: str,
        step: int,
        positions: np.ndarray,
        velocities: np.ndarray,
        rewards: np.ndarray,
    ):
        """Buffer a state snapshot. Flushes to Parquet when buffer is full."""
        n = len(positions)
        for i in range(n):
            self._buffer.append({
                "sim_id": sim_id,
                "step": step,
                "agent_id": i,
                "pos_x": float(positions[i, 0]),
                "pos_y": float(positions[i, 1]),
                "vel_x": float(velocities[i, 0]),
                "vel_y": float(velocities[i, 1]),
                "reward": float(rewards[i]),
                "ts": time.time(),
            })
        if len(self._buffer) >= self._buffer_size:
            self.flush()

    def flush(self):
        """Write buffered records to Parquet."""
        if not self._buffer:
            return
        df = pl.DataFrame(self._buffer)
        timestamp = int(time.time() * 1000)
        path = self.data_dir / f"steps_{timestamp}.parquet"
        df.write_parquet(str(path))
        self._buffer.clear()
        print(f"[DuckDB] Flushed {len(df)} records to {path.name}")

    def register_sim_run(self, sim_id: str, n_agents: int, n_steps: int,
                          mean_reward: float, elapsed_sec: float):
        self.db.execute(
            "INSERT INTO sim_runs VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)",
            [sim_id, n_agents, n_steps, mean_reward, elapsed_sec]
        )

    def query(self, sql: str) -> pl.DataFrame:
        """Run SQL query over all Parquet files in data_dir."""
        parquet_glob = str(self.data_dir / "*.parquet")
        self.db.execute(f"CREATE OR REPLACE VIEW steps AS SELECT * FROM read_parquet('{parquet_glob}')")
        result = self.db.execute(sql).fetchdf()
        return pl.from_pandas(result)

    def get_run_summary(self, sim_id: str) -> Dict[str, Any]:
        sql = f"""
            SELECT
                COUNT(*) as total_steps,
                AVG(reward) as mean_reward,
                MIN(reward) as min_reward,
                MAX(reward) as max_reward,
                COUNT(DISTINCT agent_id) as n_agents,
                MAX(step) as max_step
            FROM steps
            WHERE sim_id = '{sim_id}'
        """
        df = self.query(sql)
        if len(df) > 0:
            return df.row(0, named=True)
        return {}

    def get_top_agents(self, sim_id: str, n: int = 10) -> pl.DataFrame:
        """Return top N agents by cumulative reward."""
        sql = f"""
            SELECT agent_id, SUM(reward) as total_reward, COUNT(*) as steps
            FROM steps
            WHERE sim_id = '{sim_id}'
            GROUP BY agent_id
            ORDER BY total_reward DESC
            LIMIT {n}
        """
        return self.query(sql)

    def export_trajectory(self, sim_id: str, agent_id: int) -> pl.DataFrame:
        """Export full position/velocity trajectory for a single agent."""
        sql = f"""
            SELECT step, pos_x, pos_y, vel_x, vel_y, reward
            FROM steps
            WHERE sim_id = '{sim_id}' AND agent_id = {agent_id}
            ORDER BY step
        """
        return self.query(sql)
