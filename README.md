# Distributed Simulation Fabric

Cluster 08 of the NextAura 500 SDKs / 25 Clusters project.

Planet-scale event-driven simulation across thousands of concurrent agents. Every agent action is an event. Every event is queryable.

## Architecture

- Ray for distributed agent execution across GPU cluster
- NVIDIA Warp + CuPy for GPU-parallelized state updates
- LangGraph agents for high-level decision-making
- Kafka + Flink for event stream processing
- NATS for low-latency agent-to-agent messaging
- Temporal for long-running workflow orchestration
- DuckDB + Arrow + Polars for simulation history analytics
- Prometheus + Grafana for real-time observability

## SDKs Used

Ray SDK, NVIDIA Warp, MuJoCo SDK, Gymnasium, LangGraph, Kafka SDK, Apache Flink SDK, Redis SDK, NATS SDK, gRPC SDK, Temporal SDK, DuckDB, Apache Arrow, Polars, Prometheus Client, Grafana SDK, FastAPI, Pydantic, CuPy SDK, Numba SDK

## Quickstart

```bash
pip install -r requirements.txt
python main.py --agents 100 --steps 1000 --mode ray
python main.py --agents 10000 --steps 500 --mode gpu  # GPU parallel via Warp/CuPy
python main.py --mode serve  # start API + metrics server
```

## Structure

```
agents/        Ray actors, LangGraph decision agents, GPU state batch
envs/          Gymnasium-compatible multi-agent environments
events/        Kafka producer/consumer, Flink stream processing
messaging/     NATS pub/sub for agent-to-agent comms
workflows/     Temporal workflow definitions
storage/       DuckDB + Arrow + Polars simulation history
observability/ Prometheus metrics, Grafana dashboards
api/           FastAPI control plane
main.py        Entry point
```
