"""
distributed-simulation-fabric — Entry Point

Planet-scale multi-agent simulation with Ray distributed actors,
GPU batch stepping, Kafka event streaming, and DuckDB analytics.

Usage:
  python main.py --agents 100 --steps 1000 --mode ray
  python main.py --agents 10000 --steps 500 --mode gpu
  python main.py --agents 1000 --steps 200 --mode gpu --kafka localhost:9092
"""
import argparse
import time
import uuid
import numpy as np
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Distributed Simulation Fabric")
    parser.add_argument("--agents", type=int, default=100, help="Number of agents")
    parser.add_argument("--steps", type=int, default=1000, help="Simulation steps")
    parser.add_argument("--mode", default="gpu", choices=["ray", "gpu", "cpu"],
                        help="Execution backend")
    parser.add_argument("--actors", type=int, default=None,
                        help="Ray actors (mode=ray). Default: agents//10")
    parser.add_argument("--kafka", type=str, default=None,
                        help="Kafka bootstrap server e.g. localhost:9092")
    parser.add_argument("--output", type=str, default="./sim_data")
    parser.add_argument("--metrics-port", type=int, default=9090)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--sim-id", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    sim_id = args.sim_id or f"sim_{int(time.time())}"

    print("=" * 60)
    print("  Distributed Simulation Fabric")
    print(f"  Agents: {args.agents:,} | Steps: {args.steps:,} | Mode: {args.mode.upper()}")
    print(f"  Sim ID: {sim_id}")
    print("=" * 60)

    # Observability
    from observability.metrics import SimMetricsCollector
    metrics = SimMetricsCollector(port=args.metrics_port, sim_id=sim_id, mode=args.mode)
    metrics.start_server()

    # Storage
    from storage.duckdb_store import SimHistoryStore
    store = SimHistoryStore(data_dir=args.output)

    # Kafka event producer (optional)
    producer = None
    if args.kafka:
        from events.kafka_stream import SimEventProducer
        producer = SimEventProducer(bootstrap_servers=args.kafka, sim_id=sim_id)

    start = time.time()

    if args.mode == "ray":
        n_actors = args.actors or max(1, args.agents // 10)
        agents_per_actor = args.agents // n_actors

        from agents.ray_actor import RaySimCoordinator
        metrics.set_ray_actors(n_actors)
        coordinator = RaySimCoordinator(n_actors=n_actors, agents_per_actor=agents_per_actor)
        result = coordinator.run(n_steps=args.steps, log_every=args.log_every)

        if producer:
            producer.emit("ticks", args.steps, result)
            producer.flush()

        coordinator.shutdown()

    elif args.mode in ("gpu", "cpu"):
        from agents.gpu_batch_agent import GPUBatchSimulator
        sim = GPUBatchSimulator(n_agents=args.agents)

        all_rewards = []
        step_start = time.time()

        for step in range(args.steps):
            t0 = time.time()
            info = sim.step()
            latency = time.time() - t0

            all_rewards.append(info["mean_reward"])
            metrics.record_step(
                args.agents, info["mean_reward"],
                info["n_reached"], latency
            )

            # Snapshot every 50 steps
            if step % 50 == 0:
                state = sim.get_state_numpy()
                rewards_np = np.full(args.agents, info["mean_reward"], dtype=np.float32)
                store.write_snapshot(sim_id, step, state["positions"], state["velocities"], rewards_np)
                if producer:
                    producer.emit_batch_snapshot(step, state["positions"], state["velocities"], rewards_np)
                    metrics.record_kafka_publish("sim.snapshots")

            if (step + 1) % args.log_every == 0:
                elapsed = time.time() - step_start
                aps = (step + 1) * args.agents / elapsed
                metrics.set_gpu_throughput(aps)
                print(f"Step {step+1}/{args.steps} | reward={info['mean_reward']:.3f} | {aps/1e6:.3f}M agent-steps/sec")

        result = {
            "n_agents": args.agents,
            "n_steps": args.steps,
            "mean_reward": float(np.mean(all_rewards)),
            "elapsed_sec": time.time() - start,
        }

    store.flush()
    elapsed = time.time() - start
    store.register_sim_run(sim_id, args.agents, args.steps, result["mean_reward"], elapsed)

    if producer:
        producer.close()

    total_steps = args.agents * args.steps
    print(f"
Done. {total_steps:,} agent-steps in {elapsed:.1f}s ({total_steps/elapsed/1e6:.3f}M/sec)")
    print(f"Mean reward: {result['mean_reward']:.4f}")
    print(f"Data: {args.output}")


if __name__ == "__main__":
    main()
