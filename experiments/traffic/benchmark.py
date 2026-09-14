from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import pandas as pd

from experiments.traffic.path_algorithms import (
    astar_search,
    bidirectional_dijkstra_search,
    dijkstra_search,
)


@dataclass(slots=True)
class BenchmarkResult:
    results_path: Path
    summary_path: Path


def sample_od_pairs(g: nx.Graph, n_pairs: int, *, seed: int = 42) -> list[tuple[int, int]]:
    rng = random.Random(seed)
    nodes = list(g.nodes())
    if len(nodes) < 2:
        return []
    pairs: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    attempts = 0
    max_attempts = n_pairs * 20
    while len(pairs) < n_pairs and attempts < max_attempts:
        s, t = rng.sample(nodes, 2)
        pair = (int(s), int(t))
        attempts += 1
        if pair in seen:
            continue
        seen.add(pair)
        pairs.append(pair)
    return pairs


def run_benchmark(
    graph_path: str | Path,
    output_dir: str | Path,
    *,
    n_pairs: int = 100,
    seed: int = 42,
) -> BenchmarkResult:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    g = nx.convert_node_labels_to_integers(nx.read_graphml(graph_path))
    pairs = sample_od_pairs(g, n_pairs=n_pairs, seed=seed)
    rows: list[dict[str, float | int | str]] = []
    algorithms = {
        "dijkstra": dijkstra_search,
        "astar": astar_search,
        "bidirectional_dijkstra": bidirectional_dijkstra_search,
    }
    for source, target in pairs:
        for name, fn in algorithms.items():
            t1 = time.perf_counter()
            result = fn(g, source, target)
            t2 = time.perf_counter()
            rows.append(
                {
                    "algorithm": name,
                    "source": source,
                    "target": target,
                    "found": int(result.found),
                    "cost": float(result.cost),
                    "expanded_nodes": int(result.expanded_nodes),
                    "elapsed_ms": float((t2 - t1) * 1000.0),
                    "path_hops": int(max(len(result.path) - 1, 0)),
                }
            )
    df = pd.DataFrame(rows)
    results_path = out_dir / "benchmark_results.csv"
    df.to_csv(results_path, index=False)

    summary = (
        df.groupby("algorithm", dropna=False)
        .agg(
            found_rate=("found", "mean"),
            avg_cost=("cost", "mean"),
            avg_expanded=("expanded_nodes", "mean"),
            avg_elapsed_ms=("elapsed_ms", "mean"),
        )
        .reset_index()
        .to_dict(orient="records")
    )
    summary_payload = {"graph_path": str(graph_path), "n_pairs": len(pairs), "summary": summary}
    summary_path = out_dir / "benchmark_summary.json"
    summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    return BenchmarkResult(results_path=results_path, summary_path=summary_path)
