from __future__ import annotations

from dataclasses import asdict

import networkx as nx
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from experiments.traffic.analysis import build_relationship_report
from experiments.traffic.benchmark import run_benchmark
from experiments.traffic.critical_nodes import perturbation_report, score_critical_nodes
from finder.config import TrainConfig
from finder.model import FinderModel


def _toy_graph(path) -> None:
    g = nx.path_graph(6)
    for n in g.nodes:
        g.nodes[n]["lat"] = 35.0 + 0.001 * n
        g.nodes[n]["lon"] = 139.0 + 0.001 * n
    for u, v in g.edges:
        g[u][v]["travel_time"] = 1.0
    nx.write_graphml(g, path)


def test_score_nodes_and_perturbation(tmp_path) -> None:
    graph_path = tmp_path / "toy.graphml"
    _toy_graph(graph_path)
    cfg = TrainConfig()
    model = FinderModel(hidden_channels=cfg.hidden_size)
    model_path = tmp_path / "model.pt"
    torch.save({"model": model.state_dict(), "config": asdict(cfg)}, model_path)

    out_dir = tmp_path / "score"
    result = score_critical_nodes(graph_path, out_dir, model_path=model_path, cfg=cfg)
    df = pd.read_csv(result.table_path)
    assert "finder_rank" in df.columns
    assert "finder_score" in df.columns
    g = nx.convert_node_labels_to_integers(nx.read_graphml(graph_path))
    ranks = df.sort_values("finder_rank")["node"].astype(int).tolist()
    pert = perturbation_report(g, ranks, k_values=[2])
    assert len(pert) == 2


def test_benchmark_and_relationship_report(tmp_path) -> None:
    graph_path = tmp_path / "toy.graphml"
    _toy_graph(graph_path)
    bench = run_benchmark(graph_path, tmp_path / "bench", n_pairs=5, seed=1)

    nodes = pd.DataFrame(
        [
            {"node": i, "finder_rank": i, "finder_score": float(i), "betweenness": 0.1, "degree": 2.0}
            for i in range(6)
        ]
    )
    nodes_path = tmp_path / "nodes.csv"
    nodes.to_csv(nodes_path, index=False)
    report = build_relationship_report(bench.results_path, nodes_path, tmp_path / "report")
    assert report.merged_path.exists()
    assert report.report_path.exists()
