from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import pandas as pd
import torch

from finder.config import TrainConfig
from finder.evaluate import _load_model, _to_tensors
from finder.graph_env import FinderEnv


@dataclass(slots=True)
class CriticalNodeResult:
    table_path: Path
    metadata_path: Path


def _finder_order_scores(
    g: nx.Graph,
    model: torch.nn.Module,
    device: torch.device,
    task: str = "CN",
) -> tuple[dict[int, int], dict[int, float]]:
    env = FinderEnv(g, task=task)  # type: ignore[arg-type]
    env.reset()
    x, edge_index = _to_tensors(g, device)
    rank: dict[int, int] = {}
    score: dict[int, float] = {}
    step = 0
    while not env.is_terminal():
        actions = env.available_actions()
        if not actions:
            break
        cands = torch.tensor(actions, dtype=torch.long, device=device)
        aux = torch.zeros((len(actions), 4), dtype=torch.float32, device=device)
        with torch.no_grad():
            qvals = model(x, edge_index, cands, aux)
        idx = int(torch.argmax(qvals).item())
        action = actions[idx]
        rank[action] = step
        score[action] = float(qvals[idx].item())
        env.step(action)
        step += 1
    return rank, score


def _compute_node_features(g: nx.Graph) -> pd.DataFrame:
    deg = dict(g.degree())
    wdeg = dict(g.degree(weight="travel_time"))
    btw = nx.betweenness_centrality(g, normalized=True)
    clo = nx.closeness_centrality(g)
    clust = nx.clustering(g)
    articulation = set(nx.articulation_points(g)) if g.number_of_nodes() > 0 else set()

    rows = []
    for n, data in g.nodes(data=True):
        rows.append(
            {
                "node": int(n),
                "degree": float(deg.get(n, 0)),
                "weighted_degree": float(wdeg.get(n, 0.0)),
                "betweenness": float(btw.get(n, 0.0)),
                "closeness": float(clo.get(n, 0.0)),
                "clustering": float(clust.get(n, 0.0)),
                "is_articulation": int(n in articulation),
                "lat": float(data.get("lat", 0.0)),
                "lon": float(data.get("lon", 0.0)),
            }
        )
    return pd.DataFrame(rows)


def score_critical_nodes(
    graph_path: str | Path,
    output_dir: str | Path,
    *,
    model_path: str | Path,
    cfg: TrainConfig,
) -> CriticalNodeResult:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    g = nx.convert_node_labels_to_integers(nx.read_graphml(graph_path))
    device = torch.device(cfg.device)
    model = _load_model(model_path, device)
    rank_map, score_map = _finder_order_scores(g, model, device, task=cfg.variant)
    df = _compute_node_features(g)
    df["finder_rank"] = df["node"].map(rank_map).fillna(10**9).astype(int)
    df["finder_score"] = df["node"].map(score_map).fillna(0.0)
    df = df.sort_values(["finder_rank", "node"], ascending=[True, True]).reset_index(drop=True)
    table_path = out_dir / "critical_nodes.csv"
    df.to_csv(table_path, index=False)

    meta = {
        "graph_path": str(graph_path),
        "model_path": str(model_path),
        "nodes": int(g.number_of_nodes()),
        "edges": int(g.number_of_edges()),
    }
    metadata_path = out_dir / "critical_nodes_meta.json"
    metadata_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return CriticalNodeResult(table_path=table_path, metadata_path=metadata_path)


def perturbation_report(
    graph: nx.Graph,
    ranked_nodes: list[int],
    *,
    k_values: list[int] | None = None,
    random_seed: int = 42,
) -> pd.DataFrame:
    if k_values is None:
        k_values = [5, 10, 20]
    rng = random.Random(random_seed)
    all_nodes = list(graph.nodes())
    rows: list[dict[str, float | int | str]] = []
    baseline_lcc = len(max(nx.connected_components(graph), key=len)) if graph.number_of_nodes() else 0

    for k in k_values:
        top_k = ranked_nodes[: min(k, len(ranked_nodes))]
        random_k = rng.sample(all_nodes, k=min(k, len(all_nodes))) if all_nodes else []
        for policy, removed in (("top_finder", top_k), ("random", random_k)):
            g2 = graph.copy()
            g2.remove_nodes_from(removed)
            lcc = len(max(nx.connected_components(g2), key=len)) if g2.number_of_nodes() else 0
            rows.append(
                {
                    "policy": policy,
                    "k": int(k),
                    "remaining_nodes": int(g2.number_of_nodes()),
                    "remaining_edges": int(g2.number_of_edges()),
                    "largest_component": int(lcc),
                    "largest_component_ratio": float(lcc / baseline_lcc) if baseline_lcc else 0.0,
                }
            )
    return pd.DataFrame(rows)
