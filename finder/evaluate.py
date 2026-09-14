from __future__ import annotations

import time
from pathlib import Path

import networkx as nx
import numpy as np
import torch

from finder.config import TrainConfig
from finder.graph_env import FinderEnv
from finder.model import FinderModel


def _load_model(model_path: str | Path, device: torch.device) -> FinderModel:
    ckpt = torch.load(model_path, map_location=device)
    cfg = ckpt.get("config", {})
    hidden_size = int(cfg.get("hidden_size", 64))
    model = FinderModel(hidden_channels=hidden_size).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def _to_tensors(g: nx.Graph, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    n = g.number_of_nodes()
    x = torch.ones((n, 2), dtype=torch.float32, device=device)
    edges = list(g.edges())
    if not edges:
        edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
    else:
        undirected = edges + [(b, a) for (a, b) in edges]
        edge_index = torch.tensor(undirected, dtype=torch.long, device=device).t().contiguous()
    return x, edge_index


def evaluate_graph(
    g: nx.Graph,
    model: FinderModel,
    device: torch.device,
    task: str = "CN",
) -> tuple[float, float]:
    env = FinderEnv(g, task=task)  # type: ignore[arg-type]
    env.reset()
    x, edge_index = _to_tensors(g, device)
    t1 = time.perf_counter()
    while not env.is_terminal():
        actions = env.available_actions()
        if not actions:
            break
        cands = torch.tensor(actions, dtype=torch.long, device=device)
        aux = torch.zeros((len(actions), 4), dtype=torch.float32, device=device)
        with torch.no_grad():
            qvals = model(x, edge_index, cands, aux)
        action = actions[int(torch.argmax(qvals).item())]
        env.step(action)
    t2 = time.perf_counter()
    # Report the negative score so higher is better for all variants
    if task in ("CN", "CN_cost"):
        score = -env.remaining_cnd_score()
    else:
        score = -env._max_cc_size()
    return float(score), float(t2 - t1)


def evaluate_synthetic(data_dir: str, model_path: str, cfg: TrainConfig) -> dict[str, float]:
    device = torch.device(cfg.device)
    model = _load_model(model_path, device)
    scores, times = [], []
    for i in range(100):
        g = nx.convert_node_labels_to_integers(nx.read_gml(Path(data_dir) / f"g_{i}"))
        s, t = evaluate_graph(g, model, device, task=cfg.variant)
        scores.append(s)
        times.append(t)
    return {
        "score_mean": float(np.mean(scores)),
        "score_std": float(np.std(scores)),
        "time_mean": float(np.mean(times)),
        "time_std": float(np.std(times)),
    }


def evaluate_real(data_path: str, model_path: str, cfg: TrainConfig) -> dict[str, float]:
    device = torch.device(cfg.device)
    model = _load_model(model_path, device)
    g = nx.convert_node_labels_to_integers(nx.read_edgelist(data_path))
    score, elapsed = evaluate_graph(g, model, device, task=cfg.variant)
    return {"score": score, "time": elapsed}
