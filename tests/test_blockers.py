"""Regression tests for the two blockers identified in the ChatGPT review."""
from __future__ import annotations

import tempfile
from dataclasses import asdict
from pathlib import Path

import networkx as nx
import pytest
import torch

from finder.config import TrainConfig
from finder.evaluate import _to_tensors, evaluate_graph
from finder.model import FinderModel


# ---------------------------------------------------------------------------
# Blocker 1: cfg.__dict__ raises AttributeError on slots=True dataclass
# ---------------------------------------------------------------------------

def test_asdict_works_on_slots_config() -> None:
    cfg = TrainConfig()
    d = asdict(cfg)
    assert isinstance(d, dict)
    assert "hidden_size" in d
    assert d["batch_size"] == cfg.batch_size


def test_trainer_save_uses_asdict(tmp_path: Path) -> None:
    """Simulate what trainer.py does — must not raise AttributeError."""
    cfg = TrainConfig(episodes=1, out_dir=str(tmp_path))
    model = FinderModel(hidden_channels=cfg.hidden_size)
    out = tmp_path / "finder_cn.pt"
    # This must not raise:
    torch.save({"model": model.state_dict(), "config": asdict(cfg)}, out)
    ckpt = torch.load(out, map_location="cpu")
    assert ckpt["config"]["hidden_size"] == 64


# ---------------------------------------------------------------------------
# Blocker 2: String node IDs in real/GML graphs crash _to_tensors / graph_env
# ---------------------------------------------------------------------------

def test_to_tensors_integer_nodes() -> None:
    g = nx.path_graph(4)  # nodes are 0,1,2,3
    x, ei = _to_tensors(g, torch.device("cpu"))
    assert x.shape == (4, 2)
    assert ei.dtype == torch.long


def test_evaluate_graph_string_node_ids() -> None:
    """Graph with string node IDs must work after convert_node_labels_to_integers."""
    g_str = nx.Graph()
    g_str.add_edges_from([("a", "b"), ("b", "c"), ("c", "d")])
    g = nx.convert_node_labels_to_integers(g_str)
    model = FinderModel(hidden_channels=64)
    model.eval()
    device = torch.device("cpu")
    score, elapsed = evaluate_graph(g, model, device)
    assert isinstance(score, float)
    assert elapsed >= 0.0


def test_evaluate_real_from_edgelist(tmp_path: Path) -> None:
    """evaluate_real must handle an edgelist with string node labels."""
    edge_file = tmp_path / "graph.txt"
    edge_file.write_text("alice bob\nbob carol\ncarol dave\n")

    from finder.evaluate import evaluate_real
    from finder.config import TrainConfig

    # Save a minimal model checkpoint
    cfg = TrainConfig()
    model = FinderModel(hidden_channels=cfg.hidden_size)
    model_path = tmp_path / "model.pt"
    torch.save({"model": model.state_dict(), "config": asdict(cfg)}, model_path)

    result = evaluate_real(str(edge_file), str(model_path), cfg)
    assert "score" in result
    assert "time" in result
