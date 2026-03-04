from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib


@dataclass(slots=True)
class TrainConfig:
    variant: str = "CN"
    embedding_size: int = 64
    hidden_size: int = 64
    learning_rate: float = 1e-4
    gamma: float = 1.0
    n_step: int = 5
    memory_size: int = 50000
    batch_size: int = 64
    episodes: int = 2000
    target_update: int = 500
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 10000
    num_min: int = 30
    num_max: int = 50
    graph_type: str = "barabasi_albert"
    seed: int = 42
    device: str = "cpu"
    out_dir: str = "models_torch"
    synthetic_data_dir: str = ""
    real_data_path: str = ""
    legacy_variant_dir: str = ""
    legacy_model_ckpt: str = ""


def _merge_dataclass_defaults(raw: dict[str, Any]) -> TrainConfig:
    cfg = TrainConfig()
    for k, v in raw.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    return cfg


def load_config(path: str | Path) -> TrainConfig:
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    train = raw.get("train", {})
    return _merge_dataclass_defaults(train)
