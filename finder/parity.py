from __future__ import annotations

import importlib
import json
import sys
import time
from pathlib import Path
from statistics import mean, pstdev

from finder.config import TrainConfig
from finder.variants import VARIANTS, normalize_variant


def _import_legacy_finder(legacy_dir: Path):
    legacy_dir = legacy_dir.resolve()
    sys.path.insert(0, str(legacy_dir))
    try:
        if "FINDER" in sys.modules:
            del sys.modules["FINDER"]
        module = importlib.import_module("FINDER")
    finally:
        try:
            sys.path.remove(str(legacy_dir))
        except ValueError:
            pass
    if not hasattr(module, "FINDER"):
        raise RuntimeError(f"Legacy FINDER class not found from {legacy_dir}")
    return module.FINDER


def _resolve_legacy_paths(variant: str, legacy_dir: str | None, legacy_model: str | None) -> tuple[Path, Path]:
    info = VARIANTS[variant]
    resolved_dir = Path(legacy_dir) if legacy_dir else Path(info.legacy_dir)
    resolved_model = Path(legacy_model) if legacy_model else Path(info.legacy_checkpoint)
    return resolved_dir, resolved_model


def generate_parity_report(
    cfg: TrainConfig,
    variant: str,
    data_dir: str,
    modern_model_path: str,
    out_path: str,
    n_graphs: int = 100,
    legacy_dir: str | None = None,
    legacy_model_path: str | None = None,
) -> Path:
    import networkx as nx
    import torch

    from finder.evaluate import _load_model, evaluate_graph

    variant = normalize_variant(variant)
    data_root = Path(data_dir)
    if not data_root.exists():
        raise FileNotFoundError(f"Synthetic dataset directory not found: {data_root}")

    legacy_dir_path, legacy_model = _resolve_legacy_paths(variant, legacy_dir, legacy_model_path)
    if not legacy_dir_path.exists():
        raise FileNotFoundError(f"Legacy variant directory not found: {legacy_dir_path}")
    if not legacy_model.exists():
        raise FileNotFoundError(f"Legacy checkpoint not found: {legacy_model}")

    device = torch.device(cfg.device)
    modern_model = _load_model(modern_model_path, device)

    legacy_cls = _import_legacy_finder(legacy_dir_path)
    legacy = legacy_cls()
    legacy.LoadModel(str(legacy_model))

    modern_scores: list[float] = []
    modern_times: list[float] = []
    legacy_scores: list[float] = []
    legacy_times: list[float] = []

    for i in range(n_graphs):
        g_path = data_root / f"g_{i}"
        if not g_path.exists():
            break
        g = nx.read_gml(g_path)

        m_score, m_time = evaluate_graph(g, modern_model, device)
        modern_scores.append(float(m_score))
        modern_times.append(float(m_time))

        legacy.InsertGraph(g, is_test=True)
        t1 = time.perf_counter()
        l_score, _sol = legacy.GetSol(i)
        t2 = time.perf_counter()
        legacy_scores.append(float(l_score))
        legacy_times.append(float(t2 - t1))

    if hasattr(legacy, "ClearTestGraphs"):
        legacy.ClearTestGraphs()

    if not legacy_scores or not modern_scores:
        raise RuntimeError("No comparable graph samples were evaluated; check dataset path and graph files")

    modern_mean = float(mean(modern_scores))
    legacy_mean = float(mean(legacy_scores))
    abs_delta = abs(modern_mean - legacy_mean)
    rel_delta = abs_delta / max(1e-8, abs(legacy_mean))

    report = {
        "variant": variant,
        "num_graphs": min(len(modern_scores), len(legacy_scores)),
        "modern": {
            "score_mean": modern_mean,
            "score_std": float(pstdev(modern_scores) if len(modern_scores) > 1 else 0.0),
            "time_mean": float(mean(modern_times)),
            "time_std": float(pstdev(modern_times) if len(modern_times) > 1 else 0.0),
        },
        "legacy": {
            "score_mean": legacy_mean,
            "score_std": float(pstdev(legacy_scores) if len(legacy_scores) > 1 else 0.0),
            "time_mean": float(mean(legacy_times)),
            "time_std": float(pstdev(legacy_times) if len(legacy_times) > 1 else 0.0),
        },
        "drift": {
            "abs_score_mean_delta": float(abs_delta),
            "relative_score_mean_delta": float(rel_delta),
        },
        "inputs": {
            "data_dir": str(data_root),
            "modern_model": str(modern_model_path),
            "legacy_model": str(legacy_model),
            "legacy_dir": str(legacy_dir_path),
        },
    }

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    return out
