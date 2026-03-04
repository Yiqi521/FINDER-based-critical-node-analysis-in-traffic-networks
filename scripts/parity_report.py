from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from finder.config import load_config
from finder.parity import generate_parity_report


def _ensure_runtime_deps() -> None:
    missing = []
    for pkg in ("networkx", "torch", "torch_geometric"):
        if importlib.util.find_spec(pkg) is None:
            missing.append(pkg)
    if missing:
        joined = ", ".join(missing)
        raise RuntimeError(
            f"Missing runtime dependencies: {joined}. Install with `pip install -r requirements.txt`."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate modern-vs-legacy parity report")
    parser.add_argument("--config", default="configs/default.toml")
    parser.add_argument("--variant", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--modern-model", required=True)
    parser.add_argument("--out", default="reports/parity_report.json")
    parser.add_argument("--graphs", type=int, default=100)
    parser.add_argument("--legacy-dir", default="")
    parser.add_argument("--legacy-model", default="")
    args = parser.parse_args()
    _ensure_runtime_deps()

    cfg = load_config(args.config)
    out = generate_parity_report(
        cfg=cfg,
        variant=args.variant,
        data_dir=args.data,
        modern_model_path=args.modern_model,
        out_path=args.out,
        n_graphs=args.graphs,
        legacy_dir=args.legacy_dir or None,
        legacy_model_path=args.legacy_model or None,
    )
    print(out)


if __name__ == "__main__":
    main()
