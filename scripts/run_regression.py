from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from finder.config import load_config
from finder.variants import VARIANTS


def _run(cmd: list[str], cwd: Path = ROOT) -> str:
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{ROOT}{os.pathsep}{existing_pythonpath}" if existing_pythonpath else str(ROOT)
    try:
        proc = subprocess.run(
            cmd,
            check=True,
            text=True,
            capture_output=True,
            cwd=str(cwd),
            env=env,
        )
    except subprocess.CalledProcessError as exc:
        stdout = (exc.stdout or "").strip()
        stderr = (exc.stderr or "").strip()
        detail = textwrap.dedent(
            f"""
            Command failed: {' '.join(cmd)}
            Working directory: {cwd}
            Exit code: {exc.returncode}
            STDOUT:
            {stdout or '<empty>'}
            STDERR:
            {stderr or '<empty>'}
            """
        ).strip()
        raise RuntimeError(detail) from exc
    out = proc.stdout.strip()
    if proc.stderr.strip():
        print(proc.stderr.strip())
    return out


def _ensure_runtime_deps() -> None:
    missing = []
    for pkg in ("torch", "torch_geometric"):
        if importlib.util.find_spec(pkg) is None:
            missing.append(pkg)
    if missing:
        joined = ", ".join(missing)
        raise RuntimeError(
            f"Missing runtime dependencies: {joined}. Install with `pip install -r requirements.txt`."
        )


def _config_path(root: Path, variant: str, smoke: bool) -> Path:
    suffix = ".smoke.toml" if smoke else ".toml"
    return root / "configs" / f"{variant.lower()}{suffix}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one-command modern regression across all variants")
    parser.add_argument("--smoke", action="store_true", help="Use short smoke configs")
    parser.add_argument("--with-parity", action="store_true", help="Also run parity reports when synthetic data is available")
    parser.add_argument("--data-root", default="", help="Override synthetic data directory root for parity")
    parser.add_argument("--graphs", type=int, default=20, help="Number of graphs for parity comparisons")
    parser.add_argument("--out-dir", default="reports/regression", help="Directory for regression artifacts")
    args = parser.parse_args()
    _ensure_runtime_deps()

    root = ROOT
    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, dict[str, str | bool]] = {}

    for variant in VARIANTS:
        cfg_path = _config_path(root, variant, args.smoke)
        if not cfg_path.exists():
            print(f"skip {variant}: missing {cfg_path}")
            continue

        print(f"[{variant}] train")
        model_path = _run([sys.executable, "-m", "finder.cli", "train", "--config", str(cfg_path)])
        summary[variant] = {"model": model_path, "train_ok": True}

        if not args.with_parity:
            continue

        cfg = load_config(cfg_path)
        data_dir = cfg.synthetic_data_dir
        if args.data_root:
            data_dir = str(Path(args.data_root) / variant.lower())
        if not data_dir or not Path(data_dir).exists():
            summary[variant]["parity_ok"] = False
            summary[variant]["parity_reason"] = f"missing synthetic data dir: {data_dir}"
            continue

        report_out = out_dir / f"parity_{variant.lower()}.json"
        print(f"[{variant}] parity-report")
        _run(
            [
                sys.executable,
                "-m",
                "finder.cli",
                "parity-report",
                "--config",
                str(cfg_path),
                "--variant",
                variant,
                "--data",
                data_dir,
                "--modern-model",
                model_path,
                "--out",
                str(report_out),
                "--graphs",
                str(args.graphs),
                "--legacy-dir",
                cfg.legacy_variant_dir,
                "--legacy-model",
                cfg.legacy_model_ckpt,
            ]
        )
        summary[variant]["parity_ok"] = True
        summary[variant]["parity_report"] = str(report_out)

    summary_file = out_dir / "summary.json"
    summary_file.write_text(json.dumps(summary, indent=2))
    print(summary_file)


if __name__ == "__main__":
    main()
