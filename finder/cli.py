from __future__ import annotations

import argparse
import json

from finder.config import TrainConfig, load_config


def _load_cfg(path: str | None) -> TrainConfig:
    if path:
        return load_config(path)
    return TrainConfig()


def main() -> None:
    parser = argparse.ArgumentParser(prog="finder")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train", help="Train modern PyTorch FINDER")
    p_train.add_argument("--config", type=str, default="configs/default.toml")

    p_eval_s = sub.add_parser("eval-synth", help="Evaluate synthetic data")
    p_eval_s.add_argument("--config", type=str, default="configs/default.toml")
    p_eval_s.add_argument("--data", type=str, required=True)
    p_eval_s.add_argument("--model", type=str, required=True)

    p_eval_r = sub.add_parser("eval-real", help="Evaluate real graph data")
    p_eval_r.add_argument("--config", type=str, default="configs/default.toml")
    p_eval_r.add_argument("--data", type=str, required=True)
    p_eval_r.add_argument("--model", type=str, required=True)

    p_parity = sub.add_parser("parity-report", help="Generate legacy-vs-modern parity report")
    p_parity.add_argument("--config", type=str, default="configs/default.toml")
    p_parity.add_argument("--variant", type=str, required=True)
    p_parity.add_argument("--data", type=str, required=True)
    p_parity.add_argument("--modern-model", type=str, required=True)
    p_parity.add_argument("--out", type=str, default="reports/parity_report.json")
    p_parity.add_argument("--graphs", type=int, default=100)
    p_parity.add_argument("--legacy-dir", type=str, default="")
    p_parity.add_argument("--legacy-model", type=str, default="")

    args = parser.parse_args()
    cfg = _load_cfg(args.config)

    if args.cmd == "train":
        from finder.trainer import train

        out = train(cfg)
        print(str(out))
    elif args.cmd == "eval-synth":
        from finder.evaluate import evaluate_synthetic

        result = evaluate_synthetic(args.data, args.model, cfg)
        print(json.dumps(result, indent=2))
    elif args.cmd == "eval-real":
        from finder.evaluate import evaluate_real

        result = evaluate_real(args.data, args.model, cfg)
        print(json.dumps(result, indent=2))
    elif args.cmd == "parity-report":
        from finder.parity import generate_parity_report

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
        print(str(out))


if __name__ == "__main__":
    main()
