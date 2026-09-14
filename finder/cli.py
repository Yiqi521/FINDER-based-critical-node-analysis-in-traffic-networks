from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

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

    p_pre = sub.add_parser("traffic-preprocess", help="Preprocess map into computable Tokyo subgraph")
    p_pre.add_argument("--input", type=str, required=True)
    p_pre.add_argument("--output-dir", type=str, default="runs")
    p_pre.add_argument("--bbox", type=str, default="")
    p_pre.add_argument("--max-nodes", type=int, default=0)
    p_pre.add_argument("--max-edges", type=int, default=0)

    p_score = sub.add_parser("traffic-score-nodes", help="Score critical nodes with FINDER model")
    p_score.add_argument("--graph", type=str, required=True)
    p_score.add_argument("--model", type=str, required=True)
    p_score.add_argument("--config", type=str, default="configs/default.toml")
    p_score.add_argument("--output-dir", type=str, default="runs")

    p_bench = sub.add_parser("traffic-benchmark-paths", help="Benchmark path planners on OD pairs")
    p_bench.add_argument("--graph", type=str, required=True)
    p_bench.add_argument("--pairs", type=int, default=100)
    p_bench.add_argument("--output-dir", type=str, default="runs")

    p_report = sub.add_parser("traffic-report", help="Build relationship report from benchmarks + node scores")
    p_report.add_argument("--benchmark-csv", type=str, required=True)
    p_report.add_argument("--nodes-csv", type=str, required=True)
    p_report.add_argument("--output-dir", type=str, default="runs")

    args = parser.parse_args()
    cfg = _load_cfg(getattr(args, "config", None))

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
    elif args.cmd == "traffic-preprocess":
        from experiments.traffic.preprocess import preprocess_map

        run_dir = Path(args.output_dir) / datetime.now().strftime("%Y%m%d_%H%M%S_preprocess")
        result = preprocess_map(
            input_path=args.input,
            output_dir=run_dir,
            bbox=args.bbox or None,
            max_nodes=args.max_nodes if args.max_nodes > 0 else None,
            max_edges=args.max_edges if args.max_edges > 0 else None,
        )
        print(json.dumps({"graph": str(result.graph_path), "stats": str(result.stats_path)}, indent=2))
    elif args.cmd == "traffic-score-nodes":
        from experiments.traffic.critical_nodes import perturbation_report, score_critical_nodes
        import networkx as nx
        import pandas as pd

        run_dir = Path(args.output_dir) / datetime.now().strftime("%Y%m%d_%H%M%S_score")
        result = score_critical_nodes(args.graph, run_dir, model_path=args.model, cfg=cfg)
        g = nx.convert_node_labels_to_integers(nx.read_graphml(args.graph))
        ranks = pd.read_csv(result.table_path).sort_values("finder_rank")["node"].tolist()
        perturb = perturbation_report(g, [int(n) for n in ranks])
        perturb_path = run_dir / "perturbation.csv"
        perturb.to_csv(perturb_path, index=False)
        print(
            json.dumps(
                {
                    "critical_nodes": str(result.table_path),
                    "metadata": str(result.metadata_path),
                    "perturbation": str(perturb_path),
                },
                indent=2,
            )
        )
    elif args.cmd == "traffic-benchmark-paths":
        from experiments.traffic.benchmark import run_benchmark

        run_dir = Path(args.output_dir) / datetime.now().strftime("%Y%m%d_%H%M%S_benchmark")
        result = run_benchmark(args.graph, run_dir, n_pairs=args.pairs)
        print(json.dumps({"results": str(result.results_path), "summary": str(result.summary_path)}, indent=2))
    elif args.cmd == "traffic-report":
        from experiments.traffic.analysis import build_relationship_report

        run_dir = Path(args.output_dir) / datetime.now().strftime("%Y%m%d_%H%M%S_report")
        result = build_relationship_report(args.benchmark_csv, args.nodes_csv, run_dir)
        print(json.dumps({"merged": str(result.merged_path), "report": str(result.report_path)}, indent=2))


if __name__ == "__main__":
    main()
