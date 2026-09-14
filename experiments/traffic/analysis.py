from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(slots=True)
class AnalysisResult:
    merged_path: Path
    report_path: Path


def build_relationship_report(
    benchmark_csv: str | Path,
    critical_nodes_csv: str | Path,
    output_dir: str | Path,
) -> AnalysisResult:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bench = pd.read_csv(benchmark_csv)
    nodes = pd.read_csv(critical_nodes_csv)
    node_rank = nodes[["node", "finder_rank", "finder_score", "betweenness", "degree"]].copy()

    src = node_rank.rename(
        columns={
            "node": "source",
            "finder_rank": "source_finder_rank",
            "finder_score": "source_finder_score",
            "betweenness": "source_betweenness",
            "degree": "source_degree",
        }
    )
    tgt = node_rank.rename(
        columns={
            "node": "target",
            "finder_rank": "target_finder_rank",
            "finder_score": "target_finder_score",
            "betweenness": "target_betweenness",
            "degree": "target_degree",
        }
    )
    merged = bench.merge(src, on="source", how="left").merge(tgt, on="target", how="left")
    merged["endpoint_criticality"] = (
        merged["source_finder_score"].fillna(0.0) + merged["target_finder_score"].fillna(0.0)
    )
    merged_path = out_dir / "relationship_merged.csv"
    merged.to_csv(merged_path, index=False)

    report: dict[str, object] = {"rows": int(len(merged)), "algorithms": {}}
    for algo, chunk in merged.groupby("algorithm"):
        corr = chunk[["elapsed_ms", "expanded_nodes", "endpoint_criticality"]].corr(numeric_only=True)
        report["algorithms"][str(algo)] = {
            "count": int(len(chunk)),
            "elapsed_vs_endpoint_criticality_corr": float(corr.loc["elapsed_ms", "endpoint_criticality"])
            if "elapsed_ms" in corr.index and "endpoint_criticality" in corr.columns
            else 0.0,
            "expanded_vs_endpoint_criticality_corr": float(corr.loc["expanded_nodes", "endpoint_criticality"])
            if "expanded_nodes" in corr.index and "endpoint_criticality" in corr.columns
            else 0.0,
            "found_rate": float(chunk["found"].mean()),
            "avg_elapsed_ms": float(chunk["elapsed_ms"].mean()),
            "avg_expanded_nodes": float(chunk["expanded_nodes"].mean()),
        }

    report_path = out_dir / "relationship_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return AnalysisResult(merged_path=merged_path, report_path=report_path)
