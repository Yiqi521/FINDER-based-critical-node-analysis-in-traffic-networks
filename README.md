# FINDER (FInding key players in complex Networks through Deep Reinforcement learning)

### Requirements

- Python `>=3.10` (tested target: 3.11)
- `pip` and C/C++ build toolchain

Install:

```bash
pip install -r requirements.txt
pip install -e .
```

### Unified CLI

Train:

```bash
finder train --config configs/default.toml
```

Evaluate synthetic dataset directory (expects `g_0` ... `g_99`):

```bash
finder eval-synth --config configs/default.toml --data <synthetic_dir> --model <model.pt>
```

Evaluate real graph edgelist:

```bash
finder eval-real --config configs/default.toml --data <graph.txt> --model <model.pt>
```

Generate parity report (modern vs legacy) on synthetic graphs:

```bash
finder parity-report \
  --config configs/cn.toml \
  --variant CN \
  --data <synthetic_dir> \
  --modern-model <model.pt> \
  --out reports/parity_cn.json
```

Run one-command regression over all variants:

```bash
python scripts/run_regression.py --smoke
```

Or run parity directly as a script:

```bash
python scripts/parity_report.py \
  --config configs/cn.toml \
  --variant CN \
  --data <synthetic_dir> \
  --modern-model <model.pt>
```

## Tokyo Critical-Node Path Study Workflow

The repository includes a traffic experiment pipeline that reuses FINDER scoring and
adds shortest-path benchmarking and relationship analysis.

1) Preprocess a large map into a computable Tokyo-area subgraph.

Prefer the **`.osm.pbf`** extract (~440 MB here) over the **`.osm.xml`** (~9 GB): pass a
**bounding box** so the pipeline runs `osmium extract` first and only then builds the graph.
Install [osmium-tool](https://osmcode.org/osmium-tool/) and ensure `osmium` is on your `PATH`,
or set `OSMIUM` to the full path of the `osmium` executable.

```bash
finder traffic-preprocess \
  --input kanto-260404.osm.pbf \
  --output-dir runs \
  --bbox 35.50,139.40,35.90,139.95 \
  --max-nodes 50000
```

If you already have a small `.osm.xml`, you can pass that as `--input` instead; **`.osm.pbf`
requires `--bbox`** (clipping is mandatory so we never load the whole planet-scale dump in Python).

2) Score critical nodes with a FINDER model and generate perturbation output:

```bash
finder traffic-score-nodes \
  --config configs/default.toml \
  --graph runs/<timestamp>_preprocess/tokyo_subgraph.graphml \
  --model models_torch/finder_cn.pt \
  --output-dir runs
```

3) Benchmark path planners (Dijkstra, A*, bidirectional Dijkstra):

```bash
finder traffic-benchmark-paths \
  --graph runs/<timestamp>_preprocess/tokyo_subgraph.graphml \
  --pairs 200 \
  --output-dir runs
```

4) Build relationship report between critical-node signals and planner behavior:

```bash
finder traffic-report \
  --benchmark-csv runs/<timestamp>_benchmark/benchmark_results.csv \
  --nodes-csv runs/<timestamp>_score/critical_nodes.csv \
  --output-dir runs
```

Main outputs per stage include `stats.json`, `critical_nodes.csv`,
`perturbation.csv`, `benchmark_results.csv`, and `relationship_report.json`.

## Legacy Path (Reference / Repro)

Legacy dependencies are preserved in `requirements-legacy.txt`.

```bash
pip install -r requirements-legacy.txt
```

Build legacy extensions inside each variant directory, for example:

```bash
cd code/FINDER_CN
python setup.py build_ext -i
python train.py
```

## Baseline Freeze Utility

To record hashes and sizes of existing TF checkpoints:

```bash
python scripts/freeze_baseline.py
```

Output:

- `baseline/checkpoint_manifest.json`
