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
