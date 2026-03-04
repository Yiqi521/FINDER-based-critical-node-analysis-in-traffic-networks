# FINDER (FInding key players in complex Networks through DEep Reinforcement learning)

This repository contains two paths:

- `legacy` path (TensorFlow v1 + Cython/C++) used to reproduce the original paper behavior.
- `modern` path (PyTorch + PyG + Python 3.10/3.11) for ongoing development and maintainability.

Paper:
Fan, C., Zeng, L., Sun, Y and Liu Y-Y. [Finding key players in complex networks through deep reinforcement learning](https://www.nature.com/articles/s42256-020-0177-2.epdf?sharing_token=0CAxnrCP1THxBEtK2mS5c9RgN0jAjWel9jnR3ZoTv0O3ej6g4eVo3V4pnngJO-QMH375GbplyUstNSGUaq-zMyAnpSrZIOiiDvB0V_CqsCipIfCq-enY3sK3Uv_D_4b4aRn6lYXd8HEinWjLNM42tQZ0iVjeMBl6ZRA7D7WUBjM%3D)

## Repository Layout

- `code/`: original research implementations for `FINDER_CN`, `FINDER_CN_cost`, `FINDER_ND`, `FINDER_ND_cost`
- `finder/`: modernized Python package with unified CLI and PyTorch/PyG implementation
- `configs/`: modern training/evaluation config files
- `scripts/`: migration and baseline utility scripts
- `environment/Dockerfile`: modern Python 3.11 container

## Modern Path (Recommended)

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

## Notes on modernization

- Python 2 shebangs were updated to Python 3.
- Deprecated `np.int` aliases were replaced.
- Deprecated `time.clock()` calls were replaced with `time.perf_counter()`.
- NetworkX compatibility fixes were applied for modern versions (`g.nodes[...]`, `set_node_attributes`).
- `distutils` setup scripts were moved to `setuptools` imports.

## Citation

```bibtex
@article{fan2020finding,
   title={Finding key players in complex networks through deep reinforcement learning},
   author={Fan, Changjun and Zeng, Li and Sun, Yizhou and Liu, Yang-Yu},
   journal={Nature Machine Intelligence},
   pages={1--8},
   year={2020},
   publisher={Nature Publishing Group}
 }
```
