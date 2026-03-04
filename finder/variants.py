from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class VariantDefaults:
    name: str
    legacy_dir: str
    legacy_checkpoint: str


VARIANTS: dict[str, VariantDefaults] = {
    "CN": VariantDefaults(
        name="CN",
        legacy_dir="code/FINDER_CN",
        legacy_checkpoint="code/FINDER_CN/models/nrange_30_50_iter_93300.ckpt",
    ),
    "ND": VariantDefaults(
        name="ND",
        legacy_dir="code/FINDER_ND",
        legacy_checkpoint="code/FINDER_ND/models/nrange_30_50_iter_78000.ckpt",
    ),
    "CN_cost": VariantDefaults(
        name="CN_cost",
        legacy_dir="code/FINDER_CN_cost",
        legacy_checkpoint="code/FINDER_CN_cost/models/nrange_30_50_iter_122100.ckpt",
    ),
    "ND_cost": VariantDefaults(
        name="ND_cost",
        legacy_dir="code/FINDER_ND_cost",
        legacy_checkpoint="code/FINDER_ND_cost/models/nrange_30_50_iter_134100.ckpt",
    ),
}


def normalize_variant(name: str) -> str:
    for key in VARIANTS:
        if key.lower() == name.lower():
            return key
    raise KeyError(f"Unsupported variant: {name}. Available: {', '.join(VARIANTS.keys())}")
