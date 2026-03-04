from __future__ import annotations

import subprocess
from pathlib import Path

VARIANTS = ["FINDER_CN", "FINDER_CN_cost", "FINDER_ND", "FINDER_ND_cost"]


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    for v in VARIANTS:
        d = root / "code" / v
        print(f"building {v}...")
        subprocess.run(["python", "setup.py", "build_ext", "-i"], cwd=d, check=True)


if __name__ == "__main__":
    main()
