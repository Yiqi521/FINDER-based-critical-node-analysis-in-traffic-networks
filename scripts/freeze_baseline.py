from __future__ import annotations

import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    models = sorted(root.glob("code/**/models/*.ckpt.*"))
    report = []
    for m in models:
        report.append(
            {
                "path": str(m.relative_to(root)),
                "size": m.stat().st_size,
                "sha256": sha256(m),
            }
        )
    out_dir = root / "baseline"
    out_dir.mkdir(exist_ok=True)
    out = out_dir / "checkpoint_manifest.json"
    out.write_text(json.dumps(report, indent=2))
    print(out)


if __name__ == "__main__":
    main()
