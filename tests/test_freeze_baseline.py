from pathlib import Path


def test_checkpoint_manifest_exists() -> None:
    path = Path("baseline/checkpoint_manifest.json")
    if path.exists():
        assert path.read_text().strip().startswith("[")
