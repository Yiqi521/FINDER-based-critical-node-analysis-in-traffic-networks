from finder.config import TrainConfig


def test_default_config_values() -> None:
    cfg = TrainConfig()
    assert cfg.batch_size > 0
    assert cfg.num_min <= cfg.num_max
    assert isinstance(cfg.legacy_variant_dir, str)
    assert isinstance(cfg.legacy_model_ckpt, str)
