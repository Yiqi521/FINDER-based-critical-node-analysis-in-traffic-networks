from finder.variants import VARIANTS, normalize_variant


def test_normalize_variant_case_insensitive() -> None:
    assert normalize_variant("cn") == "CN"
    assert normalize_variant("Nd_Cost") == "ND_cost"


def test_variants_have_legacy_defaults() -> None:
    for name, v in VARIANTS.items():
        assert v.name == name
        assert v.legacy_dir
        assert v.legacy_checkpoint
