"""Belt-and-braces leakage test.

Confirms that no column marked `excluded` in config/features.yaml appears in the model's
numeric or categorical feature list. If this test fails, the FM `value_high` (or sibling
money fields) are about to leak into the Transfermarkt-targeted value model.
"""

from __future__ import annotations

from value_wage.config import ALL_TARGETS, get_settings
from value_wage.features import feature_lists_for


def test_no_excluded_column_is_a_feature() -> None:
    settings = get_settings()
    feats = settings.load_features()
    excluded = set(feats.excluded)
    for target in ALL_TARGETS:
        numeric, categorical = feature_lists_for(target, feats)
        both = set(numeric + categorical)
        intersect = both & excluded
        assert not intersect, (
            f"Target '{target}': features {sorted(intersect)} are in the model's "
            "feature list AND the excluded list. Fix config/features.yaml."
        )


def test_critical_fm_money_fields_are_excluded() -> None:
    """Even if someone reorders features.yaml, these specific fields MUST stay excluded."""
    settings = get_settings()
    excluded = set(settings.load_features().excluded)
    for col in ("value_high", "sell_value", "release_clause"):
        assert col in excluded, f"FM money field `{col}` must be in `excluded` to prevent leakage."
