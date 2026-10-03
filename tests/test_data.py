"""Integration-ish tests that exercise the real master parquet when it exists.

Skipped automatically on fresh machines where `make data` hasn't been run.
"""

from __future__ import annotations

import pytest

from value_wage.config import get_settings
from value_wage.data import load_master
from value_wage.features import build_feature_matrix


@pytest.fixture(scope="module")
def master_available() -> bool:
    return get_settings().paths.raw_master.exists()


def test_load_master_runs_or_skips(master_available: bool) -> None:
    if not master_available:
        pytest.skip("Raw master parquet not available; run `make data` first.")
    result = load_master()
    assert result.row_count > 0
    assert result.sha256
    assert len(result.season_counts) >= 1


def test_feature_matrix_has_allowlist(master_available: bool) -> None:
    if not master_available:
        pytest.skip("Raw master parquet not available; run `make data` first.")
    settings = get_settings()
    master = load_master(settings).frame
    df = build_feature_matrix(master, settings=settings)
    for col in settings.load_features().all_features():
        assert col in df.columns, f"Missing engineered feature: {col}"
    # Wage target stays on FM data; value target is now TM-sourced and only present after
    # `load_master_with_tm`, so we don't assert its presence on the raw-master path.
    assert "wage_pw" in df.columns
