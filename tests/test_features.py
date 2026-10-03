"""Unit + property tests for the pure feature-engineering primitives."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from value_wage.features import (
    encode_foot,
    months_to_contract_end,
    parse_primary_position,
    per_90,
)


class TestParsePrimaryPosition:
    def test_single_code(self) -> None:
        out = parse_primary_position(pd.Series(["GK", "DC", "ST"]))
        assert out.tolist() == ["GK", "CB", "ST"]

    def test_multi_code_priority(self) -> None:
        # GK beats everything; CB beats FB; W beats AM? No — AMC maps to AM, AML/AMR map to W.
        out = parse_primary_position(pd.Series(["GK,ST", "DC,DM,MC", "AMR,MR,ST"]))
        assert out.tolist() == ["GK", "CB", "W"]

    def test_null_and_empty(self) -> None:
        out = parse_primary_position(pd.Series([None, "", "   ", np.nan], dtype=object))
        assert out.isna().all()

    def test_unknown_code(self) -> None:
        out = parse_primary_position(pd.Series(["XYZ"]))
        assert out.isna().all()


class TestPer90:
    def test_simple_rate(self) -> None:
        rates = per_90(pd.Series([1.0, 2.0]), pd.Series([900, 1800]))
        assert rates.tolist() == [0.1, 0.1]

    def test_small_minutes_become_nan(self) -> None:
        rates = per_90(pd.Series([1.0]), pd.Series([45]), min_minutes=90)
        assert rates.isna().all()

    def test_zero_minutes_safe(self) -> None:
        rates = per_90(pd.Series([1.0]), pd.Series([0]))
        assert rates.isna().all()

    @given(
        goals=st.floats(min_value=0, max_value=50, allow_nan=False),
        minutes=st.integers(min_value=90, max_value=5000),
    )
    @settings(max_examples=100)
    def test_monotonicity(self, goals: float, minutes: int) -> None:
        """Doubling goals doubles the rate; doubling minutes halves it."""
        base = per_90(pd.Series([goals]), pd.Series([minutes])).iloc[0]
        doubled = per_90(pd.Series([goals * 2]), pd.Series([minutes])).iloc[0]
        halved = per_90(pd.Series([goals]), pd.Series([minutes * 2])).iloc[0]
        assert doubled == pytest.approx(base * 2, rel=1e-9)
        assert halved == pytest.approx(base / 2, rel=1e-9)


class TestMonthsToContractEnd:
    def test_future_date(self) -> None:
        ref = datetime(2025, 7, 1)
        out = months_to_contract_end(pd.Series(["2027-07-01"]), reference=ref)
        assert out.iloc[0] == pytest.approx(24.0, abs=0.5)

    def test_past_date_clipped(self) -> None:
        ref = datetime(2025, 7, 1)
        out = months_to_contract_end(pd.Series(["2020-01-01"]), reference=ref)
        assert out.iloc[0] == 0.0

    def test_null(self) -> None:
        out = months_to_contract_end(pd.Series([None], dtype=object))
        assert out.isna().all()

    def test_unparseable(self) -> None:
        out = months_to_contract_end(pd.Series(["not-a-date"]))
        assert out.isna().all()


class TestEncodeFoot:
    def test_known_values(self) -> None:
        out = encode_foot(pd.Series(["Right", "left", "either", "Both"]))
        assert out.tolist() == ["right", "left", "either", "either"]

    def test_unknown_becomes_null(self) -> None:
        out = encode_foot(pd.Series(["dominant"]))
        assert out.isna().all()
