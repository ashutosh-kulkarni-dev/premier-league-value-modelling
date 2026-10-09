"""Join-correctness tests pinned to real cases in the user's data.

Skip gracefully when the Kaggle dump isn't present, but FAIL loudly if the dump IS present
and the join produces the wrong result for Sancho or Disasi — these are canaries for the
multi-club / loan edge case.
"""

from __future__ import annotations

import pandas as pd
import pytest

from value_wage.config import get_settings
from value_wage.data import load_master
from value_wage.sources.transfermarkt_kaggle import (
    INFLATION_REFERENCE_COMPETITIONS,
    PL_COMPETITION_ID,
    _normalise_name,
    build_snapshot_frame,
    join_master_to_tm,
    load_tm_raw,
    market_inflation_index,
)

SETTINGS = get_settings()
TM_DIR = SETTINGS.paths.project_root / "data" / "raw" / "transfermarkt"
SANCHO_TM_ID = 401173
DISASI_TM_ID = 386047
ODEGAARD_TM_ID = 316264


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Martin Ødegaard", "martin odegaard"),   # Ø must fold to o, not vanish
        ("Albert Grønbæk", "albert gronbaek"),  # ø + æ
        ("Rasmus Højlund", "rasmus hojlund"),
        ("Wojciech Szczęsny", "wojciech szczesny"),  # ę is NFKD-decomposable
        ("Matıas", "matias"),                      # dotless ı
        ("Pervis Estupiñán", "pervis estupinan"),  # combining diacritics still fine
    ],
)
def test_normalise_name_transliterates_baked_in_latin(raw: str, expected: str) -> None:
    """Regression: baked-in Latin-extended letters (Ø, æ, ł, ı…) must transliterate, not be
    deleted by the ascii fold. Deleting them silently broke the TM name join (e.g. Ødegaard
    → 'degaard', unmatchable against the master's 'odegaard')."""
    assert _normalise_name(raw) == expected


def _dump_present() -> bool:
    return (TM_DIR / "players.csv").exists() and (TM_DIR / "player_valuations.csv").exists()


@pytest.fixture(scope="module")
def joined_frame() -> pd.DataFrame:
    if not _dump_present():
        pytest.skip("TM Kaggle dump not present at data/raw/transfermarkt/.")
    if not SETTINGS.paths.raw_master.exists():
        pytest.skip("FM/Understat master not present; run `vw data build` first.")
    master = load_master().frame
    tm = load_tm_raw(TM_DIR)
    snapshots = build_snapshot_frame(tm, pl_only=True)
    joined, _ = join_master_to_tm(master, snapshots, tm)
    return joined


def _row(df: pd.DataFrame, player_substr: str, season: str, snapshot: str) -> pd.Series:
    hit = df[
        df["player"].str.contains(player_substr, case=False, na=False)
        & (df["season"] == season)
        & (df["snapshot"] == snapshot)
    ]
    assert len(hit) == 1, f"Expected 1 row for {player_substr}/{season}/{snapshot}, got {len(hit)}"
    return hit.iloc[0]


def test_sancho_2023_24_start_matches_at_pass_1(joined_frame: pd.DataFrame) -> None:
    """Season start: Sancho was at Man Utd per TM and per master — pass 1 exact club match."""
    row = _row(joined_frame, "Jadon Sancho", "2023-24", "start")
    assert row["join_pass"] == 1, f"Expected pass 1, got {row['join_pass']}"
    assert int(row["player_tm_id"]) == SANCHO_TM_ID
    assert "united" in str(row["current_club_tm"]).lower()


def test_sancho_2023_24_end_matches_at_pass_2(joined_frame: pd.DataFrame) -> None:
    """Season end: Sancho was on loan at Dortmund (not PL). TM snapshot shows Dortmund,
    master still shows Man Utd — pass 1 fails. Pass 2 should catch it because transfers.csv
    for season 23/24 lists Dortmund among his clubs.
    """
    row = _row(joined_frame, "Jadon Sancho", "2023-24", "end")
    assert row["join_pass"] == 2, (
        f"Expected pass 2 (loan destination), got {row['join_pass']}. "
        f"TM club: {row['current_club_tm']}"
    )
    assert int(row["player_tm_id"]) == SANCHO_TM_ID
    assert "dortmund" in str(row["current_club_tm"]).lower()
    # Value sanity: Sancho's end-of-23/24 TM value was €30M.
    assert 20_000_000 <= float(row["market_value_eur"]) <= 50_000_000


def test_disasi_2024_25_end_matches(joined_frame: pd.DataFrame) -> None:
    """Disasi was Chelsea → Villa (loan Jan 2025). Master shows Villa; TM end-of-season
    snapshot (May 2025) should also show Villa. Pass 1 should succeed on exact club.
    """
    row = _row(joined_frame, "Axel Disasi", "2024-25", "end")
    assert pd.notna(row["join_pass"]), "Disasi 24/25 end snapshot didn't match any pass."
    assert int(row["player_tm_id"]) == DISASI_TM_ID
    # Villa or Chelsea are both acceptable — the snapshot date floats, but the join should land.


def test_odegaard_matches_all_snapshots(joined_frame: pd.DataFrame) -> None:
    """Canary for the Ø-folding bug: TM stores 'Martin Ødegaard', the master 'Martin Odegaard'.
    Before the _normalise_name transliteration fix the Ø was deleted ('degaard'), so every
    Ødegaard snapshot missed and he vanished from the UI. All six snapshots must now match."""
    ode = joined_frame[joined_frame["player"].str.contains("degaard", case=False, na=False)]
    matched = ode[ode["join_pass"].notna()]
    assert len(matched) == 6, f"Expected 6 matched Ødegaard snapshots, got {len(matched)}."
    assert set(matched["player_tm_id"].astype(int)) == {ODEGAARD_TM_ID}


def test_inflation_reference_excludes_premier_league() -> None:
    """The inflation index MUST be built from a population disjoint from the GB1 targets, or it
    leaks. Guard the reference-competition set so GB1 can never be added to it."""
    assert PL_COMPETITION_ID not in INFLATION_REFERENCE_COMPETITIONS


def test_market_inflation_index_is_base_normalised() -> None:
    """Index is normalised to the base season = 1.0 and is a non-leaky exogenous signal."""
    if not _dump_present():
        pytest.skip("TM Kaggle dump not present.")
    idx = market_inflation_index(TM_DIR / "player_valuations.csv", base_season="2023-24")
    assert idx["2023-24"] == 1.0
    # Market inflated over the window, so later seasons should be >= base (not below).
    assert idx["2025-26"] >= 1.0
    assert all(v > 0 for v in idx.values())


def test_coverage_is_reasonable(joined_frame: pd.DataFrame) -> None:
    """At least 60% of (player, season, snapshot) rows should match. Below that, the join is
    probably broken (name normalisation issue, wrong season window, etc.)."""
    matched = joined_frame["join_pass"].notna().sum()
    total = len(joined_frame)
    rate = matched / total if total else 0.0
    assert rate >= 0.60, f"Match rate too low: {rate:.1%} ({matched}/{total})"
