from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from value_wage.config import SplitConfig
from value_wage.splits import group_kfold_indices, make_splits


def _mini_frame() -> pd.DataFrame:
    rows = []
    for season in ("2023-24", "2024-25", "2025-26"):
        for pid in range(10):
            rows.append(
                {
                    "player": f"P{pid}",
                    "player_id": pid,
                    "season": season,
                    "market_value_eur": 1_000_000 * (pid + 1),
                    "wage_pw": 10_000 * (pid + 1),
                }
            )
    return pd.DataFrame(rows)


def test_make_splits_season_partition() -> None:
    df = _mini_frame()
    cfg = SplitConfig()
    s = make_splits(df, "value", cfg=cfg)
    assert set(s.train["season"].unique()) == {"2023-24"}
    assert set(s.val["season"].unique()) == {"2024-25"}
    assert set(s.test["season"].unique()) == {"2025-26"}


def test_make_splits_rejects_overlap() -> None:
    cfg = SplitConfig(train_seasons=("2023-24",), val_seasons=("2023-24",), test_seasons=("2025-26",))
    with pytest.raises(ValueError, match="overlap"):
        make_splits(_mini_frame(), "value", cfg=cfg)


def test_make_splits_errors_on_missing_target_col() -> None:
    df = _mini_frame().drop(columns=["market_value_eur"])
    with pytest.raises(KeyError):
        make_splits(df, "value")


def test_make_splits_errors_on_empty_slice() -> None:
    df = _mini_frame()
    df.loc[df["season"] == "2024-25", "market_value_eur"] = np.nan
    with pytest.raises(ValueError, match="val slice"):
        make_splits(df, "value")


def test_group_kfold_no_player_leakage() -> None:
    df = _mini_frame()
    df["player_id"] = np.repeat(np.arange(10), 3)  # 10 players × 3 seasons
    folds = group_kfold_indices(df, n_splits=3)
    assert len(folds) == 3
    for tr, va in folds:
        tr_players = set(df.iloc[tr]["player_id"])
        va_players = set(df.iloc[va]["player_id"])
        assert not (tr_players & va_players), "Player leaked across GroupKFold split"
