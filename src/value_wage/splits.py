"""Train / val / test split by season, with GroupKFold on `player_id` for inner CV.

Why not random KFold:
- A random split would let the model see Player X's 2024-25 wage while predicting X's 2023-24 wage,
  inflating accuracy. The season split is time-honest; the group split inside train prevents the
  same player appearing in both inner-train and inner-val folds.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from value_wage.config import TARGET_COLUMN, SplitConfig, Target, get_settings


@dataclass(frozen=True)
class SplitFrames:
    """Masks + convenience accessors for one target's train/val/test slices.

    Only rows with a non-null target survive — the mispricing board predicts unlabelled
    rows separately via `predict_unlabelled_mask`.
    """

    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    unlabelled: pd.DataFrame
    target: Target
    target_col: str

    def summary(self) -> dict[str, int]:
        return {
            "train": len(self.train),
            "val": len(self.val),
            "test": len(self.test),
            "unlabelled": len(self.unlabelled),
        }


def _assert_disjoint_seasons(cfg: SplitConfig) -> None:
    all_seasons = (*cfg.train_seasons, *cfg.val_seasons, *cfg.test_seasons)
    if len(set(all_seasons)) != len(all_seasons):
        raise ValueError(f"Train/val/test seasons overlap: {all_seasons}")


def make_splits(
    features: pd.DataFrame,
    target: Target,
    *,
    cfg: SplitConfig | None = None,
) -> SplitFrames:
    """Build the four slices for one target.

    Raises:
        KeyError: if `season` or the target column is missing.
        ValueError: if the splits overlap or a slice is empty.
    """
    cfg = cfg or get_settings().split
    _assert_disjoint_seasons(cfg)
    tcol = TARGET_COLUMN[target]
    if "season" not in features.columns:
        raise KeyError("`season` column required to split; was it dropped by feature engineering?")
    if tcol not in features.columns:
        raise KeyError(f"Target column `{tcol}` missing from features.")

    def mask(seasons: tuple[str, ...]) -> pd.Series:
        return features["season"].isin(seasons)

    labelled = features[tcol].notna()

    train = features[mask(cfg.train_seasons) & labelled].copy()
    val = features[mask(cfg.val_seasons) & labelled].copy()
    test = features[mask(cfg.test_seasons) & labelled].copy()
    unlabelled = features[~labelled].copy()

    for name, slab in (("train", train), ("val", val), ("test", test)):
        if slab.empty:
            raise ValueError(
                f"{name} slice for target '{target}' is empty. "
                f"Seasons configured: train={cfg.train_seasons}, val={cfg.val_seasons}, "
                f"test={cfg.test_seasons}. Available seasons: "
                f"{sorted(features['season'].dropna().unique().tolist())}."
            )

    return SplitFrames(
        train=train,
        val=val,
        test=test,
        unlabelled=unlabelled,
        target=target,
        target_col=tcol,
    )


def group_kfold_indices(frame: pd.DataFrame, *, n_splits: int = 3) -> list[tuple[np.ndarray, np.ndarray]]:
    """Yield (train_idx, val_idx) pairs grouped by `player_id`.

    Positional indices into `frame`, not labels — ready for `.iloc`.
    """
    if "player_id" not in frame.columns:
        raise KeyError("`player_id` is required for GroupKFold.")
    groups = frame["player_id"].to_numpy()
    n_groups = int(pd.Series(groups).nunique())
    if n_groups < n_splits:
        raise ValueError(
            f"GroupKFold needs at least {n_splits} unique groups; got {n_groups}. "
            "Either reduce `inner_cv_folds` or widen the training window."
        )
    gkf = GroupKFold(n_splits=n_splits)
    X_dummy = np.zeros((len(frame), 1))
    return [
        (cast(np.ndarray, tr), cast(np.ndarray, va))
        for tr, va in gkf.split(X_dummy, groups=groups)
    ]
