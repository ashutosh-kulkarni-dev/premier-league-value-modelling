"""Mispricing board + similarity (comparables) engine.

The board ranks test-season players by `(actual − predicted) / sigma_pred`, where
`sigma_pred` is derived from the 80% quantile interval (upper − lower) / 2.56. This
turns "€20M below prediction" into a z-like score that is comparable across the
value spectrum.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class MispricingBoard:
    full: pd.DataFrame
    undervalued: pd.DataFrame
    overvalued: pd.DataFrame

    def write(self, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        self.full.to_csv(out_dir / "mispricing_full.csv", index=False)
        self.undervalued.to_csv(out_dir / "mispricing_undervalued.csv", index=False)
        self.overvalued.to_csv(out_dir / "mispricing_overvalued.csv", index=False)


def build_board(
    predictions: pd.DataFrame,
    *,
    lower_log: np.ndarray | None = None,
    upper_log: np.ndarray | None = None,
    top_n: int = 20,
) -> MispricingBoard:
    """Rank by signed z-score of (actual − predicted) on the log scale.

    `predictions` must contain `actual_log`, `pred_log`. If `lower_log`/`upper_log` are
    passed, sigma is (upper − lower) / 2.5631 (80% interval under a normal). Otherwise
    a global sigma is used (useful before quantile models are fit).
    """
    df = predictions.copy()
    df["residual_log"] = df["actual_log"] - df["pred_log"]  # positive → market pays more than model
    if lower_log is not None and upper_log is not None:
        if not (len(df) == len(lower_log) == len(upper_log)):
            raise ValueError("Prediction/interval length mismatch.")
        sigma = (np.asarray(upper_log) - np.asarray(lower_log)) / 2.5631
        sigma = np.where(sigma <= 0, np.nan, sigma)
    else:
        sigma = np.full(len(df), float(np.std(df["residual_log"].to_numpy())))
    df["sigma_log"] = sigma
    df["z"] = df["residual_log"] / df["sigma_log"]

    df_sorted = df.sort_values("z")  # most negative z = overvalued by model (predicted > actual)
    # Convention: "undervalued" = market pays LESS than model (actual < pred), z < 0.
    # "overvalued" = market pays MORE than model (actual > pred), z > 0.
    undervalued = df_sorted.head(top_n).reset_index(drop=True)
    overvalued = df_sorted.tail(top_n).iloc[::-1].reset_index(drop=True)
    return MispricingBoard(full=df_sorted.reset_index(drop=True), undervalued=undervalued, overvalued=overvalued)


def nearest_comparables(
    query_row: pd.Series,
    pool_matrix: pd.DataFrame,
    pool_meta: pd.DataFrame,
    *,
    top_k: int = 5,
    exclude_player_id: int | None = None,
) -> pd.DataFrame:
    """Return the top-k nearest players to `query_row` in the preprocessed feature space.

    `pool_matrix` and `pool_meta` must be row-aligned.
    """
    if len(pool_matrix) != len(pool_meta):
        raise ValueError("pool_matrix and pool_meta have different lengths.")

    scaler = StandardScaler(with_mean=True, with_std=True).fit(pool_matrix)
    pool_scaled = scaler.transform(pool_matrix.fillna(pool_matrix.median(numeric_only=True)))
    query_scaled = scaler.transform(
        pd.DataFrame([query_row]).fillna(pool_matrix.median(numeric_only=True))
    )

    sims = cosine_similarity(query_scaled, pool_scaled).ravel()
    out = pool_meta.copy()
    out["similarity"] = sims
    if exclude_player_id is not None:
        out = out[out["player_id"] != exclude_player_id]
    return out.sort_values("similarity", ascending=False).head(top_k).reset_index(drop=True)
