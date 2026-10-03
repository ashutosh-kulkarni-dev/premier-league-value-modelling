"""Baselines that the real model must beat.

- `median_by_position`: predicts the per-position median of the (log) target. The sanity floor.
- `ridge_pipeline`: the competent linear benchmark; a win over this justifies the GBM.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

from value_wage.preprocess import make_linear_preprocessor


class PositionMedianRegressor:
    """Fallback-aware per-position median regressor. Works on the raw feature frame.

    Not a true sklearn estimator (no `_fit`, `get_params`); kept intentionally simple because
    it exists purely as a reference line on the metrics table.
    """

    def __init__(self, position_col: str = "primary_position") -> None:
        self.position_col = position_col
        self._per_position: dict[str, float] = {}
        self._global: float = 0.0

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> PositionMedianRegressor:
        if self.position_col not in X.columns:
            raise KeyError(f"PositionMedianRegressor needs `{self.position_col}` in X.")
        tmp = pd.DataFrame({"pos": X[self.position_col].astype("string"), "y": y})
        self._per_position = tmp.dropna().groupby("pos")["y"].median().to_dict()
        self._global = float(np.median(y))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        pos = X[self.position_col].astype("string")
        return np.array([self._per_position.get(p, self._global) for p in pos], dtype=float)


def median_by_position() -> PositionMedianRegressor:
    return PositionMedianRegressor()


def ridge_regressor(**params: Any) -> Ridge:
    """Bare ridge (without the preprocessor). Use `ridge_pipeline` to get the full pipeline."""
    defaults = {"alpha": 1.0, "random_state": 42}
    defaults.update(params)
    return Ridge(**defaults)


def ridge_pipeline(
    numeric_cols: list[str],
    categorical_cols: list[str],
    **ridge_params: Any,
) -> Pipeline:
    """Preprocessor + Ridge, as one serialisable pipeline."""
    return Pipeline(
        steps=[
            ("pre", make_linear_preprocessor(numeric_cols, categorical_cols)),
            ("ridge", ridge_regressor(**ridge_params)),
        ]
    )
