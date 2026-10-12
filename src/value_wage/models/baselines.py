"""Baselines that the real model must beat.

- `median_by_position`: predicts the per-position median of the (log) target. The sanity floor.
- `ridge_pipeline`: the competent linear benchmark; a win over this justifies the GBM.
- `elasticnet_regressor`: regularized linear with built-in feature selection — the headline
  linear baseline. L2 handles the many correlated features; L1 zeros out irrelevant ones.
- `random_forest_regressor`: bagged-tree baseline — nonlinearity without boosting.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNetCV, Ridge
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


# Default l1_ratio grid: spans ridge-like (0.1) to near-lasso (0.99). ElasticNetCV picks the
# mix and penalty strength by cross-validation; pass `cv=<folds>` to use GroupKFold splits.
_ELASTICNET_L1_GRID = (0.1, 0.5, 0.7, 0.9, 0.95, 0.99)


def elasticnet_regressor(**params: Any) -> ElasticNetCV:
    """Self-tuning ElasticNet (expects scaled inputs — use `make_linear_preprocessor`).

    L2 shrinks correlated features together (age/age_sq, the fm_*_mean aggregates, per-90
    metrics); L1 zeros out irrelevant ones. Callers may pass `cv=<fold list>` for GroupKFold.
    """
    defaults: dict[str, Any] = {
        "l1_ratio": list(_ELASTICNET_L1_GRID),
        "max_iter": 5000,
        "random_state": 42,
    }
    defaults.update(params)
    return ElasticNetCV(**defaults)


def random_forest_regressor(**params: Any) -> RandomForestRegressor:
    """Bagged-tree baseline. Handles nonlinearity; no scaling needed (scaling is harmless)."""
    defaults: dict[str, Any] = {
        "n_estimators": 400,
        "min_samples_leaf": 5,
        "max_features": "sqrt",
        "n_jobs": -1,
        "random_state": 42,
    }
    defaults.update(params)
    return RandomForestRegressor(**defaults)
