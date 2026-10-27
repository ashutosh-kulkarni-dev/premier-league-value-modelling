"""One factory to rule them all.

Every booster is built through `build_model(name, **params)`. The returned object is a
scikit-learn-compatible regressor (has `.fit`, `.predict`) so the rest of the pipeline
(tuning, evaluation, SHAP) does not branch on model type.
"""

from __future__ import annotations

from typing import Any

from sklearn.base import BaseEstimator

from value_wage.config import Model


def build_model(name: Model, **params: Any) -> BaseEstimator:
    """Instantiate a booster by name. Does not fit."""
    if name == "lgbm":
        from value_wage.models.boosters import build_lgbm
        return build_lgbm(**params)
    if name == "xgb":
        from value_wage.models.boosters import build_xgb
        return build_xgb(**params)
    if name == "catboost":
        from value_wage.models.boosters import build_catboost
        return build_catboost(**params)
    if name == "hgbr":
        from value_wage.models.boosters import build_hgbr
        return build_hgbr(**params)
    if name == "ridge":
        from value_wage.models.baselines import ridge_regressor
        return ridge_regressor(**params)
    if name == "elasticnet":
        from value_wage.models.baselines import elasticnet_regressor
        return elasticnet_regressor(**params)
    if name == "rf":
        from value_wage.models.baselines import random_forest_regressor
        return random_forest_regressor(**params)
    raise ValueError(f"Unknown model name: {name!r}")
