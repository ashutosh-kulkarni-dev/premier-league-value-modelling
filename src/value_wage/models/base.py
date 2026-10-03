"""One factory to rule them all.

Every booster is built through `build_model(name, **params)`. The returned object is a
scikit-learn-compatible regressor (has `.fit`, `.predict`) so the rest of the pipeline
(tuning, evaluation, SHAP) does not branch on model type.

Quantile variants have their own factory because the parameter names differ per library.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sklearn.base import BaseEstimator, RegressorMixin

from value_wage.config import Model


@dataclass(frozen=True)
class ModelSpec:
    """Declarative description of one booster: name + the Optuna search space."""

    name: Model
    search_space: dict[str, Any]
    fixed_params: dict[str, Any]


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
    raise ValueError(f"Unknown model name: {name!r}")


def build_quantile_model(name: Model, alpha: float, **params: Any) -> BaseEstimator:
    """Instantiate a quantile regressor for the same library."""
    if name == "lgbm":
        from value_wage.models.boosters import build_lgbm_quantile
        return build_lgbm_quantile(alpha=alpha, **params)
    if name == "xgb":
        from value_wage.models.boosters import build_xgb_quantile
        return build_xgb_quantile(alpha=alpha, **params)
    if name == "catboost":
        from value_wage.models.boosters import build_catboost_quantile
        return build_catboost_quantile(alpha=alpha, **params)
    if name == "hgbr":
        from value_wage.models.boosters import build_hgbr_quantile
        return build_hgbr_quantile(alpha=alpha, **params)
    raise ValueError(f"Quantile objective not supported for model {name!r}")


class SklearnLike(RegressorMixin, BaseEstimator):
    """Marker base — purely for type-narrowing in callers that expect a sklearn regressor."""
