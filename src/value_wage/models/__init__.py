"""Model wrappers with a single uniform interface."""

from value_wage.models.base import ModelSpec, build_model, build_quantile_model
from value_wage.models.baselines import median_by_position, ridge_pipeline
from value_wage.models.boosters import (
    build_catboost,
    build_hgbr,
    build_lgbm,
    build_xgb,
)

__all__ = [
    "ModelSpec",
    "build_catboost",
    "build_hgbr",
    "build_lgbm",
    "build_model",
    "build_quantile_model",
    "build_xgb",
    "median_by_position",
    "ridge_pipeline",
]
