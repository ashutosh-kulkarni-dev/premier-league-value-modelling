"""Model wrappers with a single uniform interface."""

from value_wage.models.base import build_model
from value_wage.models.baselines import median_by_position
from value_wage.models.boosters import (
    build_catboost,
    build_hgbr,
    build_lgbm,
    build_xgb,
)

__all__ = [
    "build_catboost",
    "build_hgbr",
    "build_lgbm",
    "build_model",
    "build_xgb",
    "median_by_position",
]
