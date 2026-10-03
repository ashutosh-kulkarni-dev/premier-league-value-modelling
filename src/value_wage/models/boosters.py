"""Thin factories around LightGBM, XGBoost, CatBoost, and sklearn HistGradientBoosting.

Each factory accepts the Optuna search-space keys verbatim; defaults are sensible for a
~1.5k-row tabular regression problem. Quantile variants are separate factories so the
objective-specific knobs are explicit.
"""

from __future__ import annotations

from typing import Any

from lightgbm import LGBMRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from xgboost import XGBRegressor

# Catboost import is deferred into functions — a cold import on Windows can be slow.

SEED = 42
N_ESTIMATORS = 2000  # paired with early stopping


def _lgbm_defaults(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "n_estimators": N_ESTIMATORS,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_data_in_leaf": 20,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 1,
        "lambda_l1": 0.0,
        "lambda_l2": 0.0,
        "objective": "regression_l1",
        "metric": "mae",
        "random_state": SEED,
        "verbosity": -1,
        "n_jobs": -1,
    }
    base.update(overrides)
    return base


def build_lgbm(**params: Any) -> LGBMRegressor:
    return LGBMRegressor(**_lgbm_defaults(**params))


def build_lgbm_quantile(alpha: float, **params: Any) -> LGBMRegressor:
    return LGBMRegressor(**_lgbm_defaults(objective="quantile", alpha=alpha, metric="quantile", **params))


def _xgb_defaults(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "n_estimators": N_ESTIMATORS,
        "learning_rate": 0.05,
        "max_depth": 6,
        "min_child_weight": 1.0,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
        "gamma": 0.0,
        "objective": "reg:absoluteerror",
        "eval_metric": "mae",
        "random_state": SEED,
        "tree_method": "hist",
        "n_jobs": -1,
        "verbosity": 0,
    }
    base.update(overrides)
    return base


def build_xgb(**params: Any) -> XGBRegressor:
    return XGBRegressor(**_xgb_defaults(**params))


def build_xgb_quantile(alpha: float, **params: Any) -> XGBRegressor:
    return XGBRegressor(
        **_xgb_defaults(
            objective="reg:quantileerror",
            quantile_alpha=alpha,
            eval_metric="mae",
            **params,
        )
    )


def _catboost_defaults(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "iterations": N_ESTIMATORS,
        "learning_rate": 0.05,
        "depth": 6,
        "l2_leaf_reg": 3.0,
        "random_strength": 1.0,
        "bagging_temperature": 1.0,
        "border_count": 128,
        "loss_function": "MAE",
        "eval_metric": "MAE",
        "random_seed": SEED,
        "allow_writing_files": False,
        "verbose": False,
    }
    base.update(overrides)
    return base


def build_catboost(**params: Any) -> Any:
    from catboost import CatBoostRegressor
    return CatBoostRegressor(**_catboost_defaults(**params))


def build_catboost_quantile(alpha: float, **params: Any) -> Any:
    from catboost import CatBoostRegressor
    return CatBoostRegressor(
        **_catboost_defaults(
            loss_function=f"Quantile:alpha={alpha}",
            eval_metric=f"Quantile:alpha={alpha}",
            **params,
        )
    )


def _hgbr_defaults(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "loss": "absolute_error",
        "max_iter": 1000,
        "learning_rate": 0.05,
        "max_leaf_nodes": 31,
        "min_samples_leaf": 20,
        "l2_regularization": 0.0,
        "random_state": SEED,
        "early_stopping": True,
        "validation_fraction": 0.15,
        "n_iter_no_change": 100,
    }
    base.update(overrides)
    return base


def build_hgbr(**params: Any) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(**_hgbr_defaults(**params))


def build_hgbr_quantile(alpha: float, **params: Any) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        **_hgbr_defaults(loss="quantile", quantile=alpha, **params)
    )
