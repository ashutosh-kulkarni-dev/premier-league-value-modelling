"""Optuna tuning harness.

Shared across all four boosters:
- Same objective: `regression_l1` / MAE on log1p(target).
- Same CV: GroupKFold on `player_id` within the train+val slab.
- Same sampler: TPE seeded at `optuna.sampler_seed`.
- Same pruner: MedianPruner with the configured warmup.
- Resumable: results persist to SQLite at `settings.paths.optuna_db`.
- Logged: every trial's params + fold scores land in MLflow under one run per study.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mlflow
import numpy as np
import optuna
import pandas as pd
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from sklearn.metrics import mean_absolute_error

from value_wage.config import TARGET_COLUMN, Model, Settings, Target, get_settings
from value_wage.features import feature_lists_for
from value_wage.models.base import build_model
from value_wage.preprocess import make_booster_preprocessor
from value_wage.splits import group_kfold_indices, make_splits
from value_wage.train import _normalize_na


@dataclass(frozen=True)
class TuningResult:
    model: Model
    target: Target
    best_params: dict[str, Any]
    best_value: float  # mean fold MAE on log target
    n_trials: int
    study_name: str


SEARCH_SPACES: dict[Model, callable] = {}  # type: ignore[type-arg]


def _suggest_lgbm(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "num_leaves": trial.suggest_int("num_leaves", 15, 127, log=True),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 5, 60),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
    }


def _suggest_xgb(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "min_child_weight": trial.suggest_float("min_child_weight", 0.5, 10.0, log=True),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "gamma": trial.suggest_float("gamma", 1e-8, 5.0, log=True),
    }


def _suggest_catboost(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "depth": trial.suggest_int("depth", 4, 10),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0, log=True),
        "random_strength": trial.suggest_float("random_strength", 0.1, 10.0, log=True),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
        "border_count": trial.suggest_int("border_count", 32, 254),
    }


def _suggest_hgbr(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 15, 127, log=True),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 60),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "l2_regularization": trial.suggest_float("l2_regularization", 1e-8, 10.0, log=True),
    }


SEARCH_SPACES = {
    "lgbm": _suggest_lgbm,
    "xgb": _suggest_xgb,
    "catboost": _suggest_catboost,
    "hgbr": _suggest_hgbr,
}


def _cv_score(
    model_name: Model,
    params: dict[str, Any],
    frames: pd.DataFrame,
    target_col: str,
    numeric_cols: list[str],
    categorical_cols: list[str],
    n_splits: int,
) -> float:
    """Mean held-out MAE across `n_splits` GroupKFold folds, scored on log1p(target)."""
    y_log = np.log1p(frames[target_col].to_numpy(dtype=float))
    X = _normalize_na(frames[[*numeric_cols, *categorical_cols]].copy())

    folds = group_kfold_indices(frames, n_splits=n_splits)
    scores = []
    for tr_idx, va_idx in folds:
        pre = make_booster_preprocessor(numeric_cols, categorical_cols)
        X_tr = pre.fit_transform(X.iloc[tr_idx])
        X_va = pre.transform(X.iloc[va_idx])
        y_tr = y_log[tr_idx]
        y_va = y_log[va_idx]

        model = build_model(model_name, **params)
        model.fit(X_tr, y_tr)
        preds = model.predict(X_va)
        scores.append(mean_absolute_error(y_va, preds))
    return float(np.mean(scores))


def tune(
    model_name: Model,
    target: Target,
    features: pd.DataFrame,
    *,
    n_trials: int | None = None,
    settings: Settings | None = None,
) -> TuningResult:
    """Run Optuna for one (model, target) pair.

    Train+val seasons are combined into the CV slab; the test season is untouched.
    """
    settings = settings or get_settings()
    settings.paths.ensure()
    n_trials = n_trials or settings.optuna.n_trials
    target_col = TARGET_COLUMN[target]
    numeric_cols, categorical_cols = feature_lists_for(target, settings.load_features())

    split = make_splits(features, target, cfg=settings.split)
    cv_frame = pd.concat([split.train, split.val], axis=0, ignore_index=True)

    sampler = TPESampler(seed=settings.optuna.sampler_seed)
    pruner = MedianPruner(n_warmup_steps=settings.optuna.pruner_warmup_trials)
    study_name = f"{target}__{model_name}"

    storage = f"sqlite:///{settings.paths.optuna_db.as_posix()}"
    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        storage=storage,
        study_name=study_name,
        load_if_exists=True,
    )

    def objective(trial: optuna.Trial) -> float:
        params = SEARCH_SPACES[model_name](trial)
        return _cv_score(
            model_name,
            params,
            cv_frame,
            target_col,
            numeric_cols,
            categorical_cols,
            n_splits=settings.split.inner_cv_folds,
        )

    # Use the sqlite backend; the file-store backend is now opt-in only in recent MLflow.
    mlflow_db = settings.paths.mlruns_dir / "mlflow.db"
    settings.paths.mlruns_dir.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{mlflow_db.as_posix()}")
    mlflow.set_experiment(settings.mlflow_experiment)
    with mlflow.start_run(run_name=f"tune__{study_name}"):
        mlflow.log_params({"target": target, "model": model_name, "n_trials": n_trials})
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        mlflow.log_metric("best_cv_log_mae", study.best_value)
        mlflow.log_params({f"best__{k}": v for k, v in study.best_params.items()})

    return TuningResult(
        model=model_name,
        target=target,
        best_params=dict(study.best_params),
        best_value=float(study.best_value),
        n_trials=n_trials,
        study_name=study_name,
    )
