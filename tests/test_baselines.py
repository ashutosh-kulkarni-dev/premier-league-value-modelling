"""Baseline models (ElasticNet, Random Forest) train, predict, and beat the naive floor.

Skips gracefully when the processed feature matrix isn't present (e.g. a fresh clone without
the Kaggle dump), but runs fully in CI once `features.parquet` has been built.
"""

from __future__ import annotations

import numpy as np
import pytest

from value_wage.config import ALL_BASELINES, TARGET_COLUMN, get_settings
from value_wage.evaluate import compute_metrics
from value_wage.models.base import build_model
from value_wage.models.baselines import (
    elasticnet_regressor,
    median_by_position,
    random_forest_regressor,
)
from value_wage.splits import make_splits
from value_wage.train import predict, train_baseline

SETTINGS = get_settings()


def _features():
    import pandas as pd

    path = SETTINGS.paths.processed_features
    if not path.exists():
        pytest.skip("data/processed/features.parquet not present; run `vw features build`.")
    return pd.read_parquet(path)


def test_build_model_routes_baselines() -> None:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import ElasticNetCV

    assert isinstance(build_model("elasticnet"), ElasticNetCV)
    assert isinstance(build_model("rf"), RandomForestRegressor)
    assert set(ALL_BASELINES) == {"elasticnet", "rf"}


def test_factory_defaults() -> None:
    assert random_forest_regressor().n_estimators == 400
    assert random_forest_regressor(n_estimators=10).n_estimators == 10
    # ElasticNet searches a mix of ridge-like to near-lasso ratios.
    assert min(elasticnet_regressor().l1_ratio) >= 0.1


@pytest.mark.parametrize("model", ["elasticnet", "rf"])
def test_baseline_trains_predicts_and_beats_floor(model: str) -> None:
    features = _features()
    target = "value"
    target_col = TARGET_COLUMN[target]
    # Keep the forest small so the test is fast; ElasticNet ignores the kwarg path here.
    params = {"n_estimators": 60} if model == "rf" else {}
    artifact = train_baseline(model, target, features, params=params, settings=SETTINGS)

    split = make_splits(features, target, cfg=SETTINGS.split)
    preds = predict(artifact, split.test)["pred"]
    assert len(preds) == len(split.test)
    assert np.isfinite(preds).all()

    y_true_log = np.log1p(split.test[target_col].to_numpy(dtype=float))
    model_r2 = compute_metrics(y_true_log, preds).r2_log

    floor = median_by_position()
    floor.fit(split.train, np.log1p(split.train[target_col].to_numpy(dtype=float)))
    floor_r2 = compute_metrics(y_true_log, floor.predict(split.test)).r2_log

    assert model_r2 > floor_r2, f"{model} R²={model_r2:.3f} did not beat floor R²={floor_r2:.3f}"
