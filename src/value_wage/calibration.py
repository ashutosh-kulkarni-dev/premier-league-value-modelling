"""Conformal interval calibration via MAPIE.

The pipeline's quantile LightGBM intervals are badly miscalibrated (actual coverage
~30-50% at an 80% nominal target). This module replaces them with split-conformal
intervals that come with a distribution-free coverage guarantee:

- Fit the point model on the TRAIN season only (not train+val).
- Conformalize on the VAL season (held-out set the model has never seen).
- Produce intervals that satisfy P(y_true in [lo, hi]) ≥ 1 − α on exchangeable data.

The resulting intervals are usually wider than the raw quantile intervals, but they
*actually* achieve the stated coverage. For the mispricing board that matters — a
narrow uncalibrated interval produces fake outliers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from mapie.regression import CrossConformalRegressor, SplitConformalRegressor
from sklearn.model_selection import GroupKFold

from value_wage.config import Settings, TARGET_COLUMN, Target, get_settings
from value_wage.features import feature_lists_for
from value_wage.models.base import build_model
from value_wage.preprocess import make_booster_preprocessor
from value_wage.splits import make_splits
from value_wage.train import _normalize_na


@dataclass
class CalibratedPredictor:
    """A fitted booster + its conformal wrapper. Serialisable via joblib."""

    model: Any
    conformal: Any  # SplitConformalRegressor or CrossConformalRegressor
    numeric_cols: list[str]
    categorical_cols: list[str]
    preprocessor: Any
    confidence_level: float

    def predict(self, frame: pd.DataFrame) -> dict[str, np.ndarray]:
        feature_names = [*self.numeric_cols, *self.categorical_cols]
        X = _normalize_na(frame[feature_names])
        X_t = self.preprocessor.transform(X)
        point, intervals = self.conformal.predict_interval(X_t)
        # intervals shape: (n_samples, 2, 1) for a single alpha
        lo = intervals[:, 0, 0]
        hi = intervals[:, 1, 0]
        return {
            "pred": np.asarray(point, dtype=float),
            "lower": np.asarray(lo, dtype=float),
            "upper": np.asarray(hi, dtype=float),
        }


def calibrate(
    model_name: str,
    target: Target,
    features: pd.DataFrame,
    params: dict[str, Any],
    *,
    confidence_level: float = 0.8,
    conformity_score: str = "absolute",
    method: str = "split",
    settings: Settings | None = None,
) -> CalibratedPredictor:
    """Refit the point model on train-only, then conformalize on val.

    The returned predictor's point predictions are from a model trained on strictly less
    data than the production `train+val` artifact. For evaluation purposes that's fine
    (test performance is what matters); for a production deployment, swap in the
    production point model via `prefit=True` and conformalize on a hold-out.
    """
    settings = settings or get_settings()
    feats = settings.load_features()
    numeric_cols, categorical_cols = feature_lists_for(target, feats)
    target_col = TARGET_COLUMN[target]
    split = make_splits(features, target, cfg=settings.split)

    pre = make_booster_preprocessor(numeric_cols, categorical_cols)

    if method == "split":
        # Fit on train only, calibrate on val. Honest independence but less data for the model.
        train_X = _normalize_na(split.train[[*numeric_cols, *categorical_cols]])
        train_y = np.log1p(split.train[target_col].to_numpy(dtype=float))
        val_X = _normalize_na(split.val[[*numeric_cols, *categorical_cols]])
        val_y = np.log1p(split.val[target_col].to_numpy(dtype=float))

        X_train_t = pre.fit_transform(train_X)
        X_val_t = pre.transform(val_X)

        model = build_model(model_name, **params)
        model.fit(X_train_t, train_y)

        conformal = SplitConformalRegressor(
            estimator=model,
            confidence_level=confidence_level,
            conformity_score=conformity_score,
            prefit=True,
        )
        conformal.conformalize(X_val_t, val_y)
        return CalibratedPredictor(
            model=model, conformal=conformal,
            numeric_cols=numeric_cols, categorical_cols=categorical_cols,
            preprocessor=pre, confidence_level=confidence_level,
        )

    if method == "cross":
        # CV+ (Barber et al. 2021): uses all train+val data via GroupKFold on player_id.
        # Trains K out-of-fold models; conformity scores come from each fold's held-out predictions.
        # Returns a wrapper that at predict time ensembles the K models and inflates intervals
        # by the conformal quantile — tighter and better-calibrated for small datasets.
        slab = pd.concat([split.train, split.val], ignore_index=True)
        slab_X = _normalize_na(slab[[*numeric_cols, *categorical_cols]])
        slab_y = np.log1p(slab[target_col].to_numpy(dtype=float))
        groups = slab["player_id"].to_numpy()
        X_t = pre.fit_transform(slab_X)

        model = build_model(model_name, **params)
        gkf = GroupKFold(n_splits=min(5, len(np.unique(groups))))
        conformal = CrossConformalRegressor(
            estimator=model,
            confidence_level=confidence_level,
            conformity_score=conformity_score,
            method="plus",
            cv=gkf,
        )
        conformal.fit_conformalize(X_t, slab_y, groups=groups)
        # CrossConformal's internal ensemble becomes the point predictor; expose it uniformly.
        return CalibratedPredictor(
            model=model, conformal=conformal,
            numeric_cols=numeric_cols, categorical_cols=categorical_cols,
            preprocessor=pre, confidence_level=confidence_level,
        )

    raise ValueError(f"Unknown calibration method: {method!r}. Use 'split' or 'cross'.")
