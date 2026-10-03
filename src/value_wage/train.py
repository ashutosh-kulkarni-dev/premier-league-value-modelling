"""End-to-end training orchestration: preprocess, fit, persist, predict.

Each `train_*` function returns a `TrainedArtifact` that bundles everything the downstream
stages (evaluate, explain, mispricing, Streamlit) need: the preprocessor, the fitted model,
the feature-name list, and metadata. Serialisation uses joblib for speed and sklearn compat.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer

from value_wage.config import TARGET_COLUMN, Model, Settings, Target, get_settings
from value_wage.features import feature_lists_for
from value_wage.models.base import build_model, build_quantile_model
from value_wage.preprocess import make_booster_preprocessor
from value_wage.splits import make_splits


@dataclass
class TrainedArtifact:
    model_name: Model
    target: Target
    preprocessor: ColumnTransformer
    model: Any
    quantile_lower: Any | None
    quantile_upper: Any | None
    numeric_cols: list[str]
    categorical_cols: list[str]
    best_params: dict[str, Any]

    def to_dir(self, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.preprocessor, out_dir / "preprocessor.joblib")
        joblib.dump(self.model, out_dir / "model.joblib")
        if self.quantile_lower is not None:
            joblib.dump(self.quantile_lower, out_dir / "quantile_lower.joblib")
        if self.quantile_upper is not None:
            joblib.dump(self.quantile_upper, out_dir / "quantile_upper.joblib")
        (out_dir / "metadata.json").write_text(
            json.dumps(
                {
                    "model_name": self.model_name,
                    "target": self.target,
                    "numeric_cols": self.numeric_cols,
                    "categorical_cols": self.categorical_cols,
                    "best_params": self.best_params,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def from_dir(cls, in_dir: Path) -> TrainedArtifact:
        meta = json.loads((in_dir / "metadata.json").read_text(encoding="utf-8"))
        q_low = in_dir / "quantile_lower.joblib"
        q_hi = in_dir / "quantile_upper.joblib"
        return cls(
            model_name=meta["model_name"],
            target=meta["target"],
            preprocessor=joblib.load(in_dir / "preprocessor.joblib"),
            model=joblib.load(in_dir / "model.joblib"),
            quantile_lower=joblib.load(q_low) if q_low.exists() else None,
            quantile_upper=joblib.load(q_hi) if q_hi.exists() else None,
            numeric_cols=meta["numeric_cols"],
            categorical_cols=meta["categorical_cols"],
            best_params=meta["best_params"],
        )


def _fit_preprocessor(
    train_frame: pd.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
) -> tuple[ColumnTransformer, pd.DataFrame]:
    pre = make_booster_preprocessor(numeric_cols, categorical_cols)
    X_train = _normalize_na(train_frame[[*numeric_cols, *categorical_cols]])
    Xt = pre.fit_transform(X_train)
    feature_names = [*numeric_cols, *categorical_cols]
    return pre, pd.DataFrame(Xt, columns=feature_names, index=train_frame.index)


def _normalize_na(frame: pd.DataFrame) -> pd.DataFrame:
    """Coerce pandas-nullable NAs (pd.NA / StringDtype) to numpy-compatible missing values.

    sklearn's SimpleImputer and friends do `X != X` to detect NaN, which fails on `pd.NA`
    (`TypeError: boolean value of NA is ambiguous`). We only touch dtypes that produce
    `pd.NA` — object/string columns become plain Python objects with `None` for NAs.
    """
    out = frame.copy()
    for col in out.columns:
        if pd.api.types.is_string_dtype(out[col]) or out[col].dtype == object:
            out[col] = out[col].astype(object).where(out[col].notna(), None)
    return out


def _apply(preprocessor: ColumnTransformer, frame: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    X = frame[feature_names] if all(c in frame.columns for c in feature_names) else frame.copy()
    arr = preprocessor.transform(_normalize_na(X[feature_names]))
    return pd.DataFrame(arr, columns=feature_names, index=frame.index)


def _assert_no_leakage(
    numeric_cols: list[str],
    categorical_cols: list[str],
    excluded: list[str],
) -> None:
    """Hard stop: no excluded column may appear in the model's feature list.

    This is the belt-and-braces check against the FM `value_high` / `sell_value` /
    `release_clause` sneaking back into features of the Transfermarkt-targeted value model.
    """
    bad = sorted(set(numeric_cols + categorical_cols) & set(excluded))
    if bad:
        raise ValueError(
            f"Leakage check failed: features {bad} are in the model's feature list "
            "but also marked `excluded` in config/features.yaml. Fix the config, or "
            "remove them from `shared_numeric` / `shared_categorical`."
        )


def train_booster(
    model_name: Model,
    target: Target,
    features: pd.DataFrame,
    *,
    params: dict[str, Any] | None = None,
    fit_quantiles: bool = True,
    quantile_alphas: tuple[float, float] = (0.1, 0.9),
    settings: Settings | None = None,
) -> TrainedArtifact:
    """Fit one booster on train+val, with optional lower/upper quantile companions.

    Returns the artifact; does not write to disk. Call `.to_dir()` to persist.
    """
    settings = settings or get_settings()
    feature_lists = settings.load_features()
    numeric_cols, categorical_cols = feature_lists_for(target, feature_lists)
    _assert_no_leakage(numeric_cols, categorical_cols, feature_lists.excluded)
    target_col = TARGET_COLUMN[target]
    params = params or {}

    split = make_splits(features, target, cfg=settings.split)
    fit_frame = pd.concat([split.train, split.val], axis=0, ignore_index=True)

    pre, X_fit = _fit_preprocessor(fit_frame, numeric_cols, categorical_cols)
    y_fit = np.log1p(fit_frame[target_col].to_numpy(dtype=float))

    model = build_model(model_name, **params)
    model.fit(X_fit, y_fit)

    q_low = q_hi = None
    if fit_quantiles:
        try:
            lo_model = build_quantile_model(model_name, alpha=quantile_alphas[0], **params)
            hi_model = build_quantile_model(model_name, alpha=quantile_alphas[1], **params)
            lo_model.fit(X_fit, y_fit)
            hi_model.fit(X_fit, y_fit)
            q_low, q_hi = lo_model, hi_model
        except Exception as exc:
            raise RuntimeError(
                f"Quantile fit for model '{model_name}' failed: {exc!r}. "
                "If this is an alpha/parameter conflict, pass fit_quantiles=False and use the "
                "conformal fallback via mapie downstream."
            ) from exc

    return TrainedArtifact(
        model_name=model_name,
        target=target,
        preprocessor=pre,
        model=model,
        quantile_lower=q_low,
        quantile_upper=q_hi,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        best_params=params,
    )


def predict(artifact: TrainedArtifact, frame: pd.DataFrame) -> dict[str, np.ndarray]:
    """Return {pred, lower, upper} on the log scale for a given frame."""
    feature_names = [*artifact.numeric_cols, *artifact.categorical_cols]
    X = _apply(artifact.preprocessor, frame, feature_names)
    out: dict[str, np.ndarray] = {"pred": np.asarray(artifact.model.predict(X), dtype=float)}
    if artifact.quantile_lower is not None and artifact.quantile_upper is not None:
        lo = np.asarray(artifact.quantile_lower.predict(X), dtype=float)
        hi = np.asarray(artifact.quantile_upper.predict(X), dtype=float)
        # Guard against the two quantile models crossing (rare but possible on small folds).
        lo_fixed = np.minimum(lo, hi)
        hi_fixed = np.maximum(lo, hi)
        out["lower"] = lo_fixed
        out["upper"] = hi_fixed
    return out
