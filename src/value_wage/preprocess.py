"""Preprocessing pipelines, one per model family.

Two shapes:
- `make_linear_preprocessor`: median-impute numerics + one-hot categoricals + standard-scale.
  Used by the ridge baseline, which cannot handle NaN or strings.
- `make_booster_preprocessor`: pass numerics through (keeping NaN — all four boosters handle it
  natively), ordinal-encode categoricals (fine for tree splits, cheaper than OHE on tiny data).

Both are `ColumnTransformer` instances so they serialise atomically with the model.
"""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler


def make_linear_preprocessor(
    numeric_cols: list[str],
    categorical_cols: list[str],
) -> ColumnTransformer:
    """For ridge / linear baselines."""
    numeric_pipe = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def make_booster_preprocessor(
    numeric_cols: list[str],
    categorical_cols: list[str],
) -> ColumnTransformer:
    """For LightGBM / XGBoost / CatBoost / HGBR.

    Numerics pass through unchanged (NaN preserved — the model handles it).
    Categoricals get ordinal codes with explicit `unknown_value=-1`, which the trees treat
    as just another level; this avoids OHE blowup on tiny data.
    """
    categorical_pipe = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            (
                "ord",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                    encoded_missing_value=-1,
                    dtype=float,
                ),
            ),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def transformed_feature_names(
    preprocessor: ColumnTransformer,
    numeric_cols: list[str],
    categorical_cols: list[str],
) -> list[str]:
    """Robust feature-name recovery: for the booster preprocessor, numeric names come through
    unchanged and categoricals keep their original names (ordinal doesn't rename).
    Falls back to sklearn's `get_feature_names_out` for the linear preprocessor.
    """
    try:
        return list(preprocessor.get_feature_names_out())
    except Exception:
        return [*numeric_cols, *categorical_cols]
