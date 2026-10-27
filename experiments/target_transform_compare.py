"""Does the `log1p` target beat a direct (raw-unit) target for this dataset?

The production pipeline trains every regressor on `log1p(target)` and back-transforms
with `expm1` (see `value_wage.train` / `value_wage.evaluate`). This experiment answers
*why* by holding everything else fixed — same features, same season splits, same model —
and changing only what `y` the booster is trained on:

  - "log1p"  : y = log1p(target), predict, then expm1 back to raw units
  - "direct" : y = target in raw units, predict raw units directly

Both sets of predictions are scored on the **raw unit scale** (euros for value, €/week
for wage) so the comparison is apples-to-apples, plus log-scale MAE/R2 for reference.

This is an analysis script, not part of the serving pipeline: it modifies nothing in the
package and only writes a CSV + Markdown report under `reports/`.

Usage:
    python experiments/target_transform_compare.py                 # value, lgbm
    python experiments/target_transform_compare.py --target wage
    python experiments/target_transform_compare.py --target value --model hgbr
Or via the Makefile:
    make target-compare TARGET=value MODEL=lgbm
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from value_wage.config import TARGET_COLUMN, Model, Target, get_settings
from value_wage.features import feature_lists_for
from value_wage.models.base import build_model
from value_wage.preprocess import make_booster_preprocessor
from value_wage.splits import make_splits
from value_wage.train import _normalize_na

# Raw-unit metrics are reported divided by this for readability (1e6 → millions).
UNIT_SCALE = {"value": 1e6, "wage": 1e3}
UNIT_LABEL = {"value": "€m", "wage": "€k/wk"}


def _fit_predict_raw(
    y_transform: str,
    fit_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
    target_col: str,
    model_name: Model,
) -> np.ndarray:
    """Fit the model on `fit_frame` with the chosen target transform; return RAW-unit test preds.

    The preprocessor and feature set are identical to the production booster path; only the
    target transform (and its inverse) differ.
    """
    pre = make_booster_preprocessor(numeric_cols, categorical_cols)
    feat = [*numeric_cols, *categorical_cols]
    X_fit = pd.DataFrame(
        pre.fit_transform(_normalize_na(fit_frame[feat])), columns=feat, index=fit_frame.index
    )
    X_test = pd.DataFrame(
        pre.transform(_normalize_na(test_frame[feat])), columns=feat, index=test_frame.index
    )

    y_raw = fit_frame[target_col].to_numpy(dtype=float)
    y_fit = np.log1p(y_raw) if y_transform == "log1p" else y_raw

    model = build_model(model_name)
    model.fit(X_fit, y_fit)
    pred = np.asarray(model.predict(X_test), dtype=float)

    pred_raw = np.expm1(pred) if y_transform == "log1p" else pred
    # A direct-target booster can emit small negatives; the quantity can't be below zero.
    return np.clip(pred_raw, 0.0, None)


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Raw-unit metrics (what the audience reads) plus log-scale references."""
    denom = np.maximum(y_true, 1.0)
    return {
        "MAE_raw": float(mean_absolute_error(y_true, y_pred)),
        "RMSE_raw": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2_raw": float(r2_score(y_true, y_pred)),
        "Spearman": float(spearmanr(y_true, y_pred).correlation),
        "MedAPE": float(np.median(np.abs(y_pred - y_true) / denom)),
        "MAE_log": float(mean_absolute_error(np.log1p(y_true), np.log1p(y_pred))),
        "R2_log": float(r2_score(np.log1p(y_true), np.log1p(y_pred))),
    }


def run(target: Target, model_name: Model) -> pd.DataFrame:
    settings = get_settings()
    features = pd.read_parquet(settings.paths.processed_features)
    feature_lists = settings.load_features()
    numeric_cols, categorical_cols = feature_lists_for(target, feature_lists)
    target_col = TARGET_COLUMN[target]

    split = make_splits(features, target, cfg=settings.split)
    fit_frame = pd.concat([split.train, split.val], axis=0, ignore_index=True)
    test_frame = split.test.reset_index(drop=True)
    y_test = test_frame[target_col].to_numpy(dtype=float)

    rows = {
        transform: _metrics(
            y_test,
            _fit_predict_raw(
                transform, fit_frame, test_frame, numeric_cols, categorical_cols, target_col, model_name
            ),
        )
        for transform in ("log1p", "direct")
    }
    table = pd.DataFrame(rows).T
    table.index.name = "target_transform"
    return table


def _write_report(table: pd.DataFrame, target: Target, model_name: Model, settings) -> tuple[Path, Path]:
    reports = settings.paths.reports_dir
    reports.mkdir(parents=True, exist_ok=True)
    csv_path = reports / f"target_transform_comparison_{target}.csv"
    md_path = reports / f"target_transform_comparison_{target}.md"

    scale, label = UNIT_SCALE[target], UNIT_LABEL[target]
    disp = table.copy()
    disp["MAE_raw"] = disp["MAE_raw"] / scale
    disp["RMSE_raw"] = disp["RMSE_raw"] / scale
    disp = disp.rename(columns={"MAE_raw": f"MAE_{label}", "RMSE_raw": f"RMSE_{label}"})

    table.to_csv(csv_path)

    def better(col: str, higher_is_better: bool) -> str:
        winner = disp[col].idxmax() if higher_is_better else disp[col].idxmin()
        return winner

    def md_table(df: pd.DataFrame) -> str:
        # Hand-rolled Markdown so we don't depend on the optional `tabulate` package.
        cols = list(df.columns)
        head = "| " + " | ".join(["target_transform", *cols]) + " |"
        sep = "| " + " | ".join(["---"] * (len(cols) + 1)) + " |"
        body = [
            "| " + " | ".join([str(idx), *(f"{df.loc[idx, c]:.4f}" for c in cols)]) + " |"
            for idx in df.index
        ]
        return "\n".join([head, sep, *body])

    lines = [
        f"# Target transform comparison — {target} ({model_name})",
        "",
        "Same features, season splits, preprocessor and model; the only difference is whether the",
        f"model trains on `log1p({TARGET_COLUMN[target]})` or on the raw value directly. All metrics",
        f"are scored on the raw-unit scale (test split). `{label}` columns are in those units.",
        "",
        md_table(disp),
        "",
        "Lower is better: MAE, RMSE, MedAPE, MAE_log. Higher is better: R2, Spearman, R2_log.",
        "",
        "| Metric | Winner |",
        "|---|---|",
        f"| MAE ({label}) | {better(f'MAE_{label}', False)} |",
        f"| RMSE ({label}) | {better(f'RMSE_{label}', False)} |",
        f"| R2 (raw) | {better('R2_raw', True)} |",
        f"| MedAPE | {better('MedAPE', False)} |",
        f"| R2 (log1p) | {better('R2_log', True)} |",
        "",
    ]
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return csv_path, md_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=["value", "wage"], default="value")
    parser.add_argument(
        "--model",
        choices=["lgbm", "xgb", "catboost", "hgbr", "rf"],
        default="lgbm",
        help="Any tree model works; log1p vs direct is the thing under test.",
    )
    args = parser.parse_args()

    settings = get_settings()
    table = run(args.target, args.model)
    csv_path, md_path = _write_report(table, args.target, args.model, settings)

    scale, label = UNIT_SCALE[args.target], UNIT_LABEL[args.target]
    disp = table.copy()
    disp["MAE_raw"] = disp["MAE_raw"] / scale
    disp["RMSE_raw"] = disp["RMSE_raw"] / scale
    disp = disp.rename(columns={"MAE_raw": f"MAE_{label}", "RMSE_raw": f"RMSE_{label}"})
    pd.set_option("display.float_format", lambda v: f"{v:,.4f}")

    print(f"Target: {TARGET_COLUMN[args.target]}  |  model: {args.model}")
    print(disp.to_string())
    print(f"\nWrote {csv_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
