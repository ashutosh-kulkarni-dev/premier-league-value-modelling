"""SHAP explanations.

TreeExplainer is used because every main model is a tree booster. The explainer runs on
the preprocessed matrix (what the model actually saw), so feature names match the
model's internal view.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

matplotlib.use("Agg")


@dataclass(frozen=True)
class ShapResult:
    values: np.ndarray  # shape (n_samples, n_features)
    base_value: float
    feature_names: list[str]
    X: pd.DataFrame

    def global_importance(self) -> pd.DataFrame:
        imp = np.abs(self.values).mean(axis=0)
        return (
            pd.DataFrame({"feature": self.feature_names, "mean_abs_shap": imp})
            .sort_values("mean_abs_shap", ascending=False)
            .reset_index(drop=True)
        )


def explain_tree_model(
    model: Any,
    X_preprocessed: pd.DataFrame,
) -> ShapResult:
    """Compute SHAP values for a fitted tree-based model.

    The DataFrame should already be in the shape the model expects (post-preprocessor).
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_preprocessed)
    base = explainer.expected_value
    if isinstance(base, (list, np.ndarray)):
        base = float(np.asarray(base).ravel()[0])
    return ShapResult(
        values=np.asarray(shap_values),
        base_value=float(base),
        feature_names=list(X_preprocessed.columns),
        X=X_preprocessed,
    )


def save_global_summary_plot(result: ShapResult, out_path: Path, *, max_display: int = 20) -> None:
    plt.figure(figsize=(8, 6))
    shap.summary_plot(
        result.values,
        result.X,
        feature_names=result.feature_names,
        max_display=max_display,
        show=False,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()


def save_waterfall_plot(
    result: ShapResult,
    row_index: int,
    out_path: Path,
    *,
    max_display: int = 15,
) -> None:
    """Local SHAP waterfall for one prediction."""
    explanation = shap.Explanation(
        values=result.values[row_index],
        base_values=result.base_value,
        data=result.X.iloc[row_index].to_numpy(),
        feature_names=result.feature_names,
    )
    plt.figure(figsize=(8, 6))
    shap.plots.waterfall(explanation, max_display=max_display, show=False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
