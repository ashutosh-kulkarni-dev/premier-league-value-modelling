"""Metrics + diagnostic plots.

All metrics are reported on both the log target (what the model trains on) and the
back-transformed scale (what the business audience understands). Bootstrap CIs quantify
how much to trust a single test-set point estimate.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, r2_score

matplotlib.use("Agg")  # headless — plots are saved, not shown


@dataclass(frozen=True)
class Metrics:
    mae_log: float
    r2_log: float
    spearman: float
    median_abs_pct_error: float  # back-transformed, median |pred/actual − 1|
    n: int


def compute_metrics(y_true_log: np.ndarray, y_pred_log: np.ndarray) -> Metrics:
    if len(y_true_log) != len(y_pred_log):
        raise ValueError(f"Length mismatch: y_true={len(y_true_log)}, y_pred={len(y_pred_log)}")
    if len(y_true_log) == 0:
        raise ValueError("Cannot compute metrics on empty arrays.")

    mae = float(mean_absolute_error(y_true_log, y_pred_log))
    r2 = float(r2_score(y_true_log, y_pred_log))
    rho = float(spearmanr(y_true_log, y_pred_log).correlation)

    y_true = np.expm1(y_true_log)
    y_pred = np.expm1(y_pred_log)
    # Guard the division: pred cannot be <= -1 in principle (expm1 of a log), but clip for safety.
    denom = np.maximum(y_true, 1.0)
    pct_err = float(np.median(np.abs(y_pred - y_true) / denom))

    return Metrics(
        mae_log=mae,
        r2_log=r2,
        spearman=rho,
        median_abs_pct_error=pct_err,
        n=len(y_true_log),
    )


def bootstrap_metric_ci(
    y_true_log: np.ndarray,
    y_pred_log: np.ndarray,
    *,
    metric: str = "mae",
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Return (point, lower, upper) for the chosen metric via nonparametric bootstrap."""
    rng = np.random.default_rng(seed)
    n = len(y_true_log)
    if n == 0:
        raise ValueError("Cannot bootstrap an empty array.")

    def score(idx: np.ndarray) -> float:
        if metric == "mae":
            return float(mean_absolute_error(y_true_log[idx], y_pred_log[idx]))
        if metric == "r2":
            return float(r2_score(y_true_log[idx], y_pred_log[idx]))
        if metric == "spearman":
            return float(spearmanr(y_true_log[idx], y_pred_log[idx]).correlation)
        raise ValueError(f"Unknown metric {metric!r}")

    stats = np.array(
        [score(rng.integers(0, n, size=n)) for _ in range(n_bootstrap)],
        dtype=float,
    )
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    point = score(np.arange(n))
    return float(point), float(lo), float(hi)


def plot_pred_vs_actual(
    y_true_log: np.ndarray,
    y_pred_log: np.ndarray,
    *,
    title: str,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true_log, y_pred_log, s=8, alpha=0.5)
    lo = float(min(y_true_log.min(), y_pred_log.min()))
    hi = float(max(y_true_log.max(), y_pred_log.max()))
    ax.plot([lo, hi], [lo, hi], ls="--", color="black", lw=1)
    ax.set_xlabel("Actual (log1p)")
    ax.set_ylabel("Predicted (log1p)")
    ax.set_title(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_residuals(
    y_true_log: np.ndarray,
    y_pred_log: np.ndarray,
    covariate: np.ndarray,
    *,
    covariate_name: str,
    title: str,
    out_path: Path,
) -> None:
    resid = y_pred_log - y_true_log
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(covariate, resid, s=8, alpha=0.5)
    ax.axhline(0, color="black", lw=1)
    ax.set_xlabel(covariate_name)
    ax.set_ylabel("Residual (pred − actual, log)")
    ax.set_title(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def interval_coverage(
    y_true_log: np.ndarray,
    lower_log: np.ndarray,
    upper_log: np.ndarray,
) -> float:
    """Fraction of actuals that fall inside the predicted interval."""
    if not (len(y_true_log) == len(lower_log) == len(upper_log)):
        raise ValueError("Length mismatch between y_true, lower, upper.")
    inside = (y_true_log >= lower_log) & (y_true_log <= upper_log)
    return float(inside.mean())


def summarise_predictions(
    frame: pd.DataFrame,
    y_true_log: np.ndarray,
    y_pred_log: np.ndarray,
    *,
    target_col: str,
) -> pd.DataFrame:
    """One row per prediction with actual, predicted, residual, back-transformed values."""
    out = frame.loc[:, ["player", "player_id", "season", "club", target_col]].copy()
    out["actual_log"] = y_true_log
    out["pred_log"] = y_pred_log
    out["actual"] = np.expm1(y_true_log)
    out["pred"] = np.expm1(y_pred_log)
    out["residual_log"] = y_pred_log - y_true_log
    out["residual_pct"] = (out["pred"] - out["actual"]) / out["actual"].replace(0, np.nan)
    return out
