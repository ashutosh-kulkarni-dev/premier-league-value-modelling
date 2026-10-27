from __future__ import annotations

import numpy as np
import pytest

from value_wage.evaluate import bootstrap_metric_ci, compute_metrics


def test_metrics_on_perfect_fit() -> None:
    y = np.log1p(np.array([1e5, 1e6, 1e7]))
    m = compute_metrics(y, y)
    assert m.mae_log == pytest.approx(0.0)
    assert m.r2_log == pytest.approx(1.0)
    assert m.spearman == pytest.approx(1.0)
    assert m.n == 3


def test_metrics_length_mismatch() -> None:
    with pytest.raises(ValueError):
        compute_metrics(np.array([1.0]), np.array([1.0, 2.0]))


def test_bootstrap_mae_covers_point() -> None:
    rng = np.random.default_rng(0)
    y_true = rng.normal(10, 1, size=200)
    y_pred = y_true + rng.normal(0, 0.1, size=200)
    point, lo, hi = bootstrap_metric_ci(y_true, y_pred, metric="mae", n_bootstrap=200)
    assert lo <= point <= hi
    assert hi - lo > 0
