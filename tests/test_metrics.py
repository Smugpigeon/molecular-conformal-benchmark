"""Test metric functions. Per CLAUDE.md §9."""

from __future__ import annotations

import numpy as np
import pytest

from src.utils.metrics import (
    classification_metrics,
    metrics_for_task,
    regression_metrics,
)


class TestRegressionMetrics:
    def test_perfect_prediction(self):
        y = np.array([1.0, 2.0, 3.0, 4.0])
        m = regression_metrics(y, y.copy())
        assert m["MAE"] == 0.0
        assert m["RMSE"] == 0.0
        assert m["PearsonR"] == pytest.approx(1.0)
        assert m["SpearmanRho"] == pytest.approx(1.0)

    def test_reports_all_four_metrics(self):
        # CLAUDE.md §9: must report 4 metrics.
        rng = np.random.default_rng(0)
        y_true = rng.normal(size=20)
        y_pred = y_true + rng.normal(scale=0.1, size=20)
        m = regression_metrics(y_true, y_pred)
        assert set(m) == {"MAE", "RMSE", "PearsonR", "SpearmanRho"}

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="shape mismatch"):
            regression_metrics(np.array([1.0, 2.0]), np.array([1.0]))


class TestClassificationMetrics:
    def test_perfect_separation(self):
        y_true = np.array([0, 0, 1, 1])
        y_score = np.array([0.1, 0.2, 0.8, 0.9])
        m = classification_metrics(y_true, y_score)
        assert m["ROC-AUC"] == pytest.approx(1.0)
        assert m["F1"] == pytest.approx(1.0)

    def test_reports_all_four_metrics(self):
        # CLAUDE.md §9: must report 4 metrics.
        rng = np.random.default_rng(0)
        y_true = rng.integers(0, 2, size=50)
        y_score = rng.random(size=50)
        m = classification_metrics(y_true, y_score)
        assert set(m) == {"ROC-AUC", "PR-AUC", "F1", "MCC"}

    def test_non_binary_raises(self):
        y_true = np.array([0, 1, 2])
        y_score = np.array([0.1, 0.5, 0.9])
        with pytest.raises(ValueError, match="binary"):
            classification_metrics(y_true, y_score)


class TestDispatch:
    def test_regression_dispatch(self):
        y = np.array([1.0, 2.0, 3.0])
        m = metrics_for_task("regression", y, y.copy())
        assert "RMSE" in m

    def test_classification_dispatch(self):
        m = metrics_for_task(
            "classification",
            np.array([0, 1, 0, 1]),
            np.array([0.1, 0.9, 0.2, 0.8]),
        )
        assert "ROC-AUC" in m

    def test_unknown_task_raises(self):
        with pytest.raises(ValueError, match="unknown task"):
            metrics_for_task("clustering", np.zeros(2), np.zeros(2))
