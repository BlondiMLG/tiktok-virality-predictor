import unittest
import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import f1_score, precision_score

from model import (
    RANDOM_STATE,
    _can_stratify,
    build_logreg_pipeline,
    cross_validate_pipeline,
    evaluate_pipeline,
    find_best_threshold,
    find_best_threshold_with_min_precision,
    metrics_at_threshold,
    predict_with_threshold,
)


def _f1(y_true: pd.Series, y_pred) -> float:
    return float(f1_score(y_true, y_pred, zero_division=0))


class _FixedProbaEstimator(BaseEstimator):
    """Estimator that returns constant or per-row positive-class probabilities."""

    def __init__(self, p: float | np.ndarray) -> None:
        self.p = p

    def fit(self, X, y=None):
        return self

    def predict_proba(self, X):
        n = len(X)
        if isinstance(self.p, (int, float)):
            pv = np.full(n, float(self.p), dtype=float)
        else:
            pv = np.asarray(self.p, dtype=float).reshape(-1)
            if pv.size == 1:
                pv = np.full(n, float(pv[0]), dtype=float)
            elif pv.size != n:
                raise ValueError("p length must match X or be scalar")
        return np.column_stack([1.0 - pv, pv])


class TestCanStratify(unittest.TestCase):
    def test_false_when_single_class(self) -> None:
        y = pd.Series([0, 0, 0])
        self.assertFalse(_can_stratify(y))

    def test_true_when_two_classes_min_two_each(self) -> None:
        y = pd.Series([0, 0, 1, 1])
        self.assertTrue(_can_stratify(y))

    def test_false_when_second_class_too_small(self) -> None:
        y = pd.Series([0, 0, 0, 1])
        self.assertFalse(_can_stratify(y, min_per_class=2))


class TestThresholdSearch(unittest.TestCase):
    def test_single_class_validation_returns_default_threshold(self) -> None:
        est = _FixedProbaEstimator(0.7)
        X_val = pd.DataFrame({"a": [1, 2]})
        y_val = pd.Series([0, 0])
        t = find_best_threshold(est, X_val, y_val, metric="f1")
        self.assertEqual(t, 0.5)

    def test_find_best_threshold_f1(self) -> None:
        est = _FixedProbaEstimator(
            np.array([0.95, 0.95, 0.05, 0.05]),
        )
        X_val = pd.DataFrame({"x": range(4)})
        y_val = pd.Series([1, 1, 0, 0])
        t = find_best_threshold(est, X_val, y_val, metric="f1")
        self.assertGreaterEqual(t, 0.10)
        self.assertLessEqual(t, 0.91)
        preds = (est.predict_proba(X_val)[:, 1] > t).astype(int)
        self.assertEqual(_f1(y_val, preds), 1.0)

    def test_find_best_threshold_precision(self) -> None:
        est = _FixedProbaEstimator(
            np.array([0.99, 0.99, 0.4, 0.01]),
        )
        X_val = pd.DataFrame({"x": range(4)})
        y_val = pd.Series([1, 1, 0, 0])
        t = find_best_threshold(est, X_val, y_val, metric="precision")
        preds = (est.predict_proba(X_val)[:, 1] > t).astype(int)
        p = precision_score(y_val, preds, zero_division=0)
        self.assertGreaterEqual(p, 0.99)

    def test_min_precision_fallback_when_impossible(self) -> None:
        est = _FixedProbaEstimator(0.6)
        X_val = pd.DataFrame({"x": range(4)})
        y_val = pd.Series([1, 0, 1, 0])
        t = find_best_threshold_with_min_precision(
            est,
            X_val,
            y_val,
            min_precision=0.99,
            min_recall=0.99,
        )
        self.assertGreaterEqual(t, 0.10)


class TestMetricsAndPredict(unittest.TestCase):
    def test_metrics_at_threshold_keys_and_ranges(self) -> None:
        est = _FixedProbaEstimator(
            np.array([0.9, 0.9, 0.1, 0.1]),
        )
        X_val = pd.DataFrame({"x": range(4)})
        y_val = pd.Series([1, 1, 0, 0])
        m = metrics_at_threshold(est, X_val, y_val, threshold=0.5)
        self.assertIn("val_f1", m)
        self.assertIn("val_precision", m)
        self.assertIn("val_recall", m)
        self.assertIn("val_pr_auc", m)
        self.assertEqual(m["val_f1"], 1.0)

    def test_predict_with_threshold(self) -> None:
        est = _FixedProbaEstimator(0.77)
        X = pd.DataFrame({"x": [1, 2, 3]})
        probs, preds = predict_with_threshold(est, X, threshold=0.5)
        np.testing.assert_allclose(probs, 0.77)
        np.testing.assert_array_equal(preds, [1, 1, 1])


class TestPipelineIntegration(unittest.TestCase):
    def test_cross_validate_pipeline_returns_finite_metrics(self) -> None:
        rng = np.random.default_rng(RANDOM_STATE)
        n = 80
        X_train = pd.DataFrame(
            {
                "video_transcription_text": [
                    f"caption words repeat {i % 7}" for i in range(n)
                ],
                "video_duration_sec": rng.uniform(5.0, 120.0, n),
                "claim_status": rng.choice(["claim", "opinion", "unknown"], n),
            }
        )
        y_train = pd.Series(np.zeros(n, dtype=int))
        y_train.iloc[n // 2 :] = 1

        est = build_logreg_pipeline(X_train, y_train)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = cross_validate_pipeline(est, X_train, y_train, n_splits=3)
        self.assertFalse(np.isnan(out["cv_f1_mean"]))
        self.assertFalse(np.isnan(out["cv_pr_auc_mean"]))

    def test_evaluate_pipeline_returns_expected_keys(self) -> None:
        rng = np.random.default_rng(1)
        n = 60
        X_train = pd.DataFrame(
            {
                "video_transcription_text": ["hello tiktok"] * n,
                "video_duration_sec": rng.uniform(10.0, 90.0, n),
                "claim_status": ["claim"] * (n // 2) + ["opinion"] * (n - n // 2),
            }
        )
        y_train = pd.Series([0] * (n // 2) + [1] * (n - n // 2))
        test_idx = list(range(8)) + list(range(n // 2, n // 2 + 8))
        X_test = X_train.iloc[test_idx].reset_index(drop=True)
        y_test = y_train.iloc[test_idx].reset_index(drop=True)

        pipe = build_logreg_pipeline(X_train, y_train)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pipe.fit(X_train, y_train)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = evaluate_pipeline(
                pipe, X_test, y_test, "Logistic Regression", threshold=0.5
            )
        for key in (
            "model",
            "threshold",
            "accuracy",
            "precision",
            "recall",
            "f1",
            "pr_auc",
            "roc_auc",
            "pred_positive_rate",
            "brier",
        ):
            self.assertIn(key, result)


if __name__ == "__main__":
    unittest.main()
