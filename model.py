import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint, uniform

from sklearn.base import BaseEstimator, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_validate
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    average_precision_score,
    roc_auc_score,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

TEXT_COL = "video_transcription_text"
RANDOM_STATE = 42

def _can_stratify(y: pd.Series, min_per_class: int = 2) -> bool:
    counts = y.value_counts(dropna=False)
    return len(counts) > 1 and counts.min() >= min_per_class

def _get_numeric_cols(X: pd.DataFrame) -> list[str]:
    return [c for c in X.select_dtypes(include=["number", "bool"]).columns]


def _get_categorical_cols(X: pd.DataFrame) -> list[str]:
    return [c for c in X.select_dtypes(include=["object", "category"]).columns if c != TEXT_COL]


def get_preprocessor(X_train: pd.DataFrame) -> ColumnTransformer:
    numeric_cols = _get_numeric_cols(X_train)
    categorical_cols = _get_categorical_cols(X_train)
    n_samples = len(X_train)
    # min_df=2 breaks on tiny train sets (no term appears twice)
    min_df_tfidf = 2 if n_samples >= 200 else 1
    transformers = [
        (
            "tfidf",
            TfidfVectorizer(
                max_features=4000,
                stop_words="english",
                ngram_range=(1, 3),
                sublinear_tf=True,
                min_df=min_df_tfidf,
            ),
            TEXT_COL,
        ),
        (
            "num",
            StandardScaler(),
            numeric_cols,
        ),
    ]

    if categorical_cols:
        transformers.append(
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_cols,
            )
        )

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )

def _class_weight(y_train: pd.Series, scale: float = 0.4) -> float:
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    if pos == 0:
        return 1.0
    return (neg / pos) * scale

def build_logreg_pipeline(X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    model = LogisticRegression(
        max_iter=3000,
        class_weight="balanced",
        solver="liblinear",
        C=0.8,
    )
    return Pipeline([
        ("preprocessor", get_preprocessor(X_train)),
        ("classifier", model),
    ])


def build_xgb_pipeline(X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    model = XGBClassifier(
        n_estimators=350,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        scale_pos_weight=_class_weight(y_train),
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        verbosity=0,
    )
    return Pipeline([
        ("preprocessor", get_preprocessor(X_train)),
        ("classifier", model),
    ])


def build_rf_pipeline(X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    model = RandomForestClassifier(
        n_estimators=350,
        max_depth=10,
        min_samples_leaf=3,
        class_weight="balanced_subsample",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    return Pipeline([
        ("preprocessor", get_preprocessor(X_train)),
        ("classifier", model),
    ])

def find_best_threshold(
    estimator: BaseEstimator,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    metric: str = "f1",
) -> float:

    if y_val.nunique() < 2:
        return 0.5

    probs = estimator.predict_proba(X_val)[:, 1]
    best_thresh, best_score = 0.5, 0.0

    for thresh in np.arange(0.10, 0.91, 0.01):
        preds = (probs > thresh).astype(int)
        if metric == "f1":
            score = f1_score(y_val, preds, zero_division=0)
        elif metric == "precision":
            score = precision_score(y_val, preds, zero_division=0)
        else:
            raise ValueError(f"Unknown metric: {metric}")

        if score > best_score:
            best_score, best_thresh = score, thresh

    return float(best_thresh)


def find_best_threshold_with_min_precision(
    estimator: BaseEstimator,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    min_precision: float = 0.50,
    min_recall: float = 0.20,
) -> float:
    if y_val.nunique() < 2:
        return 0.5

    probs = estimator.predict_proba(X_val)[:, 1]
    best_thresh, best_f1 = 0.5, -1.0

    for thresh in np.arange(0.10, 0.91, 0.01):
        preds = (probs > thresh).astype(int)
        precision = precision_score(y_val, preds, zero_division=0)
        recall = recall_score(y_val, preds, zero_division=0)
        if precision < min_precision:
            continue
        if recall < min_recall:
            continue
        score = f1_score(y_val, preds, zero_division=0)
        if score > best_f1:
            best_f1, best_thresh = score, thresh

    if best_f1 < 0:
        return find_best_threshold(estimator, X_val, y_val, metric="f1")
    return float(best_thresh)


def predict_with_threshold(
    estimator: BaseEstimator,
    X: pd.DataFrame,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    probs = estimator.predict_proba(X)[:, 1]
    preds = (probs > threshold).astype(int)
    return probs, preds


def metrics_at_threshold(
    estimator: BaseEstimator,
    X: pd.DataFrame,
    y: pd.Series,
    threshold: float,
) -> dict:
    probs = estimator.predict_proba(X)[:, 1]
    preds = (probs > threshold).astype(int)
    return {
        "val_f1": round(f1_score(y, preds, zero_division=0), 3),
        "val_precision": round(precision_score(y, preds, zero_division=0), 3),
        "val_recall": round(recall_score(y, preds, zero_division=0), 3),
        "val_pr_auc": round(float(average_precision_score(y, probs)), 3)
        if y.nunique() > 1
        else float("nan"),
    }


def _stratified_cv(n_splits: int = 3) -> StratifiedKFold:
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)


def tune_logreg_search(X_train: pd.DataFrame, y_train: pd.Series, n_iter: int) -> RandomizedSearchCV:
    pipe = build_logreg_pipeline(X_train, y_train)
    param_dist = {
        "classifier__C": loguniform(1e-2, 10),
        "classifier__solver": ["liblinear", "saga"],
    }
    search = RandomizedSearchCV(
        pipe,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring="f1",
        cv=_stratified_cv(3),
        random_state=RANDOM_STATE,
        n_jobs=-1,
        refit=True,
    )
    search.fit(X_train, y_train)
    return search


def tune_rf_search(X_train: pd.DataFrame, y_train: pd.Series, n_iter: int) -> RandomizedSearchCV:
    pipe = build_rf_pipeline(X_train, y_train)
    param_dist = {
        "classifier__n_estimators": randint(100, 501),
        "classifier__max_depth": [5, 8, 10, 12, 15, None],
        "classifier__min_samples_leaf": randint(1, 8),
    }
    search = RandomizedSearchCV(
        pipe,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring="f1",
        cv=_stratified_cv(3),
        random_state=RANDOM_STATE,
        n_jobs=-1,
        refit=True,
    )
    search.fit(X_train, y_train)
    return search


def tune_xgb_search(X_train: pd.DataFrame, y_train: pd.Series, n_iter: int) -> RandomizedSearchCV:
    pipe = build_xgb_pipeline(X_train, y_train)
    param_dist = {
        "classifier__n_estimators": randint(150, 501),
        "classifier__max_depth": randint(3, 9),
        "classifier__learning_rate": loguniform(0.02, 0.2),
        "classifier__subsample": uniform(0.7, 0.29),
        "classifier__colsample_bytree": uniform(0.7, 0.29),
    }
    search = RandomizedSearchCV(
        pipe,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring="f1",
        cv=_stratified_cv(3),
        random_state=RANDOM_STATE,
        n_jobs=-1,
        refit=True,
    )
    search.fit(X_train, y_train)
    return search


def run_hyperparam_search(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_iter: int,
) -> RandomizedSearchCV:
    if model_name == "Logistic Regression":
        return tune_logreg_search(X_train, y_train, n_iter)
    if model_name == "Random Forest":
        return tune_rf_search(X_train, y_train, n_iter)
    if model_name == "XGBoost":
        return tune_xgb_search(X_train, y_train, n_iter)
    raise ValueError(f"Unknown model for tuning: {model_name}")


def fit_estimator_for_inference(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    tune: bool,
    tune_iter: int,
    calibrate: bool,
    calibrate_method: str,
) -> tuple[BaseEstimator, RandomizedSearchCV | None]:
    search: RandomizedSearchCV | None = None

    if tune:
        print(f"  Randomized hyperparameter search (n_iter={tune_iter})...")
        search = run_hyperparam_search(model_name, X_train, y_train, tune_iter)
        if calibrate:
            print(f"  Probability calibration ({calibrate_method}, CV on train)...")
            calibrator = CalibratedClassifierCV(
                estimator=clone(search.best_estimator_),
                method=calibrate_method,
                cv=_stratified_cv(3),
            )
            final_est = calibrator.fit(X_train, y_train)
        else:
            final_est = search.best_estimator_
        return final_est, search

    base = build_model_by_name(model_name, X_train, y_train)
    if calibrate:
        print(f"  Probability calibration ({calibrate_method}, CV on train)...")
        calibrator = CalibratedClassifierCV(
            estimator=clone(base),
            method=calibrate_method,
            cv=_stratified_cv(3),
        )
        final_est = calibrator.fit(X_train, y_train)
    else:
        final_est = base.fit(X_train, y_train)

    return final_est, search


def build_model_by_name(model_name: str, X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    if model_name == "Logistic Regression":
        return build_logreg_pipeline(X_train, y_train)
    if model_name == "Random Forest":
        return build_rf_pipeline(X_train, y_train)
    if model_name == "XGBoost":
        return build_xgb_pipeline(X_train, y_train)
    raise ValueError(f"Unknown model: {model_name}")


def estimator_for_cv_metrics(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    search: RandomizedSearchCV | None,
) -> BaseEstimator:
    if search is not None:
        return clone(search.best_estimator_)
    return build_model_by_name(model_name, X_train, y_train)


def build_error_analysis(
    X_test: pd.DataFrame,
    y_test: pd.Series,
    probs: np.ndarray,
    preds: np.ndarray,
) -> pd.DataFrame:
    analysis = X_test.copy()
    analysis["true_label"] = y_test.values
    analysis["pred_label"] = preds
    analysis["pred_proba"] = probs
    analysis["is_error"] = (analysis["true_label"] != analysis["pred_label"]).astype(int)
    analysis["error_type"] = np.where(
        (analysis["true_label"] == 1) & (analysis["pred_label"] == 0),
        "false_negative",
        np.where(
            (analysis["true_label"] == 0) & (analysis["pred_label"] == 1),
            "false_positive",
            "correct",
        ),
    )
    return analysis


def cross_validate_pipeline(
    estimator: BaseEstimator,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_splits: int = 5,
) -> dict:
    nan_block = {
        "cv_f1_mean": np.nan,
        "cv_f1_std": np.nan,
        "cv_pr_auc_mean": np.nan,
        "cv_pr_auc_std": np.nan,
        "cv_precision_mean": np.nan,
        "cv_recall_mean": np.nan,
    }

    class_counts = y_train.value_counts()
    if len(class_counts) < 2:
        return nan_block

    max_valid_splits = int(class_counts.min())
    effective_splits = min(n_splits, max_valid_splits)
    if effective_splits < 2:
        return nan_block

    cv = StratifiedKFold(n_splits=effective_splits, shuffle=True, random_state=RANDOM_STATE)
    results = cross_validate(
        clone(estimator),
        X_train,
        y_train,
        cv=cv,
        scoring={
            "f1": "f1",
            "pr_auc": "average_precision",
            "precision": "precision",
            "recall": "recall",
        },
        return_train_score=False,
        n_jobs=-1,
    )
    return {
        "cv_f1_mean": round(results["test_f1"].mean(), 3),
        "cv_f1_std": round(results["test_f1"].std(), 3),
        "cv_pr_auc_mean": round(results["test_pr_auc"].mean(), 3),
        "cv_pr_auc_std": round(results["test_pr_auc"].std(), 3),
        "cv_precision_mean": round(results["test_precision"].mean(), 3),
        "cv_recall_mean": round(results["test_recall"].mean(), 3),
    }

def evaluate_pipeline(
    estimator: BaseEstimator,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str = "Model",
    threshold: float = 0.5,
) -> dict:

    print(f"\n{'-'*50}")
    print(f"  {model_name}  (threshold = {threshold:.2f})")
    print(f"{'-'*50}")

    probs, preds = predict_with_threshold(estimator, X_test, threshold)

    print("Predicted class distribution:")
    print(pd.Series(preds).value_counts().sort_index().to_string())

    acc    = accuracy_score(y_test, preds)
    f1     = f1_score(y_test, preds, zero_division=0)
    precision = precision_score(y_test, preds, zero_division=0)
    recall = recall_score(y_test, preds, zero_division=0)
    pr_auc = (
        float(average_precision_score(y_test, probs))
        if y_test.nunique() > 1
        else float("nan")
    )
    roc    = roc_auc_score(y_test, probs) if y_test.nunique() > 1 else np.nan

    print(f"\nAccuracy : {acc:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall   : {recall:.3f}")
    print(f"F1 Score : {f1:.3f}")
    print(f"PR-AUC   : {pr_auc:.3f}")
    print(f"ROC-AUC  : {roc:.3f}")
    print("\nClassification Report:")
    print(classification_report(y_test, preds, zero_division=0))

    return {
        "model":    model_name,
        "threshold": round(threshold, 2),
        "accuracy": round(acc, 3),
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1":       round(f1, 3),
        "pr_auc":   round(pr_auc, 3),
        "roc_auc":  round(roc, 3),
        "pred_positive_rate": round(float(preds.mean()), 3),
        "brier": round(float(np.mean((probs - y_test.values) ** 2)), 4),
    }