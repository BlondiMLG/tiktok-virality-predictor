import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    auc,
    average_precision_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_recall_curve,
    roc_curve,
)
from sklearn.model_selection import train_test_split

from dataset_normalize import DATASET_CHOICES
from dataloader import load_data
from preprocess import clean_data, create_target, feature_engineering
from model import (
    build_error_analysis,
    cross_validate_pipeline,
    estimator_for_cv_metrics,
    evaluate_pipeline,
    find_best_threshold,
    find_best_threshold_with_min_precision,
    fit_estimator_for_inference,
    metrics_at_threshold,
    predict_with_threshold,
    RANDOM_STATE,
    _can_stratify,
)

MODEL_ORDER = [
    "Logistic Regression",
    "Random Forest",
    "XGBoost",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TikTok virality prediction pipeline.")
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/tiktok_claims.csv",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="auto",
        choices=list(DATASET_CHOICES),
        help="TikTok CSV schema: auto-detect, or force tiktok_claims / tiktok_engagement.",
    )
    parser.add_argument("--quantile", type=float, default=0.80)
    parser.add_argument("--artifact-dir", type=str, default="artifacts")
    parser.add_argument(
        "--threshold-metric",
        type=str,
        default="f1",
        choices=["f1", "precision", "f1_with_min_precision"],
    )
    parser.add_argument("--min-precision", type=float, default=0.50)
    parser.add_argument("--min-recall", type=float, default=0.20)
    parser.add_argument(
        "--model-selection",
        type=str,
        default="val_f1",
        choices=["cv_f1", "val_f1", "val_pr_auc"],
        help="Metric for choosing the reported best model.",
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="RandomizedSearchCV on train (3 folds).",
    )
    parser.add_argument(
        "--tune-iter",
        type=int,
        default=18,
        help="Search trials per model (requires --tune).",
    )
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="Calibrate probabilities (CV on train).",
    )
    parser.add_argument(
        "--calibrate-method",
        type=str,
        default="sigmoid",
        choices=["sigmoid", "isotonic"],
    )
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if not (0.0 < args.quantile < 1.0):
        print(
            f"error: --quantile must be between 0 and 1 (exclusive), got {args.quantile!r}",
            file=sys.stderr,
        )
        raise SystemExit(2)
    if args.tune_iter < 1:
        print("error: --tune-iter must be >= 1", file=sys.stderr)
        raise SystemExit(2)


def save_artifacts(summary: pd.DataFrame, artifact_dir: Path, metadata: dict) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    summary_path = artifact_dir / "model_summary.csv"
    metadata_path = artifact_dir / "run_metadata.json"
    summary.to_csv(summary_path, index=True)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"\nSaved artifacts:")
    print(f"  - {summary_path}")
    print(f"  - {metadata_path}")


def save_model_plots(
    y_true: pd.Series,
    probs,
    preds,
    model_name: str,
    artifact_dir: Path,
) -> None:
    plots_dir = artifact_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    safe_name = model_name.lower().replace(" ", "_")

    precision_vals, recall_vals, _ = precision_recall_curve(y_true, probs)
    ap = average_precision_score(y_true, probs)
    plt.figure(figsize=(6, 5))
    plt.plot(recall_vals, precision_vals, label=f"AP={ap:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve - {model_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / f"{safe_name}_pr_curve.png", dpi=150)
    plt.close()

    fpr, tpr, _ = roc_curve(y_true, probs)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"ROC-AUC={roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {model_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / f"{safe_name}_roc_curve.png", dpi=150)
    plt.close()

    cm = confusion_matrix(y_true, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues", values_format="d")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.tight_layout()
    plt.savefig(plots_dir / f"{safe_name}_confusion_matrix.png", dpi=150)
    plt.close()


def main() -> None:
    args = parse_args()
    _validate_args(args)

    print("Loading and preprocessing data...")
    df, dataset_id = load_data(args.data_path, dataset=args.dataset)
    print(f"  Dataset schema: {dataset_id}")
    df = clean_data(df)
    if len(df) < 10:
        print(
            f"  Warning: only {len(df)} rows after clean; metrics will be very noisy.",
        )

    split_proxy = (df["video_view_count"] > df["video_view_count"].quantile(args.quantile)).astype(int)
    stratify_split = split_proxy if _can_stratify(split_proxy) else None

    df_tmp, df_test = train_test_split(
        df,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=stratify_split,
    )

    tmp_proxy = (df_tmp["video_view_count"] > df_tmp["video_view_count"].quantile(args.quantile)).astype(int)
    stratify_tmp = tmp_proxy if _can_stratify(tmp_proxy) else None
    df_train, df_val = train_test_split(
        df_tmp,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=stratify_tmp,
    )

    viral_threshold = df_train["video_view_count"].quantile(args.quantile)
    df_train = create_target(df_train, threshold=viral_threshold)
    df_val = create_target(df_val, threshold=viral_threshold)
    df_test = create_target(df_test, threshold=viral_threshold)

    df_train = feature_engineering(df_train)
    df_val = feature_engineering(df_val)
    df_test = feature_engineering(df_test)

    X_train, y_train = df_train.drop(columns=["viral"]), df_train["viral"]
    X_val, y_val = df_val.drop(columns=["viral"]), df_val["viral"]
    X_test, y_test = df_test.drop(columns=["viral"]), df_test["viral"]

    print("\nDataset splits:")
    print(f"  Train : {X_train.shape[0]:>5} rows")
    print(f"  Val   : {X_val.shape[0]:>5} rows")
    print(f"  Test  : {X_test.shape[0]:>5} rows")
    print(f"  Viral prevalence (train): {y_train.mean():.1%}")
    print(f"  Viral threshold (views): {int(viral_threshold):,}")

    all_results = []

    for name in MODEL_ORDER:
        print(f"\n{'=' * 52}")
        print(f"  Training: {name}")
        print(f"{'=' * 52}")

        final_est, search = fit_estimator_for_inference(
            name,
            X_train,
            y_train,
            tune=args.tune,
            tune_iter=args.tune_iter,
            calibrate=args.calibrate,
            calibrate_method=args.calibrate_method,
        )

        cv_base = estimator_for_cv_metrics(name, X_train, y_train, search)
        print("  Running 5-fold cross-validation on train set...")
        cv_metrics = cross_validate_pipeline(cv_base, X_train, y_train)
        print(f"  CV F1       : {cv_metrics['cv_f1_mean']:.3f} +/- {cv_metrics['cv_f1_std']:.3f}")
        print(f"  CV PR-AUC   : {cv_metrics['cv_pr_auc_mean']:.3f} +/- {cv_metrics['cv_pr_auc_std']:.3f}")
        print(f"  CV Precision: {cv_metrics['cv_precision_mean']:.3f}")
        print(f"  CV Recall   : {cv_metrics['cv_recall_mean']:.3f}")

        if args.threshold_metric == "f1_with_min_precision":
            best_thresh = find_best_threshold_with_min_precision(
                final_est,
                X_val,
                y_val,
                min_precision=args.min_precision,
                min_recall=args.min_recall,
            )
        else:
            best_thresh = find_best_threshold(final_est, X_val, y_val, metric=args.threshold_metric)
        print(f"  Best threshold (val, metric={args.threshold_metric}): {best_thresh:.2f}")

        val_metrics = metrics_at_threshold(final_est, X_val, y_val, best_thresh)
        val_ap = val_metrics["val_pr_auc"]
        val_ap_str = f"{val_ap:.3f}" if pd.notna(val_ap) else "nan"
        print(
            f"  Val (at threshold): F1={val_metrics['val_f1']:.3f} "
            f"P={val_metrics['val_precision']:.3f} R={val_metrics['val_recall']:.3f} "
            f"PR-AUC={val_ap_str}"
        )

        results = evaluate_pipeline(final_est, X_test, y_test, name, threshold=best_thresh)
        results.update(cv_metrics)
        results.update(val_metrics)
        all_results.append(results)

        probs, preds = predict_with_threshold(final_est, X_test, best_thresh)
        error_df = build_error_analysis(X_test, y_test, probs, preds)
        artifact_dir = Path(args.artifact_dir)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        error_df.to_csv(artifact_dir / f"error_analysis_{name.lower().replace(' ', '_')}.csv", index=False)
        save_model_plots(y_test, probs, preds, name, artifact_dir)

    print(f"\n{'=' * 52}")
    print("  FINAL MODEL COMPARISON")
    print(f"{'=' * 52}")

    selection_map = {
        "cv_f1": "cv_f1_mean",
        "val_f1": "val_f1",
        "val_pr_auc": "val_pr_auc",
    }
    selection_col = selection_map[args.model_selection]

    summary = pd.DataFrame(all_results).set_index("model")
    summary = summary[
        [
            "threshold",
            "val_f1",
            "val_precision",
            "val_recall",
            "val_pr_auc",
            "accuracy",
            "precision",
            "recall",
            "f1",
            "pr_auc",
            "roc_auc",
            "brier",
            "pred_positive_rate",
            "cv_f1_mean",
            "cv_f1_std",
            "cv_pr_auc_mean",
            "cv_pr_auc_std",
            "cv_precision_mean",
            "cv_recall_mean",
        ]
    ].sort_values(selection_col, ascending=False, na_position="last")
    print(summary.to_string())

    rank_scores = summary[selection_col].fillna(-1.0)
    best_model = rank_scores.idxmax()
    best_score = summary.loc[best_model, selection_col]
    best_score_str = f"{best_score:.3f}" if pd.notna(best_score) else "nan"
    print(
        f"\n  -> Best model by {args.model_selection} ({selection_col}): {best_model} "
        f"({best_score_str})"
    )

    save_artifacts(
        summary=summary,
        artifact_dir=Path(args.artifact_dir),
        metadata={
            "data_path": args.data_path,
            "dataset_id": dataset_id,
            "dataset_arg": args.dataset,
            "quantile": args.quantile,
            "threshold_metric": args.threshold_metric,
            "min_precision": args.min_precision,
            "min_recall": args.min_recall,
            "model_selection": args.model_selection,
            "tune": args.tune,
            "tune_iter": args.tune_iter,
            "calibrate": args.calibrate,
            "calibrate_method": args.calibrate_method,
            "train_rows": int(X_train.shape[0]),
            "val_rows": int(X_val.shape[0]),
            "test_rows": int(X_test.shape[0]),
            "viral_threshold": float(viral_threshold),
            "best_model": best_model,
            "selection_metric_column": selection_col,
        },
    )


if __name__ == "__main__":
    main()
