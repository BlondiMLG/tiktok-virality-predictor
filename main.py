from dataloader import load_data
from preprocess import create_target, clean_data, feature_engineering
from model import split_data, build_logreg_pipeline, build_xgb_pipeline, evaluate_pipeline
import numpy as np
from sklearn.metrics import f1_score

def find_best_threshold(pipeline, X_test, y_test):
    probs = pipeline.predict_proba(X_test)[:, 1]
    best_threshold = 0.5
    best_f1 = 0
    for thresh in np.arange(0.3, 0.95, 0.05):
        preds = (probs > thresh).astype(int)
        current_f1 = f1_score(y_test, preds)
        if current_f1 > best_f1:
            best_f1 = current_f1
            best_threshold = thresh
    return best_threshold

def main():
    df = load_data()
    df = clean_data(df)
    df = create_target(df)
    df = feature_engineering(df)

    X_train, X_test, y_train, y_test = split_data(df)

    log_pipeline = build_logreg_pipeline(y_train)
    log_pipeline.fit(X_train, y_train)
    best_log_thresh = find_best_threshold(log_pipeline, X_test, y_test)
    log_results = evaluate_pipeline(log_pipeline, X_test, y_test, "Logistic Regression", threshold=best_log_thresh)

    xgb_pipeline = build_xgb_pipeline(y_train)
    xgb_pipeline.fit(X_train, y_train)
    best_xgb_thresh = find_best_threshold(xgb_pipeline, X_test, y_test)
    xgb_results = evaluate_pipeline(xgb_pipeline, X_test, y_test, "XGBoost", threshold=best_xgb_thresh)

    print("\nFinal Model Comparison:")
    for model_name, metrics in {"Logistic Regression": log_results, "XGBoost": xgb_results}.items():
        print(f"{model_name}: {metrics}")

if __name__ == "__main__":
    main()