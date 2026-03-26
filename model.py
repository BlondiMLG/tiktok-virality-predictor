from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, f1_score, average_precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pandas as pd

def split_data(df):
    X = df.drop(columns=["viral"])
    y = df["viral"]
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

def get_preprocessor():
    return ColumnTransformer(
        transformers=[
            ("tfidf", TfidfVectorizer(max_features=1000, stop_words="english", ngram_range=(1, 3)), "video_transcription_text")
        ],
        remainder="passthrough"
    )

def build_logreg_pipeline(y_train):
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    weight = (neg / pos) * 0.4
    
    model = LogisticRegression(max_iter=2000, class_weight={0: 1, 1: weight}, solver="liblinear")
    return Pipeline(steps=[("preprocessor", get_preprocessor()), ("classifier", model)])

def build_xgb_pipeline(y_train):
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    scale_pos_weight = (neg / pos) * 0.4

    model = XGBClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.1,
        scale_pos_weight=scale_pos_weight, eval_metric="logloss"
    )
    return Pipeline(steps=[("preprocessor", get_preprocessor()), ("classifier", model)])

def evaluate_pipeline(pipeline, X_test, y_test, model_name="Model", threshold=0.5):
    print(f"\n--- {model_name} (Threshold: {threshold:.2f}) ---")

    probs = pipeline.predict_proba(X_test)[:, 1]
    preds = (probs > threshold).astype(int)

    print("Predicted class counts:")
    print(pd.Series(preds).value_counts().sort_index())

    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    pr_auc = average_precision_score(y_test, probs)

    print("Accuracy:", round(acc, 3))
    print("F1 Score:", round(f1, 3))
    print("PR-AUC:", round(pr_auc, 3))

    print("\nClassification Report:")
    print(classification_report(y_test, preds))

    return {
        "accuracy": round(acc, 3),
        "f1": round(f1, 3),
        "pr_auc": round(pr_auc, 3)
    }