import pandas as pd
import numpy as np

TEXT_COL = "video_transcription_text"
CATEGORICAL_COLS = ["claim_status", "verified_status", "author_ban_status"]


def create_target(
    df: pd.DataFrame,
    threshold: float | None = None,
    quantile: float = 0.80,
) -> pd.DataFrame:
    df = df.copy()
    if "video_view_count" not in df.columns:
        raise ValueError("Missing required column: 'video_view_count'")

    target_threshold = threshold
    if target_threshold is None:
        target_threshold = df["video_view_count"].quantile(quantile)

    df["viral"] = (df["video_view_count"] > target_threshold).astype(int)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df = df.drop(columns=["#", "video_id", "id"], errors="ignore")

    if "video_view_count" not in df.columns:
        raise ValueError("Missing required column: 'video_view_count'")
    df["video_view_count"] = pd.to_numeric(df["video_view_count"], errors="coerce")
    df = df.dropna(subset=["video_view_count"])

    if "video_duration_sec" in df.columns:
        df["video_duration_sec"] = pd.to_numeric(
            df["video_duration_sec"], errors="coerce"
        )
        median_duration = df["video_duration_sec"].median()
        df["video_duration_sec"] = df["video_duration_sec"].fillna(median_duration)
    else:
        df["video_duration_sec"] = 0.0

    for col in CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.strip()
                .str.lower()
                .replace({"nan": "unknown", "": "unknown"})
            )

    return df

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    text = (
        df[TEXT_COL].fillna("")
        if TEXT_COL in df.columns
        else pd.Series("", index=df.index, dtype=str)
    )
    text = text.astype(str)
    df[TEXT_COL] = text

    df["has_text"] = (text.str.len() > 0).astype(int)
    df["text_length"] = text.str.len()
    df["word_count"] = text.str.split().str.len().fillna(0).astype(int)
    df["exclamation_count"] = text.str.count("!")
    df["question_count"] = text.str.count(r"\?")
    df["digit_ratio"] = text.str.count(r"\d") / (df["text_length"] + 1)

    df["uppercase_ratio"] = text.str.count(r"[A-Z]") / (df["text_length"] + 1)

    _words = text.str.split()
    _wlen = _words.explode().str.len()
    avg_word_length = _wlen.groupby(level=0).mean().reindex(df.index)
    df["avg_word_length"] = avg_word_length.fillna(0.0).astype(float)

    df["duration_bucket"] = pd.cut(
        df["video_duration_sec"],
        bins=[0, 15, 30, 61, np.inf],
        labels=[0, 1, 2, 3],
        right=True,
        include_lowest=True,
    )
    df["duration_bucket"] = df["duration_bucket"].cat.add_categories([-1]).fillna(-1).astype(int)

    df["text_density"] = df["word_count"] / (df["video_duration_sec"] + 1)
    df["hook_3_words"] = (
        text.str.split().str[:3].str.join(" ").str.strip().replace({"": "no_hook"})
    )

    leakage_cols = [
        "video_view_count",
        "video_like_count",
        "video_share_count",
        "video_download_count",
        "video_comment_count",
    ]
    df = df.drop(columns=leakage_cols, errors="ignore")

    return df