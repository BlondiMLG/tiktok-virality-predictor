import pandas as pd

def create_target(df):
    df = df.copy()
    threshold = df["video_view_count"].quantile(0.8)
    df["viral"] = (df["video_view_count"] > threshold).astype(int)
    return df

def clean_data(df):
    df = df.copy()
    df = df.drop(columns=["#","video_id"], errors="ignore")
    df = df.dropna(subset=["video_view_count"])
    df["video_duration_sec"] = pd.to_numeric(df["video_duration_sec"], errors="coerce").fillna(0)
    return df

def feature_engineering(df):
    df = df.copy()
    df["video_transcription_text"] = df["video_transcription_text"].fillna("")
    df["text_length"] = df["video_transcription_text"].str.len()
    df["word_count"] = df["video_transcription_text"].str.split().str.len()
    df["exclamation_count"] = df["video_transcription_text"].str.count("!")
    df["question_count"] = df["video_transcription_text"].str.count(r"\?")
    df["uppercase_ratio"] = df["video_transcription_text"].str.count(r"[A-Z]") / (df["text_length"] + 1)
    df["first_3_words"] = df["video_transcription_text"].str.split().str[:3].str.join(" ")

    columns_to_drop = [
        "video_view_count",
        "video_like_count",
        "video_share_count",
        "video_download_count",
        "video_comment_count"
    ]
    df = df.drop(columns=columns_to_drop, errors="ignore")

    categorical_cols = ["verified_status", "author_ban_status", "claim_status", "first_3_words"]
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    return df