from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd

DATASET_TIKTOK_CLAIMS = "tiktok_claims"
DATASET_TIKTOK_ENGAGEMENT = "tiktok_engagement"

DATASET_CHOICES = (
    "auto",
    DATASET_TIKTOK_CLAIMS,
    DATASET_TIKTOK_ENGAGEMENT,
)


def _lower_index(df: pd.DataFrame) -> dict[str, str]:
    return {str(c).strip().lower(): c for c in df.columns}


def _col(df: pd.DataFrame, *names: str) -> pd.Series | None:
    idx = _lower_index(df)
    for n in names:
        if n.lower() in idx:
            return df[idx[n.lower()]]
    return None


def _first_text_column(df: pd.DataFrame) -> pd.Series:
    for key in (
        "video_transcription_text",
        "video_transcription",
        "transcript",
        "transcription",
        "video_desc",
        "description",
        "desc",
        "text",
        "caption",
        "title",
        "video_title",
        "sticker_text",
        "stickertext",
    ):
        s = _col(df, key)
        if s is not None:
            return s.fillna("").astype(str)
    return pd.Series("", index=df.index, dtype=str)


def _coalesce_numeric_by_column_names(
    df: pd.DataFrame, names: tuple[str, ...]
) -> pd.Series | None:
    """
    Merge several possible play/view or metric columns in priority order (left wins
    for non-null; later columns fill remaining NaNs). Handles exports that include
    both `plays` and `views` where one is often empty.
    """
    parts: list[pd.Series] = []
    for n in names:
        s = _col(df, n)
        if s is not None:
            parts.append(pd.to_numeric(s, errors="coerce"))
    if not parts:
        return None
    out = parts[0]
    for p in parts[1:]:
        out = out.fillna(p)
    return out.reindex(df.index)


def _concat_text_columns(df: pd.DataFrame, keys: tuple[str, ...]) -> pd.Series:
    parts: list[pd.Series] = []
    for k in keys:
        s = _col(df, k)
        if s is not None:
            parts.append(s.fillna("").astype(str))
    if not parts:
        return pd.Series("", index=df.index, dtype=str)
    out = parts[0]
    for p in parts[1:]:
        out = out + " " + p
    return out.str.replace(r"\s+", " ", regex=True).str.strip()


def parse_duration_seconds(val: Any) -> float:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return float("nan")
    if isinstance(val, (int, np.integer)):
        return float(val)
    if isinstance(val, (float, np.floating)):
        return float(val)
    s = str(val).strip()
    if not s or s.lower() in {"nan", "none"}:
        return float("nan")
    try:
        return float(s)
    except ValueError:
        pass
    if ":" in s:
        parts = s.split(":")
        try:
            parts_f = [float(p) for p in parts]
            if len(parts_f) == 2:
                return parts_f[0] * 60.0 + parts_f[1]
            if len(parts_f) == 3:
                return parts_f[0] * 3600.0 + parts_f[1] * 60.0 + parts_f[2]
        except ValueError:
            pass
    m = re.search(r"(\d+(?:\.\d+)?)", s)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    return float("nan")


def _series_duration_to_seconds(s: pd.Series) -> pd.Series:
    out = s.map(parse_duration_seconds)
    return pd.to_numeric(out, errors="coerce")


def infer_dataset_id(df: pd.DataFrame) -> str:
    c = set(_lower_index(df).keys())
    if "video_view_count" in c:
        return DATASET_TIKTOK_CLAIMS
    engagement_view_cols = {
        "play_count",
        "playcount",
        "n_plays",
        "n_views",
        "views",
        "view_count",
        "video_playcount",  # e.g. video_playCount
        "plays",
    }
    if c & engagement_view_cols:
        return DATASET_TIKTOK_ENGAGEMENT
    raise ValueError(
        "Could not infer --dataset. Pass one of: "
        f"{DATASET_TIKTOK_CLAIMS}, {DATASET_TIKTOK_ENGAGEMENT}. "
        f"Columns seen: {list(df.columns)[:25]}"
    )


def _normalize_tiktok_claims_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure common preprocess/model column names when source CSVs use variants
    (Kaggle-claims style already matches; scrapes may not).
    """
    out = df.copy()
    idx = _lower_index(out)
    if "video_duration_sec" not in idx:
        dur = _col(
            out,
            "video_dur",
            "vid_dur",
            "vid_duration",
            "video_duration",
            "duration",
            "video_length",
            "length",
            "duration_sec",
        )
        if dur is not None:
            out["video_duration_sec"] = _series_duration_to_seconds(dur).reindex(
                out.index
            )

    if "video_transcription_text" not in idx:
        t = _col(
            out,
            "video_desc",
            "description",
            "transcription",
            "transcript",
            "text",
            "caption",
            "title",
            "video_title",
        )
        if t is not None:
            out["video_transcription_text"] = t.fillna("").astype(str)

    for canonical, alts in (
        ("video_like_count", ("digg_count", "n_likes", "like_count", "likes", "hearts")),
        ("video_comment_count", ("comment_count", "n_comments", "comments")),
        ("video_share_count", ("share_count", "n_shares", "shares")),
    ):
        if canonical not in _lower_index(out):
            s = _col(out, *alts)
            if s is not None:
                out[canonical] = pd.to_numeric(s, errors="coerce")
    return out


def normalize_dataset(df: pd.DataFrame, dataset_id: str) -> pd.DataFrame:
    if dataset_id == DATASET_TIKTOK_CLAIMS:
        if "video_view_count" not in _lower_index(df):
            raise ValueError(
                f"{DATASET_TIKTOK_CLAIMS} requires column 'video_view_count'. "
                f"Got: {list(df.columns)[:30]}"
            )
        return _normalize_tiktok_claims_columns(df)

    if dataset_id == DATASET_TIKTOK_ENGAGEMENT:
        return _normalize_tiktok_engagement(df)

    raise ValueError(f"Unknown dataset_id: {dataset_id}")


def _normalize_tiktok_engagement(df: pd.DataFrame) -> pd.DataFrame:
    # Prefer explicit play_* / n_plays before generic `views` so merged scrapes
    # (plays + often-empty views) keep the correct signal.
    play = _coalesce_numeric_by_column_names(
        df,
        (
            "play_count",
            "playcount",
            "n_plays",
            "plays",
            "n_views",
            "view_count",
            "views",
            "video_playcount",
        ),
    )
    if play is None:
        raise ValueError(
            "tiktok_engagement schema requires a play/view column "
            "(e.g. play_count, n_plays, plays, views)."
        )

    text = _concat_text_columns(
        df,
        (
            "video_transcription_text",
            "video_transcription",
            "transcript",
            "description",
            "video_desc",
            "desc",
            "text",
            "caption",
            "title",
            "video_title",
        ),
    )
    if text.str.len().max() == 0:
        text = _first_text_column(df)

    likes = _coalesce_numeric_by_column_names(
        df,
        (
            "digg_count",
            "diggcount",
            "like_count",
            "likes",
            "n_likes",
            "heart_count",
            "hearts",
            "video_like_count",
        ),
    )
    shares = _col(
        df,
        "share_count",
        "shares",
        "n_shares",
        "video_sharecount",
        "video_share_count",
    )
    comments = _col(
        df,
        "comment_count",
        "comments",
        "n_comments",
        "video_commentcount",
        "video_comment_count",
    )
    downloads = _col(df, "download_count", "video_download_count", "n_downloads")

    dur_raw = _col(
        df,
        "video_duration_sec",
        "duration_sec",
        "duration",
        "video_duration",
        "video_dur",
        "vid_dur",
        "vid_duration",
        "video_length",
    )
    if dur_raw is None:
        duration_sec = pd.Series(0.0, index=df.index)
    else:
        duration_sec = _series_duration_to_seconds(dur_raw).reindex(df.index).fillna(0.0)

    claim = _col(df, "claim_status", "claim", "claim_verification")
    verified = _col(
        df,
        "verified_status",
        "verified",
        "user_verified",
        "author_verification",
    )
    banned = _col(df, "author_ban_status", "ban_status", "banned", "author_banned")

    out = pd.DataFrame(
        {
            "video_view_count": play,
            "video_duration_sec": duration_sec,
            "video_transcription_text": text.fillna("").astype(str),
            "video_like_count": pd.to_numeric(likes, errors="coerce")
            if likes is not None
            else 0,
            "video_share_count": pd.to_numeric(shares, errors="coerce")
            if shares is not None
            else 0,
            "video_download_count": pd.to_numeric(downloads, errors="coerce")
            if downloads is not None
            else 0,
            "video_comment_count": pd.to_numeric(comments, errors="coerce")
            if comments is not None
            else 0,
            "claim_status": claim.astype(str).str.lower().str.strip()
            if claim is not None
            else "unknown",
            "verified_status": verified.astype(str).str.lower().str.strip()
            if verified is not None
            else "unknown",
            "author_ban_status": banned.astype(str).str.lower().str.strip()
            if banned is not None
            else "unknown",
        },
        index=df.index,
    )
    return out
