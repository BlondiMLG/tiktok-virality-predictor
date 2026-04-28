# TikTok CSV sources (Kaggle / exports)

Place your file under `data/` and pass `--data-path`. Use `--dataset auto` (default) or force `tiktok_claims` / `tiktok_engagement`.

An older single-file export named `tiktok_dataset.csv` was removed from this repo in favor of **`tiktok_merged_data_deduplicated.csv`** (large engagement merge) and/or **`tiktok_claims.csv`** (claims default). Prefer those paths in scripts and docs.

**In this repo (defaults / extras):** `data/tiktok_claims.csv` is the default for `main.py` (~19k Kaggle-style rows, `video_view_count` + claim/text columns). `data/tiktok_merged_data_deduplicated.csv` is a large engagement export (`plays` + optional `views` + `description`); the normalizer prefers `plays` when both exist. Use `--data-path` with `auto` or `tiktok_engagement`. Optional: `data/tiktok_collected_videos.csv`, `data/tiktok_funny_hashtag_videos.csv` (smaller custom scrapes).

Install the [Kaggle API](https://github.com/Kaggle/kaggle-api): `pip install kaggle`, then put `kaggle.json` in `%USERPROFILE%\.kaggle\` (Windows) or `~/.kaggle/`.

## 1. Canonical TikTok schema (`tiktok_claims`)

Your primary CSV already matches the pipeline: `video_view_count`, `video_transcription_text`, `claim_status`, `verified_status`, `author_ban_status`, duration, engagement columns, etc.

**Schema id:** `tiktok_claims` (auto when `video_view_count` is present).

## 2. Alternative TikTok exports (`tiktok_engagement`)

Many TikTok dumps use **`play_count`**, **`digg_count`**, **`share_count`**, **`comment_count`**, **`duration`**, **`description`**, etc. The normalizer maps those to the same canonical columns.

**Schema id:** `tiktok_engagement` (auto when `play_count` is present, or `views` / `view_count` without `video_view_count`).

**Examples on Kaggle (verify columns match TikTok-style names before training):**

- [TikTok Video Dataset — wasifullahcs](https://www.kaggle.com/datasets/wasifullahcs/tiktok-video-dataset)
- [TikTok User Engagement Data — yakhyojon](https://www.kaggle.com/datasets/yakhyojon/tiktok)

## Quick test (fixture in repo)

```bash
python main.py --data-path tests/fixtures/tiktok_play_sample.csv --dataset tiktok_engagement --artifact-dir artifacts_tok
```

## Licenses

Respect each dataset’s license on Kaggle. This repository does not redistribute their CSVs.
