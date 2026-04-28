# TikTok Virality Predictor

Predicts whether a TikTok video lands in the top slice of `video_view_count` (default: above the 80th percentile), using metadata and transcript text available before or at upload time—not future engagement.

Included pieces: train/val/test split, features that drop obvious leakage (likes/shares/etc.), logistic regression plus random forest and XGBoost behind the same preprocessing, optional `RandomizedSearchCV` and probability calibration, threshold tuning on the validation set, and CSV/plots written under `artifacts/` after a run.

## Problem Definition

Binary label:

- `viral = 1` if `video_view_count` is above the train-split threshold at your chosen quantile (default 80th percentile)
- `viral = 0` otherwise

The positive class is the minority (~80/20), so reported metrics lean on F1, PR-AUC, and precision/recall at the tuned threshold rather than accuracy alone.

## Pipeline Overview

1. **Load + validate** — File exists, non-empty CSV; optional column remap for other TikTok exports (`--dataset`; details in `data/KAGGLE_DATASETS.md`).
2. **Clean** — Normalize categorical fields, fill duration gaps where needed.
3. **Target** — Virality cutoff is computed **only from the training split** and applied to val/test.
4. **Split** — Train / validation / test. Stratification uses a **proxy**: “above quantile within that slice” so folds aren’t empty; the real `viral` column always uses the **single train-derived cutoff**.
5. **Features** — Text stats (`word_count`, `uppercase_ratio`, `text_density`, …), duration buckets, `hook_3_words`, drop leakage columns (post-upload counts).
6. **Models** — TF-IDF (1–3 grams) + scaled numerics + one-hot categoricals (`ColumnTransformer`). Optional tuning on train (3-fold search per family), optional `CalibratedClassifierCV` on train, 5-fold stratified CV for headline metrics on train, threshold sweep on validation (`f1`, `precision`, or `f1_with_min_precision`).
7. **Eval + export** — Metrics at the chosen threshold on val (for picking a model) and on test; summary CSV, JSON run metadata, plots, per-model error-analysis CSVs.

## Project Structure

- `dataloader.py` — CSV load + normalization dispatch
- `dataset_normalize.py` — Map Kaggle/export columns to the schema `main.py` expects
- `data/KAGGLE_DATASETS.md` — Where to get CSVs and which `--dataset` to use
- `preprocess.py` — Cleaning, target, features
- `model.py` — Pipelines, tuning, CV, thresholds, evaluation helpers
- `main.py` — CLI and artifact writes
- `tests/test_smoke.py` — Load/normalize/clean/feature smoke tests
- `tests/test_model.py` — Threshold helpers + small sklearn integration checks
- `requirements.txt` — Pinned deps
- `artifacts/` — Run outputs (gitignored): `model_summary.csv`, `run_metadata.json`, `plots/*.png`, `error_analysis_<model>.csv`

## Installation

```bash
pip install -r requirements.txt
```

Tests:

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

## Usage

Default (expects `data/tiktok_claims.csv` locally; see `data/KAGGLE_DATASETS.md` if you need to fetch data):

```bash
python main.py
```

Example with flags:

```bash
python main.py --data-path data/tiktok_claims.csv --quantile 0.80 --threshold-metric f1_with_min_precision --min-precision 0.50 --min-recall 0.20 --artifact-dir artifacts
```

`play_count`-style export:

```bash
python main.py --data-path data/my_tiktok_export.csv --dataset tiktok_engagement --artifact-dir artifacts
```

Tuning + calibration:

```bash
python main.py --tune --tune-iter 24 --calibrate --calibrate-method sigmoid
```

### CLI reference

| Flag | Meaning |
|------|---------|
| `--data-path` | CSV path |
| `--dataset` | `auto`, `tiktok_claims`, or `tiktok_engagement` (see `data/KAGGLE_DATASETS.md`) |
| `--quantile` | Quantile for virality cutoff on train (default `0.80`) |
| `--threshold-metric` | `f1`, `precision`, or `f1_with_min_precision` |
| `--min-precision` / `--min-recall` | Floors for `f1_with_min_precision` |
| `--model-selection` | Pick “best” row: `val_f1` (default), `val_pr_auc`, or `cv_f1` (CV uses prob 0.5; val uses tuned threshold) |
| `--tune` / `--tune-iter` | Random search on train (`--tune-iter` trials per model) |
| `--calibrate` / `--calibrate-method` | Platt (`sigmoid`) or `isotonic` calibration on train |
| `--artifact-dir` | Output folder |

## Reading the outputs

- **CV columns** (`cv_f1_mean`, …): stratified folds on train; predictions at 0.5 within each fold.
- **Val columns** (`val_*`): metrics at the threshold chosen on validation.
- **Test columns**: holdout metrics at that same threshold.
- **`brier`**: rough check on probability sharpness.
- **Error CSVs**: rows from test with true label, predicted label, and score.
- **Plots**: PR (legend AP matches sklearn `average_precision_score`), ROC, confusion matrix.

For imbalance, lean on PR-AUC and precision/recall more than accuracy.

## Optional extensions

- Slice error CSVs by caption length, claim status, etc.
- Second dataset from another time period or source if you want an out-of-sample story.
- Save the fitted pipeline with `joblib` (or similar) and store a hash of the training file next to it.
