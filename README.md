# TikTok Virality Predictor

End-to-end machine learning project to predict whether a TikTok video becomes viral (top quantile by `video_view_count`) using pre-upload metadata and transcript text.

This repository is built to be portfolio-ready:

- Reproducible train/validation/test split
- Leakage-aware feature engineering
- Multiple model families (LogReg, Random Forest, XGBoost)
- Cross-validation, optional randomized hyperparameter search, and optional probability calibration
- Threshold tuning on the validation split (aligned with reported validation metrics)
- Artifact export for clean result tracking
- Automatic diagnostic plot and error-analysis export

## Problem Definition

Binary classification task:

- `viral = 1` if `video_view_count` exceeds the selected quantile threshold (default: 80th percentile)
- `viral = 0` otherwise

The dataset is imbalanced (roughly 80/20), so the main metrics are:

- F1 score
- PR-AUC (average precision)
- Precision/Recall trade-off at tuned threshold

## Pipeline Overview

1. **Load + validate data**
   - checks file existence and non-empty dataset
   - optional **schema normalization** for other TikTok CSV exports (`--dataset`; see `data/KAGGLE_DATASETS.md`)
2. **Clean data**
   - normalizes key categorical fields
   - handles missing duration values
3. **Create target**
   - threshold computed on train split only
4. **Train / validation / test split**
   - stratification uses a **proxy label**: whether `video_view_count` is above the chosen quantile **within that split**. This stabilizes class balance in each fold. The actual `viral` label always uses the **single threshold learned from the training split only**, so the proxy is not the same as the final `y`—it is only for stratification.
5. **Feature engineering**
   - text statistics (`word_count`, `uppercase_ratio`, `text_density`, etc.)
   - duration bucketing
   - hook feature (`hook_3_words`)
   - strict removal of leakage columns (post-upload engagement metrics)
6. **Modeling**
   - TF-IDF (1-3 grams) + numeric scaling + one-hot categorical encoding via `ColumnTransformer`
   - optional `RandomizedSearchCV` (3-fold) on train for each model family
   - optional `CalibratedClassifierCV` (3-fold on train; validation stays unused for fitting calibrators)
   - 5-fold stratified CV metrics on train (same hyperparameters as the final base estimator; calibration wrapper is not cross-validated in that step)
   - threshold search on validation set (`f1`, `precision`, or `f1_with_min_precision`)
7. **Evaluation + export**
   - validation metrics at the chosen threshold (for model selection)
   - full test metrics per model (holdout)
   - saves artifacts to `artifacts/` (summary, metadata, plots, error analysis)

## Project Structure

- `dataloader.py` - load CSV and apply schema normalization
- `dataset_normalize.py` - maps Kaggle-style columns to the canonical pipeline schema
- `data/KAGGLE_DATASETS.md` - TikTok-only Kaggle / export notes
- `preprocess.py` - cleaning, target creation, feature engineering
- `model.py` - preprocessors, model builders, tuning, CV, threshold tuning, evaluation
- `main.py` - orchestration and artifact export
- `tests/test_smoke.py` — loader, schema normalization, clean/target/featurize smoke tests
- `tests/test_model.py` — threshold utilities, stratification helper, CV + evaluation smoke tests on synthetic frames
- `requirements.txt` - pinned Python dependencies
- `artifacts/` - generated run outputs (gitignored; reproduce with `python main.py`):
  - `model_summary.csv`
  - `run_metadata.json`
  - `plots/*.png` (PR, ROC, confusion matrix per model)
  - `error_analysis_<model>.csv`

## Installation

```bash
pip install -r requirements.txt
```

Run tests:

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

## Usage

Default run:

```bash
python main.py
```

Run with custom options:

```bash
python main.py --data-path data/tiktok_claims.csv --quantile 0.80 --threshold-metric f1_with_min_precision --min-precision 0.50 --min-recall 0.20 --artifact-dir artifacts
```

Use another TikTok CSV with `play_count`-style columns:

```bash
python main.py --data-path data/my_tiktok_export.csv --dataset tiktok_engagement --artifact-dir artifacts
```

Enable hyperparameter search and calibration:

```bash
python main.py --tune --tune-iter 24 --calibrate --calibrate-method sigmoid
```

CLI options:

- `--data-path`: path to dataset CSV
- `--dataset`: `auto`, `tiktok_claims` (default column names), or `tiktok_engagement` (`play_count` / `digg_count`-style TikTok exports); see `data/KAGGLE_DATASETS.md`
- `--quantile`: virality quantile threshold (default `0.80`)
- `--threshold-metric`: optimize threshold for `f1`, `precision`, or `f1_with_min_precision`
- `--min-precision`: precision floor used by `f1_with_min_precision` (default `0.50`)
- `--min-recall`: recall floor used by `f1_with_min_precision` (default `0.20`)
- `--model-selection`: how to pick the reported best model: `val_f1` (default), `val_pr_auc`, or `cv_f1` (CV is at probability threshold 0.5; validation metrics use the tuned threshold)
- `--tune`: run randomized hyperparameter search per model on train
- `--tune-iter`: sampled configurations per model when `--tune` is set (default `18`)
- `--calibrate`: wrap the classifier with `CalibratedClassifierCV` (fit with 3-fold CV on train)
- `--calibrate-method`: `sigmoid` (Platt) or `isotonic`
- `--artifact-dir`: output directory for run artifacts

## How To Read Results

For each model you get:

- **Cross-validation metrics** (`cv_f1_mean`, `cv_pr_auc_mean`, etc.) from train (threshold 0.5 within each fold)
- **Validation metrics at tuned threshold** (`val_f1`, `val_precision`, `val_recall`, `val_pr_auc`)
- **Best threshold** selected on validation split
- **Test metrics** (`accuracy`, `precision`, `recall`, `f1`, `pr_auc`, `roc_auc`)
- **Calibration proxy** (`brier`) for probability quality
- **Error analysis file** with true/pred labels and confidence
- **Plots**: PR curve (legend shows **AP**, matching `average_precision_score`), ROC curve, confusion matrix

Use `pr_auc`, `precision`, and `recall` as primary guidance for imbalanced data.

## Optional extensions

- Notebook or report slicing error-analysis CSVs by metadata (FP vs FN)
- Holdout from another period or source for stronger generalization claims
- Serialize the trained `Pipeline` (e.g. `joblib`) and record a hash of the training CSV for reproducibility
