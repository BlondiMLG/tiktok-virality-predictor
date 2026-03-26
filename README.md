# TikTok Virality Predictor

This project implements a machine learning pipeline that predicts whether a TikTok video will go "viral" (defined as reaching the top 20% of view counts). The classification is based exclusively on video metadata and the corresponding transcription texts.

## Overview and Results

The goal of this project was to build a complete data science pipeline from data cleaning and feature engineering to evaluation. A key challenge was handling the highly imbalanced dataset (80/20 split).

* **Best Model:** Logistic Regression
* **Recall (Viral Class):** 1.00 (100%)
* **F1-Score:** 0.576
* **Applied Methods:** TF-IDF (incl. N-grams), Probability Threshold Tuning, Class Weighting.

Note on model performance: An F1-score of ~0.58 represents a realistic limit for this dataset. Factors like upload timing or the randomness of the algorithm cannot be derived from pre-upload metadata. To minimize missing potentially viral trends, the model was specifically calibrated for maximum recall through threshold tuning.

## Technologies

* Python 3
* pandas
* numpy
* scikit-learn
* xgboost

## Feature Engineering

To prevent data leakage, all post-upload metrics (likes, shares, comments) were strictly removed from the training set. The following features were generated from the remaining data:

1. **Text Metrics:** Word count, text length, ratio of uppercase letters, and the frequency of exclamation marks and question marks.
2. **Hook Extraction:** The first three words of the video were extracted as a separate categorical feature.
3. **NLP:** The raw text was transformed into numerical vectors using `TfidfVectorizer` (excluding stop words and including bi- and tri-grams).

## Project Structure

* `dataloader.py`: Loads the raw data.
* `preprocess.py`: Data cleaning, label generation (target), and feature engineering.
* `model.py`: Definition of the Scikit-Learn pipelines, model training, threshold tuning, and metric evaluation.
* `main.py`: Entry point to execute the entire pipeline.
* `data/tiktok_dataset.csv`: The dataset used.

## Installation and Usage

Clone the repository:
```bash
git clone [https://github.com/USERNAME/tiktok-virality-predictor.git](https://github.com/USERNAME/tiktok-virality-predictor.git)
cd tiktok-virality-predictor
```

Install dependencies:
```bash
pip install pandas numpy scikit-learn xgboost
```

Run the pipeline:
```bash
python main.py
```