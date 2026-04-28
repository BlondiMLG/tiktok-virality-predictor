from pathlib import Path

import pandas as pd

from dataset_normalize import infer_dataset_id, normalize_dataset


def load_data(
    path: str = "data/tiktok_claims.csv",
    dataset: str = "auto",
) -> tuple[pd.DataFrame, str]:
    data_path = Path(path)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    df = pd.read_csv(
        data_path,
        low_memory=False,
        encoding="utf-8-sig",
    )
    if df.empty:
        raise ValueError(f"Dataset is empty: {data_path}")

    resolved = infer_dataset_id(df) if dataset == "auto" else dataset
    normalized = normalize_dataset(df, resolved)
    return normalized, resolved
