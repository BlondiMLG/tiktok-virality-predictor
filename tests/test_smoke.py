import unittest
from pathlib import Path

import pandas as pd

from dataset_normalize import (
    DATASET_TIKTOK_CLAIMS,
    DATASET_TIKTOK_ENGAGEMENT,
    infer_dataset_id,
    normalize_dataset,
)
from dataloader import load_data
from preprocess import clean_data, create_target, feature_engineering


class TestNormalizeAndLoad(unittest.TestCase):
    def test_infer_engagement_synthetic(self) -> None:
        df = pd.DataFrame(
            {
                "plays": [1000.0, 2000.0],
                "description": ["a", "b"],
                "duration": [10, 20],
            }
        )
        self.assertEqual(infer_dataset_id(df), DATASET_TIKTOK_ENGAGEMENT)
        out = normalize_dataset(df, DATASET_TIKTOK_ENGAGEMENT)
        self.assertIn("video_view_count", out.columns)
        self.assertListEqual(out["video_view_count"].tolist(), [1000.0, 2000.0])

    def test_load_fixture_tiktok_engagement(self) -> None:
        root = Path(__file__).resolve().parents[1]
        path = root / "tests" / "fixtures" / "tiktok_play_sample.csv"
        df, dataset_id = load_data(str(path), dataset="tiktok_engagement")
        self.assertEqual(dataset_id, DATASET_TIKTOK_ENGAGEMENT)
        self.assertEqual(len(df), 3)
        self.assertIn("video_view_count", df.columns)
        c = clean_data(df)
        t = c["video_view_count"].quantile(0.5)
        fe = create_target(c, threshold=t)
        fe = feature_engineering(fe)
        self.assertIn("viral", fe.columns)
        self.assertIn("word_count", fe.columns)

    @unittest.skipUnless(
        (Path(__file__).resolve().parents[1] / "data" / "tiktok_claims.csv").is_file(),
        "data/tiktok_claims.csv not present",
    )
    def test_load_claims_sample(self) -> None:
        root = Path(__file__).resolve().parents[1]
        path = root / "data" / "tiktok_claims.csv"
        df, dataset_id = load_data(str(path), dataset=DATASET_TIKTOK_CLAIMS)
        self.assertEqual(dataset_id, DATASET_TIKTOK_CLAIMS)
        self.assertGreater(len(df), 100)
        self.assertIn("claim_status", df.columns)


if __name__ == "__main__":
    unittest.main()
