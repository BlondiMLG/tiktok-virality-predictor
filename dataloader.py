import pandas as pd

def load_data(path="data/tiktok_dataset.csv"):
    df = pd.read_csv(path)
    return df