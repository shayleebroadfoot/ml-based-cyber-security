# utils/load_data.py

import pandas as pd

def load_data(path: str = "data/training_data.csv") -> pd.DataFrame:
    """
    Load the training CSV and drop the 'id' column if present.
    """
    df = pd.read_csv(path)

    if "id" in df.columns:
        df = df.drop(columns=["id"])

    return df
