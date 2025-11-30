# trees/preprocess.py

import pandas as pd
from sklearn.model_selection import train_test_split

from utils.load_data import load_data

def split_features_and_labels(df: pd.DataFrame):
    """
    Split the dataframe into:
    - X_raw: all feature columns
    - y_binary: 0/1 label (attack vs normal)
    - y_multiclass: attack_cat string labels
    """
    if "label" not in df.columns or "attack_cat" not in df.columns:
        raise ValueError("Expected 'label' and 'attack_cat' columns in the dataset.")

    y_binary = df["label"]
    y_multiclass = df["attack_cat"]
    X_raw = df.drop(columns=["label", "attack_cat"])

    return X_raw, y_binary, y_multiclass

def encode_features(X_raw: pd.DataFrame) -> pd.DataFrame:
    """
    One-hot encode non-numeric columns (proto, service, state, etc.)
    """
    non_numeric = X_raw.select_dtypes(include=["object"]).columns.tolist()
    X = pd.get_dummies(X_raw, columns=non_numeric, drop_first=True)
    return X

def get_binary_splits(test_size: float = 0.2, random_state: int = 42):
    """
    Convenience helper:
    load data -> split -> encode -> train/test split for binary task.
    """
    df = load_data()
    X_raw, y_binary, _ = split_features_and_labels(df)
    X = encode_features(X_raw)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_binary,
        test_size=test_size,
        random_state=random_state,
        stratify=y_binary
    )

    return X_train, X_test, y_train, y_test


def get_multiclass_splits(test_size: float = 0.2, random_state: int = 42):
    """
    Same as above but for multiclass (attack_cat).
    """
    df = load_data()
    X_raw, _, y_multiclass = split_features_and_labels(df)
    X = encode_features(X_raw)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_multiclass,
        test_size=test_size,
        random_state=random_state,
        stratify=y_multiclass
    )

    return X_train, X_test, y_train, y_test
