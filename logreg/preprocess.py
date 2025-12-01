# logreg/preprocess.py

# This module loads the CSVs, cleans the data, builds the feature list,
# and creates the shared preprocessor used by both binary and multiclass
# logistic regression models.

import sys
import os
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

import sklearn
import joblib  # not used yet, but kept to match original environment

# Find project root (ml-based-cyber-security folder)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Data directory
DATA_DIR = PROJECT_ROOT / "data"

TRAIN_CSV = DATA_DIR / "training_data.csv"
FEATURES_CSV = DATA_DIR / "feature_description.csv"

print("Resolved project root:", PROJECT_ROOT)
print("Training CSV exists:", TRAIN_CSV.exists())
print("Feature description exists:", FEATURES_CSV.exists())

# Column name assumptions (same as Colab)
possible_label_cols = ["label", "Label", "attack", "is_attack"]
possible_attack_cat = ["attack_cat", "attack_catagory", "attack_category", "attack_type"]

BINARY_TARGET = "label"      # 0 normal, 1 attack
MULTI_TARGET = "attack_cat"  # multiclass attack category
ID_COL = "id"


def load_and_preprocess():
    """
    Load the CSVs, clean the dataset, create is_attack and attack_cat_encoded,
    build the feature list and the ColumnTransformer preprocessor.

    Returns
    -------
    df : pd.DataFrame
        Full cleaned dataframe including target columns.
    features : list[str]
        Feature column names used for X.
    preprocessor : ColumnTransformer
        Fitted preprocessing transformer (imputing, scaling, one-hot encoding).
    """

    # Load CSVs (latin-1 to match Colab behavior)
    df = pd.read_csv(TRAIN_CSV, encoding="latin-1")
    feat = pd.read_csv(FEATURES_CSV, encoding="latin-1")

    print("Training data shape:", df.shape)
    print("\n---\nFeature description preview:")
    print(feat.head(10))

    print("Columns:", df.columns.tolist())
    print("\nDtypes:\n", df.dtypes.value_counts())

    missing = df.isna().sum().sort_values(ascending=False)
    print("\nMissing values (top 30):")
    print(missing.head(30))

    print("\nColumns present that match common names:")
    print([c for c in df.columns if c in possible_label_cols])
    print([c for c in df.columns if c in possible_attack_cat])

    # Fix BOM id column name if present
    if "ï»¿id" in df.columns:
        df = df.rename(columns={"ï»¿id": "id"})

    # Drop ID if present
    if ID_COL in df.columns:
        df = df.drop(columns=[ID_COL])
        print("Dropped id column")

    # Clean service / proto / state columns (same logic as in Colab)
    for col in ["service", "proto", "state"]:
        if col in df.columns:
            df[col] = df[col].astype(str)
            df[col] = df[col].str.strip()
            df[col] = df[col].replace(
                ["-", "?", "None", "nan", ""], "unknown"
            )
            df[col] = df[col].fillna("unknown")

    for col in ["service", "proto", "state"]:
        if col in df.columns:
            print(
                f"-> {col}: unique count =",
                df[col].nunique(),
                "example values =",
                df[col].unique()[:10],
            )

    # Ensure binary target exists
    if BINARY_TARGET not in df.columns:
        raise KeyError(
            f"Binary target column '{BINARY_TARGET}' not found in dataframe."
        )

    # Ensure binary target is numeric 0/1 (same as original logic)
    if df[BINARY_TARGET].dtype == object:
        df["is_attack"] = (
            df[BINARY_TARGET]
            .astype(str)
            .str.lower()
            .isin(["1", "attack", "anomaly", "attack!", "attack?"])
            .astype(int)
        )
    else:
        df["is_attack"] = df[BINARY_TARGET].astype(int)

    print("\nis_attack distribution:")
    print(df["is_attack"].value_counts())

    # Encode multiclass column if present
    if MULTI_TARGET in df.columns:
        le_attack = LabelEncoder()
        df["attack_cat_encoded"] = df[MULTI_TARGET].fillna("unknown").astype(str)
        df["attack_cat_encoded"] = le_attack.fit_transform(
            df["attack_cat_encoded"]
        )

        print("\nMulticlass mapping (first 20):")
        mapping = dict(
            zip(le_attack.classes_, le_attack.transform(le_attack.classes_))
        )
        print(mapping)

        print("\nattack_cat value counts:")
        print(df[MULTI_TARGET].value_counts())
    else:
        print(
            f"No multiclass attack column '{MULTI_TARGET}' found; "
            "attack_cat_encoded will not be created."
        )

    # Drop obvious high-cardinality / leakage columns (same heuristic as Colab)
    drop_cols = []
    for cand in [
        "srcip",
        "dstip",
        "Stime",
        "Ltime",
        "stime",
        "ltime",
        "stime_ms",
    ]:
        if cand in df.columns:
            drop_cols.append(cand)

    print("\nInitial drop_cols:", drop_cols)

    # Numeric columns detection
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    exclude = ["is_attack", "attack_cat_encoded", BINARY_TARGET]
    numeric_cols = [c for c in numeric_cols if c not in exclude]
    print("\nNumeric candidate columns (sample):", numeric_cols[:20])

    # Categorical columns to encode
    cat_cols = [c for c in ["service", "proto", "state"] if c in df.columns]
    print("\nCategorical columns to encode:", cat_cols)

    # Final features list
    features = [c for c in numeric_cols + cat_cols if c not in drop_cols]
    print("\nFinal features to be used (sample):", features[:40])

    # Build preprocessors (same as original)
    num_features = [
        c
        for c in features
        if c in df.columns and np.issubdtype(df[c].dtype, np.number)
    ]
    cat_features = [
        c for c in features if c in df.columns and c not in num_features
    ]

    num_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    cat_transformer = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="constant", fill_value="unknown"),
            ),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_transformer, num_features),
            ("cat", cat_transformer, cat_features),
        ],
        remainder="drop",
    )

    print("\nnum_features count:", len(num_features))
    print("cat_features count:", len(cat_features))

    return df, features, preprocessor


if __name__ == "__main__":
    # Quick manual test
    _df, _features, _preproc = load_and_preprocess()
    print("Done preprocessing. Shape:", _df.shape)
