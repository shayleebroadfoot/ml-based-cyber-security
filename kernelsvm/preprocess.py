# kernelsvm/preprocess.py
#
# Preprocessing tailored for kernel SVMs (RBF / polynomial):
# - Same core cleaning and feature engineering as linsvm.preprocess
# - Log-transform nonnegative numeric features
# - Standard scaling + one-hot on grouped categories
# - Provides binary and multiclass splits

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# Resolve project root and data paths relative to this file
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

# Main input CSVs
TRAIN_CSV = DATA_DIR / "training_data.csv"
FEATURES_CSV = DATA_DIR / "feature_description.csv"

# Target column names in the dataset
BINARY_TARGET = "label"        # 0 normal, 1 attack (if present)
MULTI_TARGET = "attack_cat"    # attack category
ID_COL = "id"


def _clean_base_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Basic cleaning:
    - Fix weird BOM-prefixed id column name
    - Drop id column (not useful as a feature)
    - Normalize key categorical columns and replace missing/odd values
    """
    # Some CSV exports add a BOM -> "ï»¿id", so I rename it back to "id"
    if "ï»¿id" in df.columns:
        df = df.rename(columns={"ï»¿id": "id"})

    # Drop the id column entirely so it is not used as a feature
    if ID_COL in df.columns:
        df = df.drop(columns=[ID_COL])
        print("Dropped id column")

    # Standardize a few important categorical columns
    for col in ["service", "proto", "state"]:
        if col in df.columns:
            # Make sure values are clean strings with no extra whitespace
            df[col] = df[col].astype(str).str.strip()
            # Replace placeholder/missing-like values with a consistent "unknown"
            df[col] = df[col].replace(
                ["-", "?", "None", "none", "nan", ""], "unknown"
            )
            df[col] = df[col].fillna("unknown")

    return df


def _map_proto_group(proto: str) -> str:
    """
    Map raw protocol values into a smaller set of protocol groups.
    This reduces the cardinality of 'proto' before one-hot encoding.
    """
    p = str(proto).lower()

    if p == "tcp":
        return "tcp"
    if p == "udp":
        return "udp"
    if p == "icmp":
        return "icmp"
    if p == "arp":
        return "arp"
    if p in {"ospf", "rtp", "igmp", "ipv6-frag", "ipv6-icmp", "ipv6-route"}:
        return "routing"
    if p == "unknown":
        return "unknown"
    return "other"


def _map_state_group(state: str) -> str:
    """
    Group 'state' values into broader connection state categories.
    """
    s = str(state).upper()

    if s in {"CON", "FIN"}:
        return "established"
    if s in {"ECO", "REQ"}:
        return "request"
    if s in {"RST"}:
        return "reset"
    if s in {"PAR", "INT", "URN"}:
        return "partial"
    if s == "NO":
        return "none"
    return "other"


def _map_service_group(service: str) -> str:
    """
    Group 'service' values into broader service types.
    """
    s = str(service).lower()

    if s in {"http"}:
        return "web"
    if s in {"ftp", "ftp-data"}:
        return "file_transfer"
    if s in {"smtp", "pop3"}:
        return "email"
    if s in {"dns"}:
        return "dns"
    if s in {"ssh"}:
        return "remote_access"
    if s in {"snmp"}:
        return "management"
    if s in {"radius"}:
        return "auth"
    if s == "unknown":
        return "unknown"
    return "other"


def _add_grouped_categories(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add grouped categorical columns that are friendlier for one-hot encoding.
    """
    if "proto" in df.columns:
        df["proto_group"] = df["proto"].map(_map_proto_group)
    if "state" in df.columns:
        df["state_group"] = df["state"].map(_map_state_group)
    if "service" in df.columns:
        df["service_group"] = df["service"].map(_map_service_group)
    return df


def _add_engineered_numeric_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create additional numeric features from the existing traffic stats.
    These are simple ratios and totals that may help the SVM separate patterns.
    """

    def has_cols(cols):
        # Helper to check that all required columns exist before computing a feature
        return all(c in df.columns for c in cols)

    # Bytes-based totals and ratios
    if has_cols(["sbytes", "dbytes"]):
        df["bytes_total"] = df["sbytes"] + df["dbytes"]
        df["bytes_ratio"] = df["sbytes"] / (df["dbytes"] + 1.0)

    # Packet-based totals and ratios
    if has_cols(["spkts", "dpkts"]):
        df["pkts_total"] = df["spkts"] + df["dpkts"]
        df["pkts_ratio"] = df["spkts"] / (df["dpkts"] + 1.0)

    # TTL difference between source and destination
    if has_cols(["sttl", "dttl"]):
        df["ttl_diff"] = df["sttl"] - df["dttl"]

    # Load totals and ratios
    if has_cols(["sload", "dload"]):
        df["load_total"] = df["sload"] + df["dload"]
        df["load_ratio"] = df["sload"] / (df["dload"] + 1.0)

    # Packet interval ratio
    if has_cols(["sinpkt", "dinpkt"]):
        df["pkt_interval_ratio"] = df["sinpkt"] / (df["dinpkt"] + 1.0)

    # Jitter total
    if has_cols(["sjit", "djit"]):
        df["jitter_total"] = df["sjit"] + df["djit"]

    return df


def _build_targets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the binary 'is_attack' target and an encoded multiclass target.
    - is_attack: 0 normal, 1 attack
    - attack_cat_encoded: integer encoding of attack_cat
    """
    # First, build the binary target. If a binary label already exists, I reuse it.
    if BINARY_TARGET in df.columns:
        df["is_attack"] = df[BINARY_TARGET].astype(int)
    elif MULTI_TARGET in df.columns:
        # Otherwise I derive it from the multiclass label (Normal vs not Normal)
        df["is_attack"] = (df[MULTI_TARGET] != "Normal").astype(int)
    else:
        raise KeyError(
            f"Cannot create is_attack: neither '{BINARY_TARGET}' nor '{MULTI_TARGET}' found."
        )

    print("\nis_attack distribution:")
    print(df["is_attack"].value_counts())

    # Then build an encoded multiclass target if the column exists
    if MULTI_TARGET in df.columns:
        cats = df[MULTI_TARGET].fillna("unknown").astype(str)
        unique_cats = sorted(cats.unique())
        mapping = {name: idx for idx, name in enumerate(unique_cats)}
        df["attack_cat_encoded"] = cats.map(mapping)

        print("\nMulticlass mapping (first 20):")
        print({k: mapping[k] for k in list(mapping.keys())[:20]})

        print("\nattack_cat value counts:")
        print(df[MULTI_TARGET].value_counts())
    else:
        print(f"\nWARNING: multiclass column '{MULTI_TARGET}' not found.")

    return df


def _load_and_preprocess():
    """
    Load raw CSVs, apply cleaning, feature engineering, and build a ColumnTransformer
    that handles numeric and categorical preprocessing for the kernel SVM.
    """
    # Load training data and feature descriptions
    df = pd.read_csv(TRAIN_CSV, encoding="latin-1")
    feat = pd.read_csv(FEATURES_CSV, encoding="latin-1")

    print("Training data shape:", df.shape)
    print("\n---\nFeature description preview:")
    print(feat.head(10))

    print("Columns:", df.columns.tolist())
    print("\nDtypes:\n", df.dtypes.value_counts())

    # Quick look at missing values to understand data quality
    missing = df.isna().sum().sort_values(ascending=False)
    print("\nMissing values (top 30):")
    print(missing.head(30))

    # Apply my preprocessing steps in a fixed order
    df = _clean_base_columns(df)
    df = _add_grouped_categories(df)
    df = _build_targets(df)
    df = _add_engineered_numeric_features(df)

    # Columns that I explicitly do NOT want as features
    drop_cols = []
    for cand in [
        "srcip",
        "dstip",
        "Stime",
        "Ltime",
        "stime",
        "ltime",
        "stime_ms",
        "proto",
        "state",
        "service",
    ]:
        if cand in df.columns:
            drop_cols.append(cand)

    print("\nInitial drop_cols:", drop_cols)

    # Start from all numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # Exclude target-related columns so they are not fed into the model
    exclude = ["is_attack", "attack_cat_encoded", BINARY_TARGET]
    numeric_cols = [c for c in numeric_cols if c not in exclude]

    print("\nNumeric candidate columns (sample):", numeric_cols[:20])

    # Categorical columns I want to encode via one-hot
    cat_candidates = ["service_group", "proto_group", "state_group"]
    cat_cols = [c for c in cat_candidates if c in df.columns]
    print("\nCategorical columns to encode:", cat_cols)

    # Identify numeric columns that are nonnegative so log1p makes sense
    nonneg_numeric = [c for c in numeric_cols if df[c].min() >= 0]
    print("\nApplying log1p to nonnegative numeric features (count):", len(nonneg_numeric))

    # Apply log1p transform in-place on these columns
    for c in nonneg_numeric:
        df[c] = np.log1p(df[c])

    # Final feature list (numeric + grouped categorical), minus dropped columns
    features = [c for c in numeric_cols + cat_cols if c not in drop_cols]
    print("\nFinal features to be used (sample):", features[:40])

    # Split features into numeric vs categorical for the ColumnTransformer
    num_features = [
        c for c in features if c in df.columns and np.issubdtype(df[c].dtype, np.number)
    ]
    cat_features = [c for c in features if c in df.columns and c not in num_features]

    # Numeric pipeline: median imputation + standardization
    num_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    # Categorical pipeline: constant imputation + one-hot encoding
    cat_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    # Combine numeric and categorical transformers into a single preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_transformer, num_features),
            ("cat", cat_transformer, cat_features),
        ],
        remainder="drop",
    )

    print("\nnum_features count:", len(num_features))
    print("cat_features count:", len(cat_features))

    # Return the preprocessed dataframe, the feature column list, and the preprocessor
    return df, features, preprocessor


def get_binary_splits(test_size: float = 0.20, random_state: int = 42):
    """
    Create binary train/test splits plus the shared preprocessor.

    Returns:
        X_train, X_test, y_train, y_test, preprocessor

    I keep the preprocessor separate so I can build:
        Pipeline(steps=[("preprocess", preprocessor), ("clf", SVC(...))])
    """
    df, features, preprocessor = _load_and_preprocess()

    if "is_attack" not in df.columns:
        raise KeyError("Column 'is_attack' not found after preprocessing.")

    # For binary classification I use all rows (normal + attacks)
    X = df[features].copy()
    y = df["is_attack"].astype(int)

    # Stratified split to preserve attack/normal proportion in both sets
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
        shuffle=True,
    )

    print(f"\n[Kernel SVM] Binary: X_train {X_train.shape}, X_test {X_test.shape}")
    return X_train, X_test, y_train, y_test, preprocessor


def get_multiclass_splits(test_size: float = 0.20, random_state: int = 42):
    """
    Create multiclass train/test splits for attack categories only.
    Normal traffic is removed here; I only keep rows marked as attack.
    """
    df, features, preprocessor = _load_and_preprocess()

    if "is_attack" not in df.columns:
        raise KeyError("Column 'is_attack' not found after preprocessing.")
    if "attack_cat_encoded" not in df.columns:
        raise KeyError("Column 'attack_cat_encoded' not found for multiclass splits.")

    # Keep only attack rows (is_attack == 1) for the multiclass task
    df_attacks = df[df["is_attack"] == 1].copy()
    print(f"Total attack rows for multiclass (kernel SVM): {df_attacks.shape[0]}")

    X = df_attacks[features].copy()
    y = df_attacks["attack_cat_encoded"].astype(int)

    # Stratified split to preserve the distribution of attack classes
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
        shuffle=True,
    )

    print(f"[Kernel SVM] Multiclass: X_train {X_train.shape}, X_test {X_test.shape}")
    return X_train, X_test, y_train, y_test, preprocessor


if __name__ == "__main__":
    # Quick sanity check when running this file directly
    Xtr, Xte, ytr, yte, pre = get_multiclass_splits()
    print("Done. Example shapes:")
    print("X_train:", Xtr.shape, "X_test:", Xte.shape)
    print("y_train:", ytr.shape, "y_test:", yte.shape)
