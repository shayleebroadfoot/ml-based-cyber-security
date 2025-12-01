# logreg/preprocess.py
#
# Improved preprocessing for logistic regression:
# - Clean categorical attributes
# - Reduce high-cardinality proto/state/service into grouped categories
# - Add engineered numeric features (ratios / totals / diffs)
# - Log-transform nonnegative numeric features
# - Robust scaling + one-hot on grouped categories
# - Provide train/test splits for binary and multiclass models

import numpy as np
import pandas as pd

from pathlib import Path

from sklearn.preprocessing import LabelEncoder, OneHotEncoder, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

# This file lives in: ml-based-cyber-security/logreg/preprocess.py
# So parents[0] = logreg/, parents[1] = ml-based-cyber-security/
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

TRAIN_CSV = DATA_DIR / "training_data.csv"
FEATURES_CSV = DATA_DIR / "feature_description.csv"

print("Resolved project root:", PROJECT_ROOT)
print("Training CSV exists:", TRAIN_CSV.exists())
print("Feature description exists:", FEATURES_CSV.exists())

# Target column names
BINARY_TARGET = "label"       # 0 normal, 1 attack
MULTI_TARGET = "attack_cat"   # multiclass attack category
ID_COL = "id"


# ---------------------------------------------------------------------
# Helper cleaning + mapping functions
# ---------------------------------------------------------------------

def _clean_base_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Fix BOM id column
    if "ï»¿id" in df.columns:
        df = df.rename(columns={"ï»¿id": "id"})

    # Drop ID if present
    if ID_COL in df.columns:
        df = df.drop(columns=[ID_COL])
        print("Dropped id column")

    # Basic cleaning for original categorical columns
    for col in ["service", "proto", "state"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].replace(
                ["-", "?", "None", "nan", ""], "unknown"
            )
            df[col] = df[col].fillna("unknown")

    return df


def _map_proto_group(proto: str) -> str:
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
    # Create grouped categorical features with low cardinality
    if "proto" in df.columns:
        df["proto_group"] = df["proto"].map(_map_proto_group)
    if "state" in df.columns:
        df["state_group"] = df["state"].map(_map_state_group)
    if "service" in df.columns:
        df["service_group"] = df["service"].map(_map_service_group)

    return df


def _add_engineered_numeric_features(df: pd.DataFrame) -> pd.DataFrame:
    # Helper to safely create features only if source cols exist
    def has_cols(cols):
        return all(c in df.columns for c in cols)

    if has_cols(["sbytes", "dbytes"]):
        df["bytes_total"] = df["sbytes"] + df["dbytes"]
        df["bytes_ratio"] = df["sbytes"] / (df["dbytes"] + 1.0)

    if has_cols(["spkts", "dpkts"]):
        df["pkts_total"] = df["spkts"] + df["dpkts"]
        df["pkts_ratio"] = df["spkts"] / (df["dpkts"] + 1.0)

    if has_cols(["sttl", "dttl"]):
        df["ttl_diff"] = df["sttl"] - df["dttl"]

    if has_cols(["sload", "dload"]):
        df["load_total"] = df["sload"] + df["dload"]
        df["load_ratio"] = df["sload"] / (df["dload"] + 1.0)

    if has_cols(["sinpkt", "dinpkt"]):
        df["pkt_interval_ratio"] = df["sinpkt"] / (df["dinpkt"] + 1.0)

    if has_cols(["sjit", "djit"]):
        df["jitter_total"] = df["sjit"] + df["djit"]

    return df


def _build_targets(df: pd.DataFrame) -> pd.DataFrame:
    # Binary target
    if BINARY_TARGET not in df.columns:
        raise KeyError(f"Binary target column '{BINARY_TARGET}' not found in dataframe.")

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

    # Multiclass encoded target
    if MULTI_TARGET in df.columns:
        le = LabelEncoder()
        df["attack_cat_encoded"] = le.fit_transform(
            df[MULTI_TARGET].fillna("unknown").astype(str)
        )

        mapping = dict(zip(le.classes_, le.transform(le.classes_)))
        print("\nMulticlass mapping (first 20):")
        print(mapping)

        print("\nattack_cat value counts:")
        print(df[MULTI_TARGET].value_counts())
    else:
        print(f"\nWARNING: multiclass column '{MULTI_TARGET}' not found.")

    return df


# ---------------------------------------------------------------------
# Main loader + preprocessor builder
# ---------------------------------------------------------------------

def load_and_preprocess():
    """
    Load CSVs, clean, engineer features, create targets, and build ColumnTransformer.

    Returns
    -------
    df : pd.DataFrame
        Cleaned dataframe with targets and engineered features.
    features : list[str]
        Names of feature columns used as X.
    preprocessor : ColumnTransformer
        Transformer for numeric + categorical preprocessing.
    """
    # Load
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

    # Base cleaning
    df = _clean_base_columns(df)

    # Grouped categories for proto/state/service
    df = _add_grouped_categories(df)

    # Targets
    df = _build_targets(df)

    # Engineered numeric features
    df = _add_engineered_numeric_features(df)

    # Decide which columns to drop outright (high-cardinality / leakage)
    drop_cols = []
    for cand in [
        "srcip",
        "dstip",
        "Stime",
        "Ltime",
        "stime",
        "ltime",
        "stime_ms",
        "proto",    # we use proto_group instead
        "state",    # we use state_group instead
        "service"   # we use service_group instead
    ]:
        if cand in df.columns:
            drop_cols.append(cand)

    print("\nInitial drop_cols:", drop_cols)

    # Numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # Exclude targets and raw label
    exclude = ["is_attack", "attack_cat_encoded", BINARY_TARGET]
    numeric_cols = [c for c in numeric_cols if c not in exclude]

    print("\nNumeric candidate columns (sample):", numeric_cols[:20])

    # Categorical columns (grouped)
    cat_candidates = ["service_group", "proto_group", "state_group"]
    cat_cols = [c for c in cat_candidates if c in df.columns]
    print("\nCategorical columns to encode:", cat_cols)

    # Apply log1p to nonnegative numeric columns with wide range
    nonneg_numeric = [c for c in numeric_cols if df[c].min() >= 0]
    print("\nApplying log1p to nonnegative numeric features (count):", len(nonneg_numeric))

    for c in nonneg_numeric:
        df[c] = np.log1p(df[c])

    # Final features: numeric + categorical (after log transform)
    features = [c for c in numeric_cols + cat_cols if c not in drop_cols]
    print("\nFinal features to be used (sample):", features[:40])

    # Build transformers
    num_features = [c for c in features if c in df.columns and np.issubdtype(df[c].dtype, np.number)]
    cat_features = [c for c in features if c in df.columns and c not in num_features]

    num_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler())
        ]
    )

    cat_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_transformer, num_features),
            ("cat", cat_transformer, cat_features)
        ],
        remainder="drop"
    )

    print("\nnum_features count:", len(num_features))
    print("cat_features count:", len(cat_features))

    return df, features, preprocessor


# ---------------------------------------------------------------------
# Public API used by binary.py and multiclass.py
# ---------------------------------------------------------------------

def get_binary_splits(test_size: float = 0.2, random_state: int = 42):
    """
    Return X_train, X_test, y_train, y_test, preprocessor for binary classification.
    """
    df, features, preprocessor = load_and_preprocess()

    X = df[features]
    y = df["is_attack"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )

    return X_train, X_test, y_train, y_test, preprocessor


def get_multiclass_splits(test_size: float = 0.2, random_state: int = 42):
    """
    Return X_train, X_test, y_train, y_test, preprocessor for multiclass classification.
    Uses only rows where is_attack == 1.
    """
    df, features, preprocessor = load_and_preprocess()

    df_attacks = df[df["is_attack"] == 1].copy()
    if "attack_cat_encoded" not in df_attacks.columns:
        raise KeyError("attack_cat_encoded column not found for multiclass splits.")

    X = df_attacks[features]
    y = df_attacks["attack_cat_encoded"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )

    print(f"Total attack rows for multiclass: {df_attacks.shape[0]}")

    return X_train, X_test, y_train, y_test, preprocessor


if __name__ == "__main__":
    # Manual quick test
    _df, _features, _preproc = load_and_preprocess()
    print("Done preprocessing. Shape:", _df.shape)
    print("Number of features used:", len(_features))
