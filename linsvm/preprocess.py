# linsvm/preprocess.py
#
# Strong preprocessing for linear SVM:
# - Clean categorical attributes
# - Group proto/state/service into low-cardinality buckets
# - Add engineered numeric features (ratios / totals / diffs)
# - Log-transform nonnegative numeric features
# - Robust scaling + one-hot on grouped categories
# - Provide train/test splits for binary and multiclass SVM

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler


# ---------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

TRAIN_CSV = DATA_DIR / "training_data.csv"
FEATURES_CSV = DATA_DIR / "feature_description.csv"

BINARY_TARGET = "label"        # 0 normal, 1 attack (if present)
MULTI_TARGET = "attack_cat"    # attack category
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
                ["-", "?", "None", "none", "nan", ""], "unknown"
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
    if "proto" in df.columns:
        df["proto_group"] = df["proto"].map(_map_proto_group)
    if "state" in df.columns:
        df["state_group"] = df["state"].map(_map_state_group)
    if "service" in df.columns:
        df["service_group"] = df["service"].map(_map_service_group)
    return df


def _add_engineered_numeric_features(df: pd.DataFrame) -> pd.DataFrame:
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
    """
    Build:
      - is_attack (binary)
      - attack_cat_encoded (multiclass) if MULTI_TARGET present
    """
    # Binary target
    if BINARY_TARGET in df.columns:
        df["is_attack"] = df[BINARY_TARGET].astype(int)
    elif MULTI_TARGET in df.columns:
        df["is_attack"] = (df[MULTI_TARGET] != "Normal").astype(int)
    else:
        raise KeyError(
            f"Cannot create is_attack: neither '{BINARY_TARGET}' nor '{MULTI_TARGET}' found."
        )

    print("\nis_attack distribution:")
    print(df["is_attack"].value_counts())

    # Multiclass encoded target
    if MULTI_TARGET in df.columns:
        # Encode attack_cat to integer codes
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


# ---------------------------------------------------------------------
# Core loader + preprocessor
# ---------------------------------------------------------------------

def _load_and_preprocess():
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

    # Grouped categories
    df = _add_grouped_categories(df)

    # Targets
    df = _build_targets(df)

    # Engineered numeric features
    df = _add_engineered_numeric_features(df)

    # Decide which columns to drop outright
    drop_cols = []
    for cand in [
        "srcip",
        "dstip",
        "Stime",
        "Ltime",
        "stime",
        "ltime",
        "stime_ms",
        "proto",    # use proto_group instead
        "state",    # use state_group instead
        "service",  # use service_group instead
    ]:
        if cand in df.columns:
            drop_cols.append(cand)

    print("\nInitial drop_cols:", drop_cols)

    # Numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    exclude = ["is_attack", "attack_cat_encoded", BINARY_TARGET]
    numeric_cols = [c for c in numeric_cols if c not in exclude]

    print("\nNumeric candidate columns (sample):", numeric_cols[:20])

    # Categorical columns (grouped)
    cat_candidates = ["service_group", "proto_group", "state_group"]
    cat_cols = [c for c in cat_candidates if c in df.columns]
    print("\nCategorical columns to encode:", cat_cols)

    # Apply log1p to nonnegative numeric columns
    nonneg_numeric = [c for c in numeric_cols if df[c].min() >= 0]
    print("\nApplying log1p to nonnegative numeric features (count):", len(nonneg_numeric))

    for c in nonneg_numeric:
        df[c] = np.log1p(df[c])

    # Final features: numeric + categorical, excluding dropped cols
    features = [c for c in numeric_cols + cat_cols if c not in drop_cols]
    print("\nFinal features to be used (sample):", features[:40])

    # Build transformers
    num_features = [
        c for c in features if c in df.columns and np.issubdtype(df[c].dtype, np.number)
    ]
    cat_features = [c for c in features if c in df.columns and c not in num_features]

    # For linear svm
    num_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
        ]
    )

    # # For rbf svm
    # num_transformer = Pipeline([
    #     ("imputer", SimpleImputer(strategy="median")),
    #     ("scaler", StandardScaler()),
    # ])

    cat_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
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


# ---------------------------------------------------------------------
# Public API for SVM scripts
# ---------------------------------------------------------------------

def get_binary_splits(test_size: float = 0.20, random_state: int = 42):
    """
    Return X_train, X_test, y_train, y_test, preprocessor for binary SVM.
    """
    df, features, preprocessor = _load_and_preprocess()

    if "is_attack" not in df.columns:
        raise KeyError("Column 'is_attack' not found after preprocessing.")

    X = df[features].copy()
    y = df["is_attack"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
        shuffle=True,
    )

    print(f"\nBinary SVM: X_train {X_train.shape}, X_test {X_test.shape}")
    return X_train, X_test, y_train, y_test, preprocessor


def get_multiclass_splits(test_size: float = 0.20, random_state: int = 42):
    """
    Return X_train, X_test, y_train, y_test, preprocessor for multiclass SVM.
    Uses only rows where is_attack == 1 (attacks only).
    Target = attack_cat_encoded.
    """
    df, features, preprocessor = _load_and_preprocess()

    if "is_attack" not in df.columns:
        raise KeyError("Column 'is_attack' not found after preprocessing.")
    if "attack_cat_encoded" not in df.columns:
        raise KeyError("Column 'attack_cat_encoded' not found for multiclass splits.")

    df_attacks = df[df["is_attack"] == 1].copy()
    print(f"Total attack rows for multiclass: {df_attacks.shape[0]}")

    X = df_attacks[features].copy()
    y = df_attacks["attack_cat_encoded"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
        shuffle=True,
    )

    print(f"Multiclass SVM: X_train {X_train.shape}, X_test {X_test.shape}")
    return X_train, X_test, y_train, y_test, preprocessor


if __name__ == "__main__":
    # Quick smoke test
    Xtr, Xte, ytr, yte, pre = get_multiclass_splits()
    print("Done. Example shapes:")
    print("X_train:", Xtr.shape, "X_test:", Xte.shape)
    print("y_train:", ytr.shape, "y_test:", yte.shape)
