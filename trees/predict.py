# trees/predict.py

import joblib

from utils.load_data import load_data
from trees.preprocess import split_features_and_labels, encode_features
from utils.metrics import print_binary_metrics, print_multiclass_metrics


def _prepare_features(df):
    """
    Common preprocessing for both binary and multiclass:
    - drop id (already done in load_data)
    - split into X_raw, y_binary, y_multiclass
    - one-hot encode X_raw

    Returns: X, y_binary, y_multiclass
    """
    X_raw, y_binary, y_multiclass = split_features_and_labels(df)
    X = encode_features(X_raw)
    return X, y_binary, y_multiclass


# ===================== BINARY PREDICTION =====================

def load_binary_model(model_path: str = "trees/models/rf_binary.joblib"):
    return joblib.load(model_path)


def predict_binary(
    data_path: str = "data/training_data.csv",
    model_path: str = "trees/models/rf_binary.joblib",
    evaluate: bool = True
):
    """
    Predicts binary label (0 = Normal, 1 = Attack) using the saved RF model.
    If evaluate=True, also prints metrics assuming 'label' column is present.
    """
    # Load data and preprocess
    df = load_data(data_path)
    X, y_binary, _ = _prepare_features(df)

    # Load model
    model = load_binary_model(model_path)

    # Predict
    y_pred = model.predict(X)

    if evaluate and y_binary is not None:
        print_binary_metrics(y_binary, y_pred, title="Binary Prediction (Loaded Model)")

    return y_pred


# ===================== MULTICLASS PREDICTION =====================

def load_multiclass_model(model_path: str = "trees/models/rf_multiclass.joblib"):
    return joblib.load(model_path)


def predict_multiclass(
    data_path: str = "data/training_data.csv",
    model_path: str = "trees/models/rf_multiclass.joblib",
    evaluate: bool = True
):
    """
    Predicts attack category ('attack_cat') using the saved RF model.
    If evaluate=True, also prints metrics assuming 'attack_cat' column is present.
    """
    # Load data and preprocess
    df = load_data(data_path)
    X, _, y_multiclass = _prepare_features(df)

    # Load model
    model = load_multiclass_model(model_path)

    # Predict
    y_pred = model.predict(X)

    if evaluate and y_multiclass is not None:
        print_multiclass_metrics(
            y_multiclass,
            y_pred,
            title="Multiclass Prediction (Loaded Model)"
        )

    return y_pred
