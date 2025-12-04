# trees/binary.py

import os
import time
import joblib
import tracemalloc
import math

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error

from trees.preprocess import get_binary_splits
from utils.metrics import print_binary_metrics

BINARY_MODEL_PATH = os.path.join("trees", "models", "rf_binary.joblib")


def train_binary_model(model_path: str = BINARY_MODEL_PATH):
    """
    Train the binary Random Forest model on the TRAIN split only
    and save it to disk. No testing or prediction happens here.
    """
    X_train, X_test, y_train, y_test = get_binary_splits()

    rf = RandomForestClassifier(
        n_estimators=20,
        max_depth=16,
        min_samples_split=10,
        min_samples_leaf=5,
        max_features=0.35,
        class_weight="balanced",
        n_jobs=-1,
        bootstrap=False,
        random_state=42
    )

    start = time.perf_counter()
    rf.fit(X_train, y_train)
    end = time.perf_counter()

    # TRAIN RMSE (0/1 labels are already numeric)
    y_train_pred = rf.predict(X_train)
    train_rmse = math.sqrt(mean_squared_error(y_train, y_train_pred))
    print(f"Train RMSE (binary labels): {train_rmse:.4f}")

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(rf, model_path)

    size_bytes = os.path.getsize(model_path)
    size_kb = size_bytes / 1024.0

    print("=== Binary Random Forest: TRAINING ONLY ===")
    print(f"Training time: {end - start:.3f} s")
    print(f"Saved model to: {model_path}")
    print(f"Model size: {size_kb:.1f} KB\n")


def test_binary_model(model_path: str = BINARY_MODEL_PATH):
    """
    Load the trained binary model and evaluate it on the TEST split only.
    This is the ONLY place where testing/prediction on the test set happens.
    """
    X_train, X_test, y_train, y_test = get_binary_splits()

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Binary model not found at: {model_path}")

    tracemalloc.start()
    rf = joblib.load(model_path)

    start = time.perf_counter()
    y_pred = rf.predict(X_test)
    end = time.perf_counter()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # TEST RMSE (binary labels)
    test_rmse = math.sqrt(mean_squared_error(y_test, y_pred))
    print(f"Test RMSE (binary labels): {test_rmse:.4f}")

    total_time = end - start
    per_sample_ms = (total_time / len(X_test)) * 1000.0
    peak_mb = peak / (1024 ** 2)

    print("=== Binary Random Forest: TEST RESULTS (held-out test set) ===")
    print_binary_metrics(y_test, y_pred)
    print(f"\n[Test] prediction time: {total_time:.3f} s "
          f"({per_sample_ms:.4f} ms per sample)")
    print(f"[Test] prediction peak memory: {peak_mb:.3f} MB\n")

# # trees/binary.py
#
# import os
# import time
# import joblib
# import tracemalloc
#
# from sklearn.ensemble import RandomForestClassifier
#
# from trees.preprocess import get_binary_splits
# from utils.metrics import print_binary_metrics
#
# BINARY_MODEL_PATH = os.path.join("trees", "models", "rf_binary.joblib")
#
#
# def train_binary_model(model_path: str = BINARY_MODEL_PATH):
#     """
#     Train the binary Random Forest model on the TRAIN split only
#     and save it to disk. No testing or prediction happens here.
#     """
#     X_train, X_test, y_train, y_test = get_binary_splits()
#
#     rf = RandomForestClassifier(
#         n_estimators=40,
#         max_depth=10,
#         min_samples_split=13,  # 13
#         min_samples_leaf=5, # Try 2
#         max_features=0.2, #Try 0.2, sqrt
#         class_weight="balanced", #Try None, balanced
#         # criterion="entropy",
#         n_jobs=-1,
#         bootstrap=True,
#         random_state=42
#     )
#
#     start = time.perf_counter()
#     rf.fit(X_train, y_train)
#     end = time.perf_counter()
#
#     os.makedirs(os.path.dirname(model_path), exist_ok=True)
#     joblib.dump(rf, model_path)
#
#     size_bytes = os.path.getsize(model_path)
#     size_kb = size_bytes / 1024.0
#
#     print("=== Binary Random Forest: TRAINING ONLY ===")
#     print(f"Training time: {end - start:.3f} s")
#     print(f"Saved model to: {model_path}")
#     print(f"Model size: {size_kb:.1f} KB\n")
#
#
# def test_binary_model(model_path: str = BINARY_MODEL_PATH):
#     """
#     Load the trained binary model and evaluate it on the TEST split only.
#     This is the ONLY place where testing/prediction on the test set happens.
#     """
#     X_train, X_test, y_train, y_test = get_binary_splits()
#
#     if not os.path.exists(model_path):
#         raise FileNotFoundError(f"Binary model not found at: {model_path}")
#
#     rf = joblib.load(model_path)
#
#     tracemalloc.start()
#     start = time.perf_counter()
#     y_pred = rf.predict(X_test)
#     end = time.perf_counter()
#     current, peak = tracemalloc.get_traced_memory()
#     tracemalloc.stop()
#
#     total_time = end - start
#     per_sample_ms = (total_time / len(X_test)) * 1000.0
#     peak_mb = peak / (1024 ** 2)
#
#     print("=== Binary Random Forest: TEST RESULTS (held-out test set) ===")
#     print_binary_metrics(y_test, y_pred)
#     print(f"\n[Test] prediction time: {total_time:.3f} s "
#           f"({per_sample_ms:.4f} ms per sample)")
#     print(f"[Test] prediction peak memory: {peak_mb:.3f} MB\n")