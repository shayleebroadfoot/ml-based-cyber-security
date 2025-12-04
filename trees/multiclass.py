
# trees/multiclass.py

import os
import time
import joblib
import tracemalloc
import math
import numpy as np

from sklearn.ensemble import RandomForestClassifier

from trees.preprocess import get_multiclass_splits
from utils.metrics import print_multiclass_metrics
from sklearn.metrics import mean_squared_error


MULTICLASS_MODEL_PATH = os.path.join("trees", "models", "rf_multiclass.joblib")


def train_multiclass_model(model_path: str = MULTICLASS_MODEL_PATH):
    """
    Train the multiclass Random Forest model on the TRAIN split only
    and save it to disk. No testing or prediction happens here.
    """
    X_train, X_test, y_train, y_test = get_multiclass_splits()

    rf = RandomForestClassifier(
        n_estimators=60, #480, 50, 60
        max_depth=12, #10
        min_samples_split=8, #13
        min_samples_leaf=4, #2
        max_features=0.2, #0.2
        class_weight=None, #None
        n_jobs=-1,
        bootstrap=True,
        random_state=42
    )


    start = time.perf_counter()
    rf.fit(X_train, y_train)
    end = time.perf_counter()

    # TRAIN RMSE on numerically encoded labels
    y_train_pred = rf.predict(X_train)
    class_to_int = {label: idx for idx, label in enumerate(rf.classes_)}
    y_train_int = np.array([class_to_int[label] for label in y_train])
    y_train_pred_int = np.array([class_to_int[label] for label in y_train_pred])
    train_rmse = math.sqrt(mean_squared_error(y_train_int, y_train_pred_int))
    print(f"Train RMSE (encoded labels): {train_rmse:.4f}")

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(rf, model_path)

    size_bytes = os.path.getsize(model_path)
    size_kb = size_bytes / 1024.0

    print("=== Multiclass Random Forest: TRAINING ONLY ===")
    print(f"Training time: {end - start:.3f} s")
    print(f"Saved model to: {model_path}")
    print(f"Model size: {size_kb:.1f} KB\n")


def test_multiclass_model(model_path: str = MULTICLASS_MODEL_PATH):
    """
    Load the trained multiclass model and evaluate it on the TEST split only.
    This is the ONLY place where testing/prediction on the test set happens.
    """
    X_train, X_test, y_train, y_test = get_multiclass_splits()

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Multiclass model not found at: {model_path}")

    tracemalloc.start()
    rf = joblib.load(model_path)

    # tracemalloc.start()
    start = time.perf_counter()
    y_pred = rf.predict(X_test)
    end = time.perf_counter()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # TEST RMSE on numerically encoded labels
    class_to_int = {label: idx for idx, label in enumerate(rf.classes_)}
    y_test_int = np.array([class_to_int[label] for label in y_test])
    y_pred_int = np.array([class_to_int[label] for label in y_pred])

    test_rmse = math.sqrt(mean_squared_error(y_test_int, y_pred_int))
    print(f"Test RMSE (encoded labels): {test_rmse:.4f}")

    total_time = end - start
    per_sample_ms = (total_time / len(X_test)) * 1000.0
    peak_mb = peak / (1024 ** 2)

    print("=== Multiclass Random Forest: TEST RESULTS (held-out test set) ===")
    print_multiclass_metrics(y_test, y_pred)
    print(f"\n[Test] prediction time: {total_time:.3f} s "
          f"({per_sample_ms:.4f} ms per sample)")
    print(f"[Test] prediction peak memory: {peak_mb:.3f} MB\n")