# trees/multiclass.py

import os
import time
import joblib
import tracemalloc

from sklearn.ensemble import RandomForestClassifier

from trees.preprocess import get_multiclass_splits
from utils.metrics import print_multiclass_metrics

MULTICLASS_MODEL_PATH = os.path.join("trees", "models", "rf_multiclass.joblib")


def train_multiclass_model(model_path: str = MULTICLASS_MODEL_PATH):
    """
    Train the multiclass Random Forest model on the TRAIN split only
    and save it to disk. No testing or prediction happens here.
    """
    X_train, X_test, y_train, y_test = get_multiclass_splits()

    rf = RandomForestClassifier(
        n_estimators=60,  # same
        max_depth=22,  # same
        min_samples_split=2,  # new
        min_samples_leaf=1,  # higher -> smoother, fewer noisy leaves
        max_features=0.5,  # fewer features per split, more diverse trees
        class_weight="balanced",  # softer than balanced_subsample
        criterion="entropy",  # try different split criterion
        n_jobs=-1,
        random_state=42
    )

    start = time.perf_counter()
    rf.fit(X_train, y_train)
    end = time.perf_counter()

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

    rf = joblib.load(model_path)

    tracemalloc.start()
    start = time.perf_counter()
    y_pred = rf.predict(X_test)
    end = time.perf_counter()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    total_time = end - start
    per_sample_ms = (total_time / len(X_test)) * 1000.0
    peak_mb = peak / (1024 ** 2)

    print("=== Multiclass Random Forest: TEST RESULTS (held-out test set) ===")
    print_multiclass_metrics(y_test, y_pred)
    print(f"\n[Test] prediction time: {total_time:.3f} s "
          f"({per_sample_ms:.4f} ms per sample)")
    print(f"[Test] prediction peak memory: {peak_mb:.3f} MB\n")