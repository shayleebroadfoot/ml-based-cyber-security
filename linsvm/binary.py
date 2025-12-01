# linsvm/binary.py

import os
import time
import joblib

from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from .preprocess import get_binary_splits
from utils.metrics import print_binary_metrics

SVM_BINARY_MODEL_PATH = os.path.join("linsvm", "models", "svm_binary.joblib")


def train_binary_svm_model(model_path: str = SVM_BINARY_MODEL_PATH):
    """
    Train a binary Linear SVM (attack vs normal) on the TRAIN split
    and save it to disk. No testing happens here.
    """
    X_train, X_test, y_train, y_test, preprocessor = get_binary_splits()

    svm = LinearSVC(
        C=1.0,                # you can try 0.5 or 2.0 later
        loss="squared_hinge",
        max_iter=2000,
        class_weight=None,    # keep precision-focused; avoid "balanced"
        random_state=42
    )

    model = Pipeline(
        steps=[
            ("preproc", preprocessor),
            ("clf", svm)
        ]
    )

    start = time.perf_counter()
    model.fit(X_train, y_train)
    end = time.perf_counter()

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)

    size_bytes = os.path.getsize(model_path)
    size_kb = size_bytes / 1024.0

    print("=== Binary Linear SVM: TRAINING ONLY (attack vs normal) ===")
    print(f"Training time: {end - start:.3f} s")
    print(f"Saved model to: {model_path}")
    print(f"Model size: {size_kb:.1f} KB\n")


def test_binary_svm_model(model_path: str = SVM_BINARY_MODEL_PATH):
    """
    Load the trained binary Linear SVM and evaluate it
    on the TEST split (attack vs normal).
    Also measures prediction time and peak memory during prediction.
    """
    import tracemalloc

    _, X_test, _, y_test, _ = get_binary_splits()

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Binary SVM model not found at: {model_path}")

    model = joblib.load(model_path)

    tracemalloc.start()
    start = time.perf_counter()
    y_pred = model.predict(X_test)
    end = time.perf_counter()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    total_time = end - start
    per_sample_ms = (total_time / len(X_test)) * 1000.0
    peak_mb = peak / (1024 ** 2)

    print("=== Binary Linear SVM: TEST RESULTS (held-out test set) ===")
    print_binary_metrics(y_test, y_pred)
    print(f"\n[Test] prediction time: {total_time:.3f} s "
          f"({per_sample_ms:.4f} ms per sample)")
    print(f"[Test] prediction peak memory: {peak_mb:.3f} MB\n")
