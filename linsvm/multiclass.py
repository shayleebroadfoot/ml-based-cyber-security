# linsvm/multiclass.py

import os
import time
import joblib

from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from .preprocess import get_multiclass_splits
from utils.metrics import print_multiclass_metrics
from sklearn.svm import SVC

SVM_MULTICLASS_MODEL_PATH = os.path.join("linsvm", "models", "svm_multiclass.joblib")


def train_multiclass_svm_model(model_path: str = SVM_MULTICLASS_MODEL_PATH):
    """
    Train a multiclass Linear SVM and save it to disk
    """
    X_train, X_test, y_train, y_test, preprocessor = get_multiclass_splits()

    svm = LinearSVC(
        C=1.0,                # tweak this if you want: 0.5, 1.0, 2.0...
        loss="squared_hinge",
        max_iter=4000,
        class_weight=None,    # keep precision-focused; avoid "balanced"
        random_state=42
    )

    # svm = SVC(
    #     kernel="rbf",
    #     C=1.0,
    #     gamma="scale",
    #     class_weight="balanced",  # often helpful with class imbalance
    #     decision_function_shape="ovr"  # default; one-vs-rest
    # )

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

    print("=== Multiclass Linear SVM: TRAINING ONLY (attacks subset) ===")
    print(f"Training time: {end - start:.3f} s")
    print(f"Saved model to: {model_path}")
    print(f"Model size: {size_kb:.1f} KB\n")


def test_multiclass_svm_model(model_path: str = SVM_MULTICLASS_MODEL_PATH):
    """
    Load the trained multiclass Linear SVM and evaluate it
    on the TEST split (attacks only).
    Also measures prediction time and peak memory during prediction.
    """
    import tracemalloc

    _, X_test, _, y_test, _ = get_multiclass_splits()

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Multiclass SVM model not found at: {model_path}")

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

    print("=== Multiclass Linear SVM: TEST RESULTS (held-out test set) ===")
    print_multiclass_metrics(y_test, y_pred)
    print(f"\n[Test] prediction time: {total_time:.3f} s "
          f"({per_sample_ms:.4f} ms per sample)")
    print(f"[Test] prediction peak memory: {peak_mb:.3f} MB\n")
