# kernelsvm/multiclass.py
#
# Multiclass SVM with nonlinear kernels (RBF / poly) for attacks only.
# Uses kernelsvm.preprocess for feature engineering and scaling.

import os
import time
import joblib

from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import mean_squared_error

from .preprocess import get_multiclass_splits
from utils.metrics import print_multiclass_metrics


RBF_MULTICLASS_MODEL_PATH = os.path.join("kernelsvm", "models", "rbf_multiclass.joblib")


def train_multiclass_kernel_svm_model(model_path: str = RBF_MULTICLASS_MODEL_PATH):
    """
    Train a multiclass kernel SVM (attacks only) and save it to disk.
    Default is RBF kernel.
    """
    X_train, X_test, y_train, y_test, preprocessor = get_multiclass_splits()

    svm = SVC(
        kernel="rbf",
        C=1, #1
        gamma=0.075,
        class_weight=None,
        decision_function_shape="ovr",
        probability=False,
        random_state=42
    )

    model = Pipeline(
        steps=[
            ("preproc", preprocessor),
            ("clf", svm),
        ]
    )

    start = time.perf_counter()
    model.fit(X_train, y_train)
    end = time.perf_counter()

    # Train RMSE
    y_train_pred = model.predict(X_train)
    mse_train = mean_squared_error(y_train, y_train_pred)
    rmse_train = mse_train ** 0.5

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)

    size_bytes = os.path.getsize(model_path)
    size_kb = size_bytes / 1024.0

    print("=== Multiclass Kernel SVM (RBF): TRAINING ONLY (attacks subset) ===")
    print(f"Training time: {end - start:.3f} s")
    print(f"Saved model to: {model_path}")
    print(f"Model size: {size_kb:.1f} KB")
    print(f"Train RMSE: {rmse_train:.4f}\n")


def test_multiclass_kernel_svm_model(model_path: str = RBF_MULTICLASS_MODEL_PATH):
    """
    Load the trained multiclass kernel SVM and evaluate it
    on the TEST split (attacks only).
    Measures prediction time and RMSE.
    """
    import tracemalloc

    _, X_test, _, y_test, _ = get_multiclass_splits()

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Multiclass kernel SVM model not found at: {model_path}")

    tracemalloc.start()
    model = joblib.load(model_path)

    start = time.perf_counter()
    y_pred = model.predict(X_test)
    end = time.perf_counter()

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    total_time = end - start
    per_sample_ms = (total_time / len(X_test)) * 1000.0
    peak_mb = peak / (1024 ** 2)

    mse_test = mean_squared_error(y_test, y_pred)
    rmse_test = mse_test ** 0.5

    print("=== Multiclass Kernel SVM (RBF): TEST RESULTS (held-out test set) ===")
    print_multiclass_metrics(y_test, y_pred)
    print(f"Test RMSE: {rmse_test:.4f}")
    print(f"\n[Test] prediction time: {total_time:.3f} s "
          f"({per_sample_ms:.4f} ms per sample)")
    print(f"[Test] prediction peak memory: {peak_mb:.3f} MB\n")

if __name__ == "__main__":
    train_multiclass_kernel_svm_model()
    test_multiclass_kernel_svm_model()