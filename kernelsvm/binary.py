# kernelsvm/binary.py

import os
import time
import tracemalloc

import joblib
import numpy as np

from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    mean_squared_error,
)

from kernelsvm.preprocess import get_binary_splits

BINARY_MODEL_PATH = os.path.join("kernelsvm", "models", "svm_binary_rbf.joblib")


def _print_header(title: str):
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def train_binary_model(model_path: str = BINARY_MODEL_PATH):
    """
    Train the binary RBF-kernel SVM (Normal vs Attack) on the TRAIN split only
    and save it to disk. No testing or prediction on the test set happens here.
    """
    _print_header("=== Binary RBF SVM: TRAINING ONLY ===")

    # Split once, but only use the TRAIN part for fitting and metrics
    X_train, X_test, y_train, y_test, preprocessor = get_binary_splits()

    clf = Pipeline(
        steps=[
            ("pre", preprocessor),
            ("svm", SVC(
                kernel="rbf",
                C=9.0,
                gamma=0.05,
                class_weight='balanced',
                decision_function_shape="ovr",
                probability=False,
            )),
        ]
    )

    start = time.time()
    clf.fit(X_train, y_train)
    train_time = time.time() - start

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(clf, model_path)

    model_size_mb = os.path.getsize(model_path) / (1024 * 1024)

    # TRAIN metrics only (no use of X_test / y_test)
    y_train_pred = clf.predict(X_train)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    train_acc = accuracy_score(y_train, y_train_pred)

    print(f"Training time: {train_time:.3f} s")
    print(f"Saved model to: {model_path}")
    print(f"Model size: {model_size_mb:.1f} MB")
    print(f"Train Accuracy : {train_acc:.6f}")
    print(f"Train RMSE     : {train_rmse:.4f}\n")


def test_binary_model(model_path: str = BINARY_MODEL_PATH):
    """
    Load the trained binary RBF SVM pipeline and evaluate it on the TEST split only.
    This is the ONLY place where testing/prediction on the test set happens.
    """
    _print_header("=== Binary RBF SVM: TEST RESULTS (held-out test set) ===")

    if not os.path.exists(model_path):
        print(f"[ERROR] Model file not found at: {model_path}")
        print("Run train_binary_model() first.")
        return

    # Same split as training; test metrics only use X_test / y_test
    X_train, X_test, y_train, y_test, _ = get_binary_splits()

    tracemalloc.start()
    clf = joblib.load(model_path)

    start = time.time()
    y_test_pred = clf.predict(X_test)
    pred_time = time.time() - start

    current, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    peak_mb = peak_bytes / (1024 * 1024)

    # TEST metrics
    acc = accuracy_score(y_test, y_test_pred)
    prec = precision_score(y_test, y_test_pred, zero_division=0)
    rec = recall_score(y_test, y_test_pred, zero_division=0)
    f1 = f1_score(y_test, y_test_pred, zero_division=0)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

    print(f"Prediction time (test only): {pred_time:.3f} s")
    print(f"[Test] prediction peak memory: {peak_mb:.3f} MB\n")

    print(f"Accuracy           : {acc:.6f}")
    print(f"Precision (binary) : {prec:.6f}")
    print(f"Recall (binary)    : {rec:.6f}")
    print(f"F1 Score (binary)  : {f1:.6f}")
    print(f"Test  RMSE         : {test_rmse:.4f}")
    print()
    print("Confusion Matrix (test):")
    print(confusion_matrix(y_test, y_test_pred))




if __name__ == "__main__":
    train_binary_model()
    test_binary_model()
