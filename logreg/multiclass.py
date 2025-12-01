# logreg/multiclass.py

import os
import time
import joblib

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from logreg.preprocess import get_multiclass_splits
from utils.metrics import print_multiclass_metrics

LOGREG_MULTICLASS_MODEL_PATH = os.path.join("logreg", "models", "logreg_multiclass.joblib")


def train_multiclass_logreg_model(model_path: str = LOGREG_MULTICLASS_MODEL_PATH):
    """
    Train a multiclass Logistic Regression (attacks only) on the TRAIN split
    and save it to disk. No testing happens here.
    """
    X_train, X_test, y_train, y_test, preprocessor = get_multiclass_splits()

    # clf = LogisticRegression(
    #     max_iter=2000,
    #     solver="saga",
    #     class_weight="balanced",
    #     multi_class="ovr",
    #     n_jobs=-1,
    #     random_state=42,
    #     verbose=0,
    #     # C=0.5,  # optional: stronger regularization if you want to try later
    # )
    logreg = LogisticRegression(
        penalty="l2",
        C=0.5,  # strong-ish regularization → smoother, less overfitting
        solver="liblinear",  # good for OvR small-ish feature set
        multi_class="ovr",  # one-vs-rest: more conservative per class
        max_iter=300,
        class_weight=None  # DO NOT use 'balanced' – that boosts rare classes and hurts precision
        # n_jobs is not supported for liblinear, so leave it out
    )

    model = Pipeline(
        steps=[
            ("preproc", preprocessor),
            ("clf", logreg),
        ]
    )

    start = time.perf_counter()
    model.fit(X_train, y_train)
    end = time.perf_counter()

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)

    size_bytes = os.path.getsize(model_path)
    size_kb = size_bytes / 1024.0

    print("=== Multiclass Logistic Regression: TRAINING ONLY (attacks subset) ===")
    print(f"Training time: {end - start:.3f} s")
    print(f"Saved model to: {model_path}")
    print(f"Model size: {size_kb:.1f} KB\n")


def test_multiclass_logreg_model(model_path: str = LOGREG_MULTICLASS_MODEL_PATH):
    """
    Load the trained multiclass Logistic Regression and evaluate it
    on the TEST split (attacks only). Also measures prediction time and
    peak memory during prediction.
    """
    import tracemalloc

    # We don't actually need the preprocessor here, since it's baked into the Pipeline
    _, X_test, _, y_test, _ = get_multiclass_splits()

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Multiclass logistic model not found at: {model_path}")

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

    print("=== Multiclass Logistic Regression: TEST RESULTS (held-out test set) ===")
    print_multiclass_metrics(y_test, y_pred)
    print(f"\n[Test] prediction time: {total_time:.3f} s "
          f"({per_sample_ms:.4f} ms per sample)")
    print(f"[Test] prediction peak memory: {peak_mb:.3f} MB\n")
