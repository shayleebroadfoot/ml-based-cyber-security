# trees/multiclass_tree.py

import os
import time
import joblib
from sklearn.tree import DecisionTreeClassifier

from trees.preprocess import get_multiclass_splits
from utils.metrics import print_multiclass_metrics

TREE_MULTICLASS_MODEL_PATH = os.path.join("trees", "models", "tree_multiclass.joblib")


def train_multiclass_tree_model(model_path: str = TREE_MULTICLASS_MODEL_PATH):
    """
    Train a multiclass Decision Tree on the TRAIN split only
    and save it to disk. No testing happens here.
    """
    X_train, X_test, y_train, y_test = get_multiclass_splits()

    tree = DecisionTreeClassifier(
        criterion="entropy",
        max_depth=18,
        min_samples_split=5,
        min_samples_leaf=4,
        class_weight=None,
        ccp_alpha=0.0001,
        random_state=42
    )

    start = time.perf_counter()
    tree.fit(X_train, y_train)
    end = time.perf_counter()

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(tree, model_path)

    size_bytes = os.path.getsize(model_path)
    size_kb = size_bytes / 1024.0

    print("=== Multiclass Decision Tree: TRAINING ONLY ===")
    print(f"Training time: {end - start:.3f} s")
    print(f"Saved model to: {model_path}")
    print(f"Model size: {size_kb:.1f} KB\n")


def test_multiclass_tree_model(model_path: str = TREE_MULTICLASS_MODEL_PATH):
    """
    Load the trained multiclass Decision Tree and evaluate it
    on the TEST split only. Also measures prediction time and
    peak memory during prediction.
    """
    import tracemalloc

    X_train, X_test, y_train, y_test = get_multiclass_splits()

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Multiclass tree model not found at: {model_path}")

    tree = joblib.load(model_path)

    tracemalloc.start()
    start = time.perf_counter()
    y_pred = tree.predict(X_test)
    end = time.perf_counter()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    total_time = end - start
    per_sample_ms = (total_time / len(X_test)) * 1000.0
    peak_mb = peak / (1024 ** 2)

    print("=== Multiclass Decision Tree: TEST RESULTS (held-out test set) ===")
    print_multiclass_metrics(y_test, y_pred)
    print(f"\n[Test] prediction time: {total_time:.3f} s "
          f"({per_sample_ms:.4f} ms per sample)")
    print(f"[Test] prediction peak memory: {peak_mb:.3f} MB\n")
