# trees/binary.py

import os
import time
import joblib

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from trees.preprocess import get_binary_splits
from utils.metrics import print_binary_metrics


def train_binary_models():
    X_train, X_test, y_train, y_test = get_binary_splits()

    # Decision Tree baseline
    tree_bin = DecisionTreeClassifier(
        max_depth=10,
        min_samples_leaf=5,
        random_state=42
    )
    start_tree = time.perf_counter()
    tree_bin.fit(X_train, y_train)
    end_tree = time.perf_counter()

    y_pred_tree = tree_bin.predict(X_test)

    print_binary_metrics(y_test, y_pred_tree, title="Decision Tree (Binary)")
    print(f"Decision Tree training time: {end_tree - start_tree:.3f} s\n")

    # ----------------- MOBILE-FRIENDLY Random Forest -----------------
    # Previous heavier model:
    # forest_bin = RandomForestClassifier(
    #     n_estimators=200,
    #     max_depth=20,
    #     min_samples_leaf=3,
    #     max_features=None,
    #     n_jobs=-1,
    #     random_state=42
    # )

    # New lighter RF:
    # - fewer trees (n_estimators)
    # - shallower trees (max_depth)
    # - more regularization (min_samples_leaf)
    forest_bin = RandomForestClassifier(
        n_estimators=40,
        max_depth=10,
        min_samples_leaf=5,
        max_features="sqrt",
        n_jobs=-1,
        random_state=42
    )

    start_rf = time.perf_counter()
    forest_bin.fit(X_train, y_train)
    end_rf = time.perf_counter()

    y_pred_rf = forest_bin.predict(X_test)

    print_binary_metrics(y_test, y_pred_rf, title="Random Forest (Binary, mobile-friendly)")
    print(f"Random Forest training time: {end_rf - start_rf:.3f} s")

    # Save RF model and report file size
    os.makedirs("trees/models", exist_ok=True)
    model_path = os.path.join("trees", "models", "rf_binary.joblib")
    joblib.dump(forest_bin, model_path)

    size_bytes = os.path.getsize(model_path)
    size_kb = size_bytes / 1024.0

    print(f"Saved binary RF model to: {model_path}")
    print(f"Binary RF model size: {size_kb:.1f} KB\n")

    return forest_bin