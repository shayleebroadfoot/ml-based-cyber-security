# trees/multiclass.py

import os
import time
import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from trees.preprocess import get_multiclass_splits
from utils.metrics import print_multiclass_metrics


def train_multiclass_models():
    X_train, X_test, y_train, y_test = get_multiclass_splits()

    # ----------------- Baseline Decision Tree -----------------
    tree_multi = DecisionTreeClassifier(
        max_depth=15,
        min_samples_leaf=5,
        random_state=42
    )

    start_tree = time.perf_counter()
    tree_multi.fit(X_train, y_train)
    end_tree = time.perf_counter()

    y_pred_tree = tree_multi.predict(X_test)

    print_multiclass_metrics(y_test, y_pred_tree, title="Decision Tree (Multiclass)")
    print(f"Decision Tree training time: {end_tree - start_tree:.3f} s\n")

    # ----------------- MOBILE-FRIENDLY Random Forest -----------------
    # Previous heavier model:
    # forest_multi = RandomForestClassifier(
    #     n_estimators=400,
    #     max_depth=25,
    #     min_samples_split=10,
    #     min_samples_leaf=1,
    #     max_features=0.5,
    #     class_weight=None,
    #     n_jobs=-1,
    #     bootstrap=False,
    #     random_state=42
    # )

    # New lighter RF:
    forest_multi = RandomForestClassifier(
        n_estimators=60,
        max_depth=12,
        min_samples_leaf=5,
        max_features="sqrt",
        n_jobs=-1,
        random_state=42
    )

    start_rf = time.perf_counter()
    forest_multi.fit(X_train, y_train)
    end_rf = time.perf_counter()

    y_pred_rf = forest_multi.predict(X_test)

    print_multiclass_metrics(y_test, y_pred_rf, title="Random Forest (Multiclass, mobile-friendly)")
    print(f"Random Forest training time: {end_rf - start_rf:.3f} s")

    # Save RF model and report file size
    os.makedirs("trees/models", exist_ok=True)
    model_path = os.path.join("trees", "models", "rf_multiclass.joblib")
    joblib.dump(forest_multi, model_path)

    size_bytes = os.path.getsize(model_path)
    size_kb = size_bytes / 1024.0

    print(f"\nSaved multiclass RF model to: {model_path}")
    print(f"Multiclass RF model size: {size_kb:.1f} KB\n")

    return forest_multi
