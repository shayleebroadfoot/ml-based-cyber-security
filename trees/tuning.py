# trees/tuning.py

import os
import time
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier

from trees.preprocess import get_binary_splits, get_multiclass_splits
from utils.metrics import print_binary_metrics, print_multiclass_metrics


# =====================================================================
# BINARY RF TUNING (optional – you can ignore if you are happy with binary)
# =====================================================================

def tune_binary_rf(random_state: int = 42):
    """
    Tune the binary Random Forest.
    - We DO search over n_estimators and max_depth (your grid).
    - Multi-metric scoring (accuracy, precision_macro, recall_macro, f1_macro),
      but we refit on precision_macro to reduce false alarms.
    """
    X_train, X_test, y_train, y_test = get_binary_splits()

    base_rf = RandomForestClassifier(
        n_jobs=-1,
        random_state=random_state,
        class_weight="balanced",
        criterion="entropy",
    )

    # YOUR grid, including n_estimators + max_depth
    param_distributions = {
        "min_samples_leaf": [2, 4, 6, 8],
        "min_samples_split": [2, 4, 6, 8],
        "n_estimators": [30, 40, 50, 60],
        "max_depth": [20, 30, 40, 50],
        "max_features": ["sqrt", "log2", 0.3, 0.5],
    }

    scoring = {
        "accuracy": "accuracy",
        "precision_macro": "precision_macro",
        "recall_macro": "recall_macro",
        "f1_macro": "f1_macro",
    }

    tuner = RandomizedSearchCV(
        estimator=base_rf,
        param_distributions=param_distributions,
        n_iter=20,
        scoring=scoring,
        refit="precision_macro",   # choose best by macro precision
        cv=3,
        n_jobs=-1,
        random_state=random_state,
        verbose=1,
    )

    print("=== Tuning binary Random Forest (multi-metric, n_estimators & max_depth included) ===")
    start = time.perf_counter()
    tuner.fit(X_train, y_train)
    end = time.perf_counter()

    print(f"\nBinary RF tuning time: {end - start:.3f} s")
    print("Best parameters (binary RF):")
    print(tuner.best_params_)

    best_idx = tuner.best_index_

    # CV metrics for the BEST param combo
    print("\n=== CV results for BEST binary RF params ===")
    print(f"CV Accuracy (mean):        {tuner.cv_results_['mean_test_accuracy'][best_idx]:.4f}")
    print(f"CV Precision_macro (mean): {tuner.cv_results_['mean_test_precision_macro'][best_idx]:.4f}")
    print(f"CV Recall_macro (mean):    {tuner.cv_results_['mean_test_recall_macro'][best_idx]:.4f}")
    print(f"CV F1_macro (mean):        {tuner.cv_results_['mean_test_f1_macro'][best_idx]:.4f}")

    # Test-set metrics
    best_rf = tuner.best_estimator_
    y_pred = best_rf.predict(X_test)

    print("\n=== Best binary RF on held-out test set ===")
    print_binary_metrics(y_test, y_pred)

    return best_rf, tuner.best_params_


# =====================================================================
# MULTICLASS RF TUNING (main one you care about – optimize macro precision)
# =====================================================================

def tune_multiclass_rf(random_state: int = 42, save_model: bool = False):
    """
    Tune the multiclass Random Forest.
    We search over:
      - n_estimators (number of trees)
      - max_depth
      - min_samples_leaf
      - min_samples_split
      - max_features
      - bootstrap

    Multi-metric scoring is used (accuracy, precision_macro, recall_macro, f1_macro),
    but the best model is chosen based on precision_macro to reduce false alarms.
    """
    X_train, X_test, y_train, y_test = get_multiclass_splits()

    base_rf = RandomForestClassifier(
        criterion="entropy",
        class_weight="balanced",
        n_jobs=-1,
        random_state=random_state,
    )

    # Reasonable ranges so the model doesn't get absurdly huge
    param_distributions = {
        "n_estimators": [40, 60, 80, 100],      # includes your 80
        "max_depth": [10, 14, 18, 22],          # includes your 14
        "min_samples_leaf": [1, 2, 5, 10],
        "min_samples_split": [2, 5, 10, 20],
        "max_features": ["sqrt", "log2", 0.3, 0.5, 0.7],
        "bootstrap": [True, False],
    }

    scoring = {
        "accuracy": "accuracy",
        "precision_macro": "precision_macro",
        "recall_macro": "recall_macro",
        "f1_macro": "f1_macro",
    }

    tuner = RandomizedSearchCV(
        estimator=base_rf,
        param_distributions=param_distributions,
        n_iter=25,
        scoring=scoring,
        refit="precision_macro",   # choose best by macro precision
        cv=3,
        n_jobs=-1,
        random_state=random_state,
        verbose=1,
    )

    print("=== Tuning multiclass Random Forest (multi-metric, n_estimators & max_depth INCLUDED) ===")
    start = time.perf_counter()
    tuner.fit(X_train, y_train)
    end = time.perf_counter()

    print(f"\nMulticlass RF tuning time: {end - start:.3f} s")
    print("Best parameters (multiclass RF):")
    print(tuner.best_params_)

    best_idx = tuner.best_index_

    print("\n=== CV results for BEST multiclass RF params ===")
    print(f"CV Accuracy (mean):        {tuner.cv_results_['mean_test_accuracy'][best_idx]:.4f}")
    print(f"CV Precision_macro (mean): {tuner.cv_results_['mean_test_precision_macro'][best_idx]:.4f}")
    print(f"CV Recall_macro (mean):    {tuner.cv_results_['mean_test_recall_macro'][best_idx]:.4f}")
    print(f"CV F1_macro (mean):        {tuner.cv_results_['mean_test_f1_macro'][best_idx]:.4f}")

    # Test-set evaluation
    best_rf = tuner.best_estimator_
    y_pred = best_rf.predict(X_test)

    print("\n=== Best multiclass RF on held-out test set ===")
    print_multiclass_metrics(y_test, y_pred)

    model_path = None
    if save_model:
        os.makedirs("trees/models", exist_ok=True)
        model_path = os.path.join("trees", "models", "rf_multiclass_tuned.joblib")
        joblib.dump(best_rf, model_path)

        size_kb = os.path.getsize(model_path) / 1024.0
        print(f"\nSaved tuned multiclass RF model to: {model_path}")
        print(f"Tuned multiclass RF model size: {size_kb:.1f} KB")

    return best_rf, tuner.best_params_, model_path

def tune_multiclass_tree(random_state: int = 42, save_model: bool = False):
    """
    Tune the multiclass Decision Tree, but stay in a 'reasonable' region:
    - don't over-prune
    - don't force gigantic leaves/splits
    - explore around your current good tree.

    Multi-metric scoring; refit on macro precision but we keep
    the search space from destroying recall/F1.
    """
    X_train, X_test, y_train, y_test = get_multiclass_splits()

    base_tree = DecisionTreeClassifier(
        criterion="entropy",
        random_state=random_state
    )

    # New, tighter search space
    param_distributions = {
        # Your current tree seems to like a medium depth; explore around that
        "max_depth": [10, 12, 14, 16, 18],

        # Smaller leaves (1–8) so we don't lose all detail
        "min_samples_leaf": [1, 2, 4, 8],

        # Reasonable split thresholds, but no insane 40 anymore
        "min_samples_split": [2, 5, 10, 20],

        # Keep both options; your current best was class_weight=None
        "class_weight": [None, "balanced"],

        # MUCH gentler pruning – no 5e-4 or 1e-3
        "ccp_alpha": [0.0, 1e-5, 5e-5, 1e-4],
    }

    scoring = {
        "accuracy": "accuracy",
        "precision_macro": "precision_macro",
        "recall_macro": "recall_macro",
        "f1_macro": "f1_macro",
    }

    tuner = RandomizedSearchCV(
        estimator=base_tree,
        param_distributions=param_distributions,
        n_iter=400,                 # we can afford a bit more, it's fast
        scoring=scoring,
        refit="precision_macro",   # still choose by macro precision
        cv=3,
        n_jobs=-1,
        random_state=random_state,
        verbose=1,
    )

    print("=== Tuning multiclass Decision Tree (constrained search) ===")
    start = time.perf_counter()
    tuner.fit(X_train, y_train)
    end = time.perf_counter()

    print(f"\nMulticlass DT tuning time: {end - start:.3f} s")
    print("Best parameters (multiclass DT):")
    print(tuner.best_params_)

    best_idx = tuner.best_index_

    print("\n=== CV results for BEST multiclass DT params ===")
    print(f"CV Accuracy (mean):        {tuner.cv_results_['mean_test_accuracy'][best_idx]:.4f}")
    print(f"CV Precision_macro (mean): {tuner.cv_results_['mean_test_precision_macro'][best_idx]:.4f}")
    print(f"CV Recall_macro (mean):    {tuner.cv_results_['mean_test_recall_macro'][best_idx]:.4f}")
    print(f"CV F1_macro (mean):        {tuner.cv_results_['mean_test_f1_macro'][best_idx]:.4f}")

    # Test-set evaluation with the best tree
    best_tree = tuner.best_estimator_
    y_pred = best_tree.predict(X_test)

    print("\n=== Best multiclass DT on held-out test set ===")
    print_multiclass_metrics(y_test, y_pred)

    model_path = None
    if save_model:
        os.makedirs("trees/models", exist_ok=True)
        model_path = os.path.join("trees", "models", "tree_multiclass_tuned.joblib")
        joblib.dump(best_tree, model_path)

        size_kb = os.path.getsize(model_path) / 1024.0
        print(f"\nSaved tuned multiclass DT model to: {model_path}")
        print(f"Tuned multiclass DT model size: {size_kb:.1f} KB")

    return best_tree, tuner.best_params_, model_path
