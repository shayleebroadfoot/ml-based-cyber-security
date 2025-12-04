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

# trees/tune_multiclass_rf.py

import time
import itertools

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)
from sklearn.preprocessing import LabelEncoder
import joblib

from trees.preprocess import get_multiclass_splits

MODEL_PATH_BEST_ACC = "trees/models/rf_multiclass_best_acc.joblib"
MODEL_PATH_BEST_F1 = "trees/models/rf_multiclass_best_f1.joblib"


def load_attack_only_splits(test_size=0.2, random_state=42):
    """
    Use the existing multiclass splits, then filter out 'Normal' so we only have attack rows.
    This keeps behaviour consistent with your main RF multiclass training.
    """
    X_train, X_test, y_train, y_test = get_multiclass_splits(
        test_size=test_size,
        random_state=random_state,
    )

    # Filter out 'Normal' if it exists (attacks only)
    mask_train = y_train != "Normal"
    mask_test = y_test != "Normal"

    X_train_attacks = X_train[mask_train]
    y_train_attacks = y_train[mask_train]

    X_test_attacks = X_test[mask_test]
    y_test_attacks = y_test[mask_test]

    return X_train_attacks, X_test_attacks, y_train_attacks, y_test_attacks


def summarize_labels(y_train, y_test):
    print("Multiclass unique labels (train):", sorted(y_train.unique().tolist()))
    print("Multiclass unique labels (test) :", sorted(y_test.unique().tolist()))
    print("Any 'Normal' in train:", "Normal" in y_train.unique())
    print("Any 'Normal' in test :", "Normal" in y_test.unique())
    print()
    print("Train label distribution:")
    print(y_train.value_counts())
    print()
    print("Test label distribution:")
    print(y_test.value_counts())
    print()


def tune_multiclass_rf():
    print("[RF Multiclass Tuning] Loading attack-only splits...")
    X_train, X_test, y_train, y_test = load_attack_only_splits()

    print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
    summarize_labels(y_train, y_test)

    # Encode labels once for potential RMSE or other numeric metrics if needed
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)

    # ---- Tuning grid (tree shape first, fixed n_estimators) ----
    # You can tweak this if you want a smaller or larger search space.
    n_estimators = 300  # keep this fixed for shape tuning

    max_depth_list = [8, 12, 16, 20]
    min_samples_leaf_list = [1, 2, 5, 10]
    min_samples_split_list = [2, 8, 16]
    max_features_list = ["sqrt", "log2"]

    param_grid = list(
        itertools.product(
            max_depth_list,
            min_samples_leaf_list,
            min_samples_split_list,
            max_features_list,
        )
    )

    print(f"Total combinations to try: {len(param_grid)}")
    print()

    best_acc = -1.0
    best_f1 = -1.0

    best_params_acc = None
    best_params_f1 = None

    results = []

    combo_index = 0

    for max_depth, min_samples_leaf, min_samples_split, max_features in param_grid:
        combo_index += 1
        print("=" * 80)
        print(
            f"[{combo_index}/{len(param_grid)}] "
            f"max_depth={max_depth}, "
            f"min_samples_leaf={min_samples_leaf}, "
            f"min_samples_split={min_samples_split}, "
            f"max_features={max_features}"
        )

        rf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            min_samples_split=min_samples_split,
            max_features=max_features,
            n_jobs=-1,
            random_state=42,
        )

        start = time.time()
        rf.fit(X_train, y_train)
        train_time = time.time() - start

        # Evaluate on test split
        y_pred = rf.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            y_test, y_pred, average="macro", zero_division=0
        )

        # For completeness, you can compute encoded RMSE if you care
        y_pred_enc = le.transform(y_pred)
        rmse = np.sqrt(((y_test_enc - y_pred_enc) ** 2).mean())

        results.append(
            {
                "max_depth": max_depth,
                "min_samples_leaf": min_samples_leaf,
                "min_samples_split": min_samples_split,
                "max_features": max_features,
                "accuracy": acc,
                "precision_macro": precision_macro,
                "recall_macro": recall_macro,
                "f1_macro": f1_macro,
                "rmse": rmse,
                "train_time": train_time,
            }
        )

        print(f"[Train] time: {train_time:.3f} s")
        print(f"Accuracy       : {acc:.6f}")
        print(f"Precision macro: {precision_macro:.6f}")
        print(f"Recall macro   : {recall_macro:.6f}")
        print(f"F1 macro       : {f1_macro:.6f}")
        print(f"RMSE (encoded) : {rmse:.6f}")
        print()

        # --- Track best by accuracy (primary) ---
        if acc > best_acc or (acc == best_acc and f1_macro > best_f1):
            best_acc = acc
            best_params_acc = {
                "max_depth": max_depth,
                "min_samples_leaf": min_samples_leaf,
                "min_samples_split": min_samples_split,
                "max_features": max_features,
            }

        # --- Track separate best by macro F1 ---
        if f1_macro > best_f1 or (f1_macro == best_f1 and acc > best_acc):
            best_f1 = f1_macro
            best_params_f1 = {
                "max_depth": max_depth,
                "min_samples_leaf": min_samples_leaf,
                "min_samples_split": min_samples_split,
                "max_features": max_features,
            }

    print("=" * 80)
    print("[Summary] Best by ACCURACY")
    print("Best accuracy:", best_acc)
    print("Params:", best_params_acc)
    print()

    print("[Summary] Best by MACRO F1")
    print("Best macro F1:", best_f1)
    print("Params:", best_params_f1)
    print()

    # ---- Refit and save best-by-accuracy model ----
    print("[Refit] Training best-by-accuracy model on full training data...")
    rf_best_acc = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=best_params_acc["max_depth"],
        min_samples_leaf=best_params_acc["min_samples_leaf"],
        min_samples_split=best_params_acc["min_samples_split"],
        max_features=best_params_acc["max_features"],
        n_jobs=-1,
        random_state=42,
    )

    rf_best_acc.fit(X_train, y_train)
    y_pred_best_acc = rf_best_acc.predict(X_test)

    print("=== Best-by-Accuracy Classification Report ===")
    print(classification_report(y_test, y_pred_best_acc))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred_best_acc))
    print()

    joblib.dump(rf_best_acc, MODEL_PATH_BEST_ACC)
    print(f"Saved best-accuracy RF model to: {MODEL_PATH_BEST_ACC}")
    print()

    # ---- Refit and save best-by-F1 model (optional but useful) ----
    print("[Refit] Training best-by-macro-F1 model on full training data...")
    rf_best_f1 = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=best_params_f1["max_depth"],
        min_samples_leaf=best_params_f1["min_samples_leaf"],
        min_samples_split=best_params_f1["min_samples_split"],
        max_features=best_params_f1["max_features"],
        n_jobs=-1,
        random_state=42,
    )

    rf_best_f1.fit(X_train, y_train)
    y_pred_best_f1 = rf_best_f1.predict(X_test)

    print("=== Best-by-Macro-F1 Classification Report ===")
    print(classification_report(y_test, y_pred_best_f1))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred_best_f1))
    print()

    joblib.dump(rf_best_f1, MODEL_PATH_BEST_F1)
    print(f"Saved best-F1 RF model to: {MODEL_PATH_BEST_F1}")
    print()

    # If you want to inspect the grid results later, you can dump them to CSV
    df_results = pd.DataFrame(results)
    df_results.sort_values(by=["accuracy", "f1_macro"], ascending=False, inplace=True)
    df_results.to_csv("trees/models/rf_multiclass_tuning_results.csv", index=False)
    print("Saved tuning results table to trees/models/rf_multiclass_tuning_results.csv")


if __name__ == "__main__":
    tune_multiclass_rf()
