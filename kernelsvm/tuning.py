# kernelsvm/tune_multiclass.py
#
# Simple C-tuning script for multiclass RBF SVM (attacks only).
# - Reuses kernelsvm.preprocess.get_multiclass_splits
# - Keeps gamma, kernel, class_weight fixed
# - Loops over a grid of C values and prints metrics

import time

from sklearn.pipeline import Pipeline

from .preprocess import get_multiclass_splits, get_binary_splits

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

C_GRID = [1, 2, 3, 4, 5, 6, 7, 8]

def tune_multiclass_kernel_svm():
    """
    Run a small C-sweep for the multiclass RBF SVM using the same
    preprocessing and splits as kernelsvm.multiclass.
    """
    X_train, X_test, y_train, y_test, preprocessor = get_multiclass_splits()

    print("\n[Kernel SVM Tuning] Multiclass (attacks only)")
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    print(f"C grid: {C_GRID}\n")

    for C_value in C_GRID:
        print("=" * 80)
        print(f"=== C = {C_value} (kernel='rbf', gamma=0.1) ===")

        svm = SVC(
            kernel="rbf",
            C=C_value,
            gamma=0.1,
            class_weight=None,
            decision_function_shape="ovr",
            random_state=42,
        )

        model = Pipeline(
            steps=[
                ("preproc", preprocessor),
                ("clf", svm),
            ]
        )

        start_fit = time.perf_counter()
        model.fit(X_train, y_train)
        end_fit = time.perf_counter()
        fit_time = end_fit - start_fit

        # Predictions
        start_pred = time.perf_counter()
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        end_pred = time.perf_counter()
        pred_time = end_pred - start_pred

        # RMSE
        train_mse = mean_squared_error(y_train, y_train_pred)
        test_mse = mean_squared_error(y_test, y_test_pred)
        train_rmse = train_mse ** 0.5
        test_rmse = test_mse ** 0.5

        # Metrics on test set
        print(f"\n[Train] time: {fit_time:.3f} s")
        print(f"[Train] RMSE: {train_rmse:.4f}")
        print(f"[Test ] RMSE: {test_rmse:.4f}")
        print(f"[Pred ] time (train+test preds): {pred_time:.3f} s\n")

        print_multiclass_metrics(y_test, y_test_pred)
        print("\n")  # spacer between C runs


from sklearn.svm import SVC
from utils.metrics import print_multiclass_metrics
from sklearn.metrics import mean_squared_error
import time
import numpy as np

# Tune gamma
def tune_rbf_multiclass():
    X_train, X_test, y_train, y_test, preprocessor = get_multiclass_splits()

    print("\n[Kernel SVM Tuning] Multiclass (attacks only)")
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    C_fixed = 1.0
    gamma_grid = [0.05, 0.075, 0.1, 0.15, 0.2]

    for gamma in gamma_grid:
        print("\n" + "=" * 80)
        print(f"=== C = {C_fixed} (kernel='rbf', gamma={gamma}) ===\n")

        svm = SVC(
            kernel="rbf",
            C=C_fixed,
            gamma=gamma,
            class_weight=None,
            decision_function_shape="ovr",
            random_state=42,
        )

        model = Pipeline(
            steps=[
                ("preproc", preprocessor),
                ("clf", svm),
            ]
        )

        # TRAIN
        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        t1 = time.perf_counter()

        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        t2 = time.perf_counter()

        rmse_train = mean_squared_error(y_train, y_train_pred) ** 0.5
        rmse_test = mean_squared_error(y_test, y_test_pred) ** 0.5

        print(f"[Train] time: {t1 - t0:.3f} s")
        print(f"[Train] RMSE: {rmse_train:.4f}")
        print(f"[Test ] RMSE: {rmse_test:.4f}")
        print(f"[Pred ] time (train+test preds): {t2 - t0:.3f} s\n")

        print_multiclass_metrics(y_test, y_test_pred)

def tune_C_values(C_list):
    """
    Simple tuning loop for the binary RBF SVM.
    Cycles through the list of C values, trains a model for each,
    and evaluates on the held-out test split.
    """

    print("\n=== C-VALUE TUNING (Binary RBF SVM) ===")

    # Load the data + preprocessing once
    X_train, X_test, y_train, y_test, preprocessor = get_binary_splits()

    results = []

    for C in C_list:
        print(f"\n--- Testing C = {C} ---")

        clf = Pipeline(
            steps=[
                ("pre", preprocessor),
                ("svm", SVC(
                    kernel="rbf",
                    C=C,
                    gamma=0.1,                  # keep your current gamma
                    class_weight='balanced',    # keep your weighting
                    probability=False,
                )),
            ]
        )

        # Train
        start = time.time()
        clf.fit(X_train, y_train)
        train_time = time.time() - start

        # Predict on test
        y_pred = clf.predict(X_test)

        # Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        print(f"Train Time: {train_time:.2f} s")
        print(f"Accuracy : {acc:.6f}")
        print(f"Precision: {prec:.6f}")
        print(f"Recall   : {rec:.6f}")
        print(f"F1 Score : {f1:.6f}")

        results.append((C, acc, prec, rec, f1))

    # Summary table
    print("\n=== SUMMARY OF C RESULTS ===")
    print("C\tAccuracy\tPrecision\tRecall\t\tF1")
    for C, acc, prec, rec, f1 in results:
        print(f"{C}\t{acc:.4f}\t\t{prec:.4f}\t\t{rec:.4f}\t\t{f1:.4f}")

    # Return results in case you want to sort or log them
    return results

def tune_gamma_values(gamma_list, C=9.0):
    """
    Tune gamma for the binary RBF SVM while keeping C fixed (default C=9).
    Cycles through each gamma value and evaluates accuracy, precision,
    recall, and F1 on the held-out test set.
    """
    print("\n=== GAMMA TUNING (Binary RBF SVM) ===")
    print(f"Using fixed C = {C}")

    # Load data/preprocessing once
    X_train, X_test, y_train, y_test, preprocessor = get_binary_splits()

    results = []

    for gamma in gamma_list:
        print(f"\n--- Testing gamma = {gamma} ---")

        clf = Pipeline(
            steps=[
                ("pre", preprocessor),
                ("svm", SVC(
                    kernel="rbf",
                    C=C,
                    gamma=gamma,
                    class_weight='balanced',
                    probability=False,
                )),
            ]
        )

        # Train model
        start = time.time()
        clf.fit(X_train, y_train)
        train_time = time.time() - start

        # Predict test set
        y_pred = clf.predict(X_test)

        # Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        print(f"Train Time: {train_time:.2f} s")
        print(f"Accuracy : {acc:.6f}")
        print(f"Precision: {prec:.6f}")
        print(f"Recall   : {rec:.6f}")
        print(f"F1 Score : {f1:.6f}")

        results.append((gamma, acc, prec, rec, f1))

    # Summary
    print("\n=== SUMMARY OF GAMMA RESULTS ===")
    print("gamma\tAccuracy\tPrecision\tRecall\t\tF1")
    for g, acc, prec, rec, f1 in results:
        print(f"{g}\t{acc:.4f}\t\t{prec:.4f}\t\t{rec:.4f}\t\t{f1:.4f}")

    return results


if __name__ == "__main__":
    # tune_multiclass_kernel_svm()
    # tune_rbf_multiclass()
    # C_tests = [0.1, 0.25, 0.5, 1, 2, 3, 5, 7, 10]
    # C_tests = [7.5, 8.0, 8.5, 9.0, 9.5]
    # tune_C_values(C_tests)
    gamma_tests = [0.001, 0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15]
    tune_gamma_values(gamma_tests, C=9.0)

#--- Testing C = 7 ---
# Train Time: 248.53 s
# Accuracy : 0.943369
# Precision: 0.979533
# Recall   : 0.936361
# F1 Score : 0.957460
#--- Testing C = 9.0 ---
# Train Time: 215.98 s
# Accuracy : 0.943483
# Precision: 0.979494
# Recall   : 0.936570
# F1 Score : 0.957552

