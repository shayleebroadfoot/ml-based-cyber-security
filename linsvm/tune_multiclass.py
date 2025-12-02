# linsvm/tune_multiclass.py
#
# Hyperparameter tuning for MULTICLASS Linear SVM (attacks only).
# Uses:
#   - linsvm.preprocess.get_multiclass_splits()
#   - GridSearchCV over C, penalty, class_weight
#
# Primary goal: improve MACRO PRECISION on attacks.

from pathlib import Path

import joblib
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    make_scorer,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from linsvm.preprocess import get_multiclass_splits


def main():
    print("[TUNE] Loading multiclass attack-only splits...")
    X_train, X_test, y_train, y_test, preprocessor = get_multiclass_splits(
        test_size=0.20,
        random_state=42,
    )

    print("[TUNE] Shapes:")
    print("  X_train:", X_train.shape)
    print("  X_test :", X_test.shape)
    print("  y_train:", y_train.shape)
    print("  y_test :", y_test.shape)

    # Base pipeline: preprocessor + LinearSVC
    base_clf = LinearSVC(
        random_state=42,
        max_iter=8000,   # more iters to actually converge
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("svm", base_clf),
        ]
    )

    # ---- KEY CHANGE 1: sharper grid, allow L1 vs L2, keep precision-friendly settings ----
    param_grid = {
        # Focus around what already worked (C=10) and nearby values
        "svm__C": [1.0, 5.0, 10.0, 20.0],
        # L2 is your current; L1 can sparsify and sometimes improve precision
        "svm__penalty": ["l2", "l1"],
        # You care about precision; class_weight=None usually helps that more than "balanced"
        "svm__class_weight": [None, "balanced"],
        # squared_hinge + dual=False is required for L1 and usually better behaved here
        "svm__loss": ["squared_hinge"],
        "svm__dual": [False],
    }

    # ---- KEY CHANGE 2: proper macro-precision scorer (no undefined metric warnings) ----
    precision_macro_scorer = make_scorer(
        precision_score,
        average="macro",
        zero_division=0,
    )

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    total_combos = (
        len(param_grid["svm__C"])
        * len(param_grid["svm__penalty"])
        * len(param_grid["svm__class_weight"])
        * len(param_grid["svm__loss"])
        * len(param_grid["svm__dual"])
    )
    print(f"[TUNE] Starting GridSearchCV over {total_combos} combinations...")

    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring=precision_macro_scorer,   # REFIT ON MACRO PRECISION
        cv=cv,
        n_jobs=-1,
        verbose=2,
        refit=True,
    )

    grid.fit(X_train, y_train)

    print("\n[TUNE] Best parameters (by macro precision):")
    print(grid.best_params_)

    print(f"[TUNE] Best CV macro precision: {grid.best_score_:.4f}")

    # Evaluate best model on held-out test set
    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec_macro = precision_score(y_test, y_pred, average="macro", zero_division=0)
    rec_macro = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)

    print("\n=== Multiclass Linear SVM (TUNED) – Test Results ===")
    print(f"Accuracy : {acc:.6f}")
    print(f"Precision (macro): {prec_macro:.6f}")
    print(f"Recall (macro)   : {rec_macro:.6f}")
    print(f"F1 Score (macro) : {f1_macro:.6f}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    # Save tuned model
    models_dir = Path(__file__).resolve().parent / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    out_path = models_dir / "svm_multiclass_tuned.joblib"
    joblib.dump(best_model, out_path)
    print(f"\n[TUNE] Saved tuned multiclass SVM model to: {out_path}")


if __name__ == "__main__":
    main()
