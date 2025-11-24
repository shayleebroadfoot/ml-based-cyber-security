# trees/tuning.py

from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier

from trees.preprocess import get_binary_splits, get_multiclass_splits

def tune_binary_rf():
    X_train, X_test, y_train, y_test = get_binary_splits()

    param_distributions = {
        "n_estimators": [50, 100, 200],
        "max_depth": [10, 15, 20, None],
        "min_samples_leaf": [1, 3, 5],
        "max_features": ["sqrt", "log2", None],
    }

    base_rf = RandomForestClassifier(
        n_jobs=-1,
        random_state=42
    )

    tuner = RandomizedSearchCV(
        estimator=base_rf,
        param_distributions=param_distributions,
        n_iter=10,
        scoring="f1",
        cv=3,
        n_jobs=-1,
        random_state=42,
        verbose=1
    )

    print("Starting hyperparameter search for binary Random Forest...")
    tuner.fit(X_train, y_train)

    print("\nBest parameters:")
    print(tuner.best_params_)
    print("Best CV F1 score:", tuner.best_score_)

    return tuner.best_params_

def tune_multiclass_rf():
    """
    Hyperparameter tuning for the multiclass Random Forest (attack_cat).
    Uses macro F1 so rare classes have equal weight.
    """
    X_train, X_test, y_train, y_test = get_multiclass_splits()

    param_distributions = {
        "n_estimators": [100, 200, 300, 400],
        "max_depth": [15, 20, 25, 30, None],
        "min_samples_leaf": [1, 2, 4, 8],
        "min_samples_split": [2, 5, 10, 20],
        "max_features": ["sqrt", "log2", 0.3, 0.5],
        "bootstrap": [True, False],
        "class_weight": [None, "balanced", "balanced_subsample"],
    }

    base_rf = RandomForestClassifier(
        n_jobs=-1,
        random_state=42
    )

    tuner = RandomizedSearchCV(
        estimator=base_rf,
        param_distributions=param_distributions,
        n_iter=40,            # more combinations (heavier training)
        scoring="f1_macro",   # macro F1 for multiclass
        cv=3,
        n_jobs=-1,
        random_state=42,
        verbose=1
    )

    print("Starting hyperparameter search for multiclass Random Forest...")
    tuner.fit(X_train, y_train)

    print("\nBest parameters (multiclass RF):")
    print(tuner.best_params_)
    print("Best CV macro F1 score:", tuner.best_score_)

    return tuner.best_params_
