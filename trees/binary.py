# trees/binary.py

import os
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
    tree_bin.fit(X_train, y_train)
    y_pred_tree = tree_bin.predict(X_test)

    print_binary_metrics(y_test, y_pred_tree, title="Decision Tree (Binary)")

    # Random Forest main model
    forest_bin = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_leaf=3,
        max_features=None,
        n_jobs=-1,
        random_state=42
    )
    forest_bin.fit(X_train, y_train)
    y_pred_rf = forest_bin.predict(X_test)

    print_binary_metrics(y_test, y_pred_rf, title="Random Forest (Binary)")

    # Save RF model
    os.makedirs("trees/models", exist_ok=True)
    model_path = os.path.join("trees", "models", "rf_binary.joblib")
    joblib.dump(forest_bin, model_path)
    print("\nSaved binary RF model to:", model_path)

    return forest_bin
