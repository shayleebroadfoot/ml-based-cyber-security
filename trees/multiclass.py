# trees/multiclass.py

import os
import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from trees.preprocess import get_multiclass_splits
from utils.metrics import print_multiclass_metrics


def train_multiclass_models():
    X_train, X_test, y_train, y_test = get_multiclass_splits()

    # Decision Tree baseline
    tree_multi = DecisionTreeClassifier(
        max_depth=15,
        min_samples_leaf=5,
        random_state=42
    )
    tree_multi.fit(X_train, y_train)
    y_pred_tree = tree_multi.predict(X_test)

    print_multiclass_metrics(y_test, y_pred_tree, title="Decision Tree (Multiclass)")

    # Random Forest main model
    # forest_multi = RandomForestClassifier(
    #     n_estimators=100,
    #     max_depth=25,
    #     min_samples_split=2,
    #     min_samples_leaf=2,
    #     max_features="sqrt",
    #     class_weight="balanced",
    #     n_jobs=-1,
    #     random_state=42
    # )

    forest_multi = RandomForestClassifier(
        n_estimators=400,
        max_depth=25,
        min_samples_split=10,
        min_samples_leaf=1,
        max_features=0.5,
        class_weight=None,
        n_jobs=-1,
        bootstrap=False,
        random_state=42
    )
    forest_multi.fit(X_train, y_train)
    y_pred_rf = forest_multi.predict(X_test)

    print_multiclass_metrics(y_test, y_pred_rf, title="Random Forest (Multiclass)")

    # Save RF model
    os.makedirs("trees/models", exist_ok=True)
    model_path = os.path.join("trees", "models", "rf_multiclass.joblib")
    joblib.dump(forest_multi, model_path)
    print("\nSaved multiclass RF model to:", model_path)

    return forest_multi
