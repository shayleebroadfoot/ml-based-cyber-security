# logreg/binary.py

# Binary logistic regression model (Solution 1 - binary task)

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

from .preprocess import load_and_preprocess


def run_binary_logreg(test_size=0.2, random_state=42):
    # Load data and preprocessor (same logic as Colab)
    df, features, preprocessor = load_and_preprocess()

    # Prepare data for binary classification
    X = df[features]
    y = df["is_attack"]

    # Stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=test_size, random_state=random_state
    )

    # Binary logistic regression pipeline (with class weights)
    binary_model = Pipeline(
        steps=[
            ("preproc", preprocessor),
            (
                "clf",
                LogisticRegression(
                    max_iter=5000,
                    class_weight="balanced",
                    solver="saga",
                ),
            ),
        ]
    )

    print("\nTraining binary logistic regression...")
    binary_model.fit(X_train, y_train)
    print("Done.")

    # Predictions and evaluation (same as original)
    y_pred = binary_model.predict(X_test)
    print("\nClassification report (binary):")
    print(classification_report(y_test, y_pred, digits=4))
    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    return binary_model


if __name__ == "__main__":
    run_binary_logreg()
