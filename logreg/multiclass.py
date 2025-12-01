# logreg/multiclass.py

# Multiclass (softmax) logistic regression for attack categories
# Only trained on rows where is_attack = 1 (same as original Colab logic).

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

from .preprocess import load_and_preprocess


def run_multiclass_logreg(test_size=0.2, random_state=42):
    # Load data and preprocessor
    df, features, preprocessor = load_and_preprocess()

    if "attack_cat_encoded" not in df.columns:
        print(
            "attack_cat_encoded not found: skipping multiclass training."
        )
        return None

    # Filter only attack rows
    df_attacks = df[df["is_attack"] == 1].copy()
    X_m = df_attacks[features]
    y_m = df_attacks["attack_cat_encoded"]

    print("Multiclass attack rows shape:", X_m.shape)

    # Stratified split
    Xtr, Xte, ytr, yte = train_test_split(
        X_m, y_m, stratify=y_m, test_size=test_size, random_state=random_state
    )

    # Softmax logistic regression (multinomial)
    softmax_model = Pipeline(
        steps=[
            ("preproc", preprocessor),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000,
                    multi_class="multinomial",
                    solver="saga",
                    class_weight="balanced",
                ),
            ),
        ]
    )

    print("\nTraining softmax (multinomial) logistic regression...")
    softmax_model.fit(Xtr, ytr)
    print("Done.")

    # Predict and evaluate (same as original, no target_names)
    ypred_m = softmax_model.predict(Xte)
    print("\nClassification report (multiclass):")
    print(classification_report(yte, ypred_m, digits=4))

    return softmax_model


if __name__ == "__main__":
    run_multiclass_logreg()
