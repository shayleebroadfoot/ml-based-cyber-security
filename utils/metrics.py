# utils/metrics.py

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report



def print_binary_metrics(y_true, y_pred, title: str = ""):
    if title:
        print(f'===== {title} =====')

    print("Accuracy :", accuracy_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred))
    print("Recall   :", recall_score(y_true, y_pred))
    print("F1 Score :", f1_score(y_true, y_pred))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, zero_division=0))

def print_multiclass_metrics(y_true, y_pred, title: str = ""):
    if title:
        print(f"\n=== {title} ===")
    print("Accuracy :", accuracy_score(y_true, y_pred))
    print("Precision (macro):", precision_score(y_true, y_pred, average="macro", zero_division=0))
    print("Recall (macro)   :", recall_score(y_true, y_pred, average="macro", zero_division=0))
    print("F1 Score (macro) :", f1_score(y_true, y_pred, average="macro", zero_division=0))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, zero_division=0))


