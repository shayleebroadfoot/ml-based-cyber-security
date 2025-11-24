# Original .py file from Google colab
import pandas as pd # For csv
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

DATA_FILE = "data/training_data.csv"

# Load data from file
df = pd.read_csv(DATA_FILE)

print("Shape:", df.shape) # (rows, columns)
print("First 5 rows:")
print(df.head())

df[["id", "attack_cat", "label"]].head() # Inspect key columns

# Make a copy and drop ID column
df_no_id = df.drop(columns=["id"])

# 0 = normal, 1 = attack
y_binary = df_no_id["label"]

# attack type (string)
y_multiclass = df_no_id["attack_cat"]

# All remaining columns are features (inputs)
X_raw = df_no_id.drop(columns=["label", "attack_cat"])

print("X_raw shape:", X_raw.shape)
print("y_binary shape:", y_binary.shape)
print("y_multiclass shape:", y_multiclass.shape)

print("Data types in X_raw:")
print(X_raw.dtypes.value_counts())

print("\nColumns that are NOT numeric:")
non_numeric = X_raw.dtypes[X_raw.dtypes == "object"].index.tolist()
print(non_numeric)

# Use one-hot encoding on columns that are non-numeric
X = pd.get_dummies(X_raw, columns=non_numeric, drop_first=True)

# Inspect changes
print("Original feature count:", X_raw.shape[1])
print("Encoded feature count:", X.shape[1])
print("Encoded columns sample:", X.columns[:10].tolist())

# Split dataset for binary classification: 20% for test set, use same 0/1 ratio in train and test sets
X_train_bin, X_test_bin, y_train_bin, y_test_bin = train_test_split(
    X, y_binary,
    test_size=0.2,
    random_state=42,
    stratify=y_binary
)

print("Binary train shape:", X_train_bin.shape)
print("Binary test shape:", X_test_bin.shape)

# Create the binary model
tree_bin = DecisionTreeClassifier(
    max_depth=10,
    min_samples_leaf=5,
    random_state=42
)

# Train / Fit the Decision Tree
tree_bin.fit(X_train_bin, y_train_bin)

# Predict on the test set
y_pred_bin = tree_bin.predict(X_test_bin)

print("Decision Tree (Binary) Evaluation")

print("Accuracy :", accuracy_score(y_test_bin, y_pred_bin))
print("Precision:", precision_score(y_test_bin, y_pred_bin))
print("Recall   :", recall_score(y_test_bin, y_pred_bin))
print("F1 Score :", f1_score(y_test_bin, y_pred_bin))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test_bin, y_pred_bin))

print("\nClassification Report:")
print(classification_report(y_test_bin, y_pred_bin, zero_division=0))

forest_bin = RandomForestClassifier(
    n_estimators=100,       # 100 trees
    max_depth=15,          # prevents huge trees
    min_samples_leaf=5,    # regularization
    max_features="sqrt",   # common RF setting, speeds up training
    random_state=42,
    n_jobs=-1              # use all CPU cores
)

forest_bin.fit(X_train_bin, y_train_bin)

y_pred_forest_bin = forest_bin.predict(X_test_bin)

print("=== Random Forest (Binary) Evaluation ===")
print("Accuracy :", accuracy_score(y_test_bin, y_pred_forest_bin))
print("Precision:", precision_score(y_test_bin, y_pred_forest_bin))
print("Recall   :", recall_score(y_test_bin, y_pred_forest_bin))
print("F1 Score :", f1_score(y_test_bin, y_pred_forest_bin))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test_bin, y_pred_forest_bin))

"""
========= Multiclass Classification =======
"""

# Split dataset for multiclass classification (attack_cat)
X_train_multi, X_test_multi, y_train_multi, y_test_multi = train_test_split(
    X, y_multiclass,
    test_size=0.2,
    random_state=42,
    stratify=y_multiclass
)

print("Multiclass train shape:", X_train_multi.shape)
print("Multiclass test shape:", X_test_multi.shape)

print("\n================ MULTICLASS: attack_cat ================")

# Decision Tree for multiclass classification
tree_multi = DecisionTreeClassifier(
    max_depth=15,        # a bit deeper; multiclass boundaries are more complex
    min_samples_leaf=5,
    random_state=42
)

tree_multi.fit(X_train_multi, y_train_multi)
y_pred_tree_multi = tree_multi.predict(X_test_multi)

print("\n=== Decision Tree (Multiclass) Evaluation ===")
print("Accuracy :", accuracy_score(y_test_multi, y_pred_tree_multi))
print("Precision (macro):", precision_score(y_test_multi, y_pred_tree_multi, average="macro", zero_division=0))
print("Recall (macro)   :", recall_score(y_test_multi, y_pred_tree_multi, average="macro", zero_division=0))
print("F1 Score (macro) :", f1_score(y_test_multi, y_pred_tree_multi, average="macro", zero_division=0))

print("\nConfusion Matrix (Decision Tree Multiclass):")
print(confusion_matrix(y_test_multi, y_pred_tree_multi))

print("\nClassification Report (Decision Tree Multiclass):")
print(classification_report(y_test_multi, y_pred_tree_multi, zero_division=0))

# Random Forest for multiclass classification
forest_multi = RandomForestClassifier(
    n_estimators=150,     # a bit more trees, multiclass is harder
    max_depth=20,
    min_samples_leaf=3,
    max_features="sqrt",
    n_jobs=-1,
    random_state=42
)

forest_multi.fit(X_train_multi, y_train_multi)
y_pred_forest_multi = forest_multi.predict(X_test_multi)

print("\n=== Random Forest (Multiclass) Evaluation ===")
print("Accuracy :", accuracy_score(y_test_multi, y_pred_forest_multi))
print("Precision (macro):", precision_score(y_test_multi, y_pred_forest_multi, average="macro", zero_division=0))
print("Recall (macro)   :", recall_score(y_test_multi, y_pred_forest_multi, average="macro", zero_division=0))
print("F1 Score (macro) :", f1_score(y_test_multi, y_pred_forest_multi, average="macro", zero_division=0))

print("\nConfusion Matrix (Random Forest Multiclass):")
print(confusion_matrix(y_test_multi, y_pred_forest_multi))

print("\nClassification Report (Random Forest Multiclass):")
print(classification_report(y_test_multi, y_pred_forest_multi, zero_division=0))
