import sys
from pathlib import Path

# -------------------------------
# Solution 1 – Logistic Regression
# -------------------------------
from logreg.binary import (
    train_binary_logreg_model,
    test_binary_logreg_model,
)
from logreg.multiclass import (
    train_multiclass_logreg_model,
    test_multiclass_logreg_model,
)

# -------------------------------
# Solution 2 – SVM
# -------------------------------
from linsvm.binary import (
    train_binary_svm_model,
    test_binary_svm_model,
)
from linsvm.multiclass import (
    train_multiclass_svm_model,
    test_multiclass_svm_model,
)

# -------------------------------
# Solution 3 – Tree-based Models (RF / DT)
# -------------------------------
# Multiclass Random Forest (attacks only)
from trees.multiclass import (
    train_multiclass_model as train_multiclass_rf_model,
    test_multiclass_model as test_multiclass_rf_model,
)

# Multiclass Decision Tree (attacks only)
from trees.multiclass_tree import (
    train_multiclass_tree_model,
    test_multiclass_tree_model,
)

# If you ever want binary tree models:
# from trees.binary import train_binary_model, test_binary_model


# ---------------------------------------------------------------------
# Resolve project root and data paths (for sanity checks only)
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"

TRAIN_CSV = DATA_DIR / "training_data.csv"
FEATURES_CSV = DATA_DIR / "feature_description.csv"

print("Resolved project root:", PROJECT_ROOT)
print("Training CSV exists:", TRAIN_CSV.exists())
print("Feature description exists:", FEATURES_CSV.exists())


# ---------------------------------------------------------------------
# Menu + main loop
# ---------------------------------------------------------------------

def print_menu():
    print("=== SENG 4610 Intrusion Detection Project ===\n")

    print("Solution 1 – Logistic Regression (LogReg)")
    print("  1) Train Binary Logistic Regression")
    print("  2) Test  Binary Logistic Regression")
    print("  3) Train Multiclass Logistic Regression (attacks only)")
    print("  4) Test  Multiclass Logistic Regression (attacks only)\n")

    print("Solution 2 – Support Vector Machines (SVM)")
    print("  5) Train Binary SVM")
    print("  6) Test  Binary SVM")
    print("  7) Train Multiclass SVM (attacks only)")
    print("  8) Test  Multiclass SVM (attacks only)\n")

    print("Solution 3 – Tree-based Models (RF / DT)")
    print("  9)  Train Multiclass Decision Tree (attacks only)")
    print(" 10)  Test  Multiclass Decision Tree (attacks only)")
    print(" 11)  Train Multiclass Random Forest (attacks only)")
    print(" 12)  Test  Multiclass Random Forest (attacks only)\n")

    print("  0) Exit")


def main():
    while True:
        print_menu()
        choice = input("Select an option: ").strip()

        if choice == "0":
            print("Exiting.")
            sys.exit(0)

        # -------------------------
        # Solution 1 – LogReg
        # -------------------------
        elif choice == "1":
            print("\n[Running Solution 1: TRAIN Binary Logistic Regression]\n")
            train_binary_logreg_model()

        elif choice == "2":
            print("\n[Running Solution 1: TEST Binary Logistic Regression]\n")
            test_binary_logreg_model()

        elif choice == "3":
            print("\n[Running Solution 1: TRAIN Multiclass Logistic Regression]\n")
            train_multiclass_logreg_model()

        elif choice == "4":
            print("\n[Running Solution 1: TEST Multiclass Logistic Regression]\n")
            test_multiclass_logreg_model()

        # -------------------------
        # Solution 2 – SVM
        # -------------------------
        elif choice == "5":
            print("\n[Running Solution 2: TRAIN Binary SVM]\n")
            train_binary_svm_model()

        elif choice == "6":
            print("\n[Running Solution 2: TEST Binary SVM]\n")
            test_binary_svm_model()

        elif choice == "7":
            print("\n[Running Solution 2: TRAIN Multiclass SVM (attacks only)]\n")
            train_multiclass_svm_model()

        elif choice == "8":
            print("\n[Running Solution 2: TEST Multiclass SVM (attacks only)]\n")
            test_multiclass_svm_model()

        # -------------------------
        # Solution 3 – Trees (RF / DT)
        # -------------------------
        elif choice == "9":
            print("\n[Running Solution 3: TRAIN Multiclass Decision Tree (attacks only)]\n")
            train_multiclass_tree_model()

        elif choice == "10":
            print("\n[Running Solution 3: TEST Multiclass Decision Tree (attacks only)]\n")
            test_multiclass_tree_model()

        elif choice == "11":
            print("\n[Running Solution 3: TRAIN Multiclass Random Forest (attacks only)]\n")
            train_multiclass_rf_model()

        elif choice == "12":
            print("\n[Running Solution 3: TEST Multiclass Random Forest (attacks only)]\n")
            test_multiclass_rf_model()

        else:
            print("Invalid option. Please try again.\n")


if __name__ == "__main__":
    main()

# # from trees.binary import train_binary_model, test_binary_model
# # from trees.multiclass import train_multiclass_model, test_multiclass_model
# # from trees.multiclass_tree import train_multiclass_tree_model, test_multiclass_tree_model
# # from trees.tuning import tune_multiclass_rf, tune_multiclass_tree
# #
# # if __name__ == "__main__":
# #
# #     # _, rf_params, _ = tune_multiclass_rf(save_model=False)
# #     # _, tree_params, _ = tune_multiclass_tree(save_model=False)
# #     # Train RF + Tree
# #     # train_binary_model()
# #     train_multiclass_model()  # RF
# #     # train_multiclass_tree_model()  # DT
# #     #
# #     # # Test RF + Tree
# #
# #     # # test_binary_model()
# #     test_multiclass_model()  # RF
# #     # test_multiclass_tree_model()  # DT
#
# import sys
# from pathlib import Path
#
# # -------------------------------
# # Solution 1 – Logistic Regression
# # -------------------------------
# from logreg.binary import (
#     train_binary_logreg_model,
#     test_binary_logreg_model,
# )
# from logreg.multiclass import (
#     train_multiclass_logreg_model,
#     test_multiclass_logreg_model,
# )
#
# # -------------------------------
# # Solution 2 – SVM
# # -------------------------------
# from linsvm.binary import (
#     train_binary_svm_model,
#     test_binary_svm_model,
# )
# from linsvm.multiclass import (
#     train_multiclass_svm_model,
#     test_multiclass_svm_model,
# )
#
# # ---------------------------------------------------------------------
# # Resolve project root and data paths (for sanity checks only)
# # ---------------------------------------------------------------------
#
# PROJECT_ROOT = Path(__file__).resolve().parent
# DATA_DIR = PROJECT_ROOT / "data"
#
# TRAIN_CSV = DATA_DIR / "training_data.csv"
# FEATURES_CSV = DATA_DIR / "feature_description.csv"
#
# print("Resolved project root:", PROJECT_ROOT)
# print("Training CSV exists:", TRAIN_CSV.exists())
# print("Feature description exists:", FEATURES_CSV.exists())
#
#
# # ---------------------------------------------------------------------
# # Menu + main loop
# # ---------------------------------------------------------------------
#
# def print_menu():
#     print("=== SENG 4610 Intrusion Detection Project ===\n")
#
#     print("Solution 1 – Logistic Regression (LogReg)")
#     print("  1) Train Binary Logistic Regression")
#     print("  2) Test  Binary Logistic Regression")
#     print("  3) Train Multiclass Logistic Regression (attacks only)")
#     print("  4) Test  Multiclass Logistic Regression (attacks only)\n")
#
#     print("Solution 2 – Support Vector Machines (SVM)")
#     print("  5) Train Binary SVM")
#     print("  6) Test  Binary SVM")
#     print("  7) Train Multiclass SVM (attacks only)")
#     print("  8) Test  Multiclass SVM (attacks only)\n")
#
#     # If you later want trees / RF as Solution 3, you can add:
#     # print("Solution 3 – Tree-based Models (RF / DT)")
#     # print("  9) Train Multiclass Decision Tree")
#     # print(" 10) Test  Multiclass Decision Tree")
#     # print(" 11) Train Multiclass Random Forest")
#     # print(" 12) Test  Multiclass Random Forest\n")
#
#     print("  0) Exit")
#
#
# def main():
#     while True:
#         print_menu()
#         choice = input("Select an option: ").strip()
#
#         if choice == "0":
#             print("Exiting.")
#             sys.exit(0)
#
#         # -------------------------
#         # Solution 1 – LogReg
#         # -------------------------
#         elif choice == "1":
#             print("\n[Running Solution 1: TRAIN Binary Logistic Regression]\n")
#             train_binary_logreg_model()
#
#         elif choice == "2":
#             print("\n[Running Solution 1: TEST Binary Logistic Regression]\n")
#             test_binary_logreg_model()
#
#         elif choice == "3":
#             print("\n[Running Solution 1: TRAIN Multiclass Logistic Regression]\n")
#             train_multiclass_logreg_model()
#
#         elif choice == "4":
#             print("\n[Running Solution 1: TEST Multiclass Logistic Regression]\n")
#             test_multiclass_logreg_model()
#
#         # -------------------------
#         # Solution 2 – SVM
#         # -------------------------
#         elif choice == "5":
#             print("\n[Running Solution 2: TRAIN Binary SVM]\n")
#             train_binary_svm_model()
#
#         elif choice == "6":
#             print("\n[Running Solution 2: TEST Binary SVM]\n")
#             test_binary_svm_model()
#
#         elif choice == "7":
#             print("\n[Running Solution 2: TRAIN Multiclass SVM (attacks only)]\n")
#             train_multiclass_svm_model()
#
#         elif choice == "8":
#             print("\n[Running Solution 2: TEST Multiclass SVM (attacks only)]\n")
#             test_multiclass_svm_model()
#
#         else:
#             print("Invalid option. Please try again.\n")
#
#
# if __name__ == "__main__":
#     main()
