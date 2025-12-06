import sys
from pathlib import Path

# -------------------------------
# Solution 1 – Kernel SVM
# -------------------------------

from kernelsvm.binary import (
    train_binary_model as train_binary_kernelsvm_model,
    test_binary_model as test_binary_kernelsvm_model,
)

from kernelsvm.multiclass import (
    train_multiclass_kernel_svm_model,
    test_multiclass_kernel_svm_model
)

# -------------------------------
# Solution 2 – Tree-based Models (RF / DT)
# -------------------------------

# Binary Random Forest
from trees.binary import (
train_binary_model as train_binary_rf_model,
test_binary_model as test_binary_rf_model,
)
# Multiclass Random Forest
from trees.multiclass import (
    train_multiclass_model as train_multiclass_rf_model,
    test_multiclass_model as test_multiclass_rf_model,
)

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

    print("Solution 1 – Kernel SVM (Nonlinear)")
    print(" 1) Train Binary Kernel SVM (RBF)")
    print(" 2) Test  Binary Kernel SVM (RBF)")
    print(" 3) Train Multiclass Kernel SVM (RBF)")
    print(" 4) Test  Multiclass Kernel SVM (RBF)")

    print("Solution 2 – Random Forest")
    print("  5)  Train Binary Random Forest")
    print("  6)  Test  Binary Random Forest")
    print("  7)  Train Multiclass Random Forest")
    print("  8)  Test  Multiclass Random Forest\n")

    print("  0) Exit")

def main():
    while True:
        print_menu()
        choice = input("Select an option: ").strip()

        if choice == "0":
            print("Exiting.")
            sys.exit(0)

        # -------------------------
        # Solution 1 – Kernel SVM
        # -------------------------
        elif choice == "1":
            print("\n[Running Solution 4: TRAIN Kernel SVM (RBF)]\n")
            train_binary_kernelsvm_model()

        elif choice == "2":
            print("\n[Running Solution 4: TEST Kernel SVM (RBF)]\n")
            test_binary_kernelsvm_model()

        elif choice == "3":
            print("\n[Running Solution 4: TRAIN Multiclass Kernel SVM (RBF)]\n")
            train_multiclass_kernel_svm_model()

        elif choice == "4":
            print("\n[Running Solution 4: TEST Multiclass Kernel SVM (RBF)]\n")
            test_multiclass_kernel_svm_model()

        # -------------------------
        # Solution 2 – Random Forest
        # -------------------------
        elif choice == "5":
            print("\n[Running Solution 3: TRAIN Binary Random Forest (attacks only)]\n")
            train_binary_rf_model()

        elif choice == "6":
            print("\n[Running Solution 3: TEST Binary Random Forest (attacks only)]\n")
            test_binary_rf_model()

        elif choice == "7":
            print("\n[Running Solution 3: TRAIN Multiclass Random Forest (attacks only)]\n")
            train_multiclass_rf_model()

        elif choice == "8":
            print("\n[Running Solution 3: TEST Multiclass Random Forest (attacks only)]\n")
            test_multiclass_rf_model()

        else:
            print("Invalid option. Please try again.\n")

if __name__ == "__main__":
    main()
