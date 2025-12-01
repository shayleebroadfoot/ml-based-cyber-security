# from trees.binary import train_binary_model, test_binary_model
# from trees.multiclass import train_multiclass_model, test_multiclass_model
# from trees.multiclass_tree import train_multiclass_tree_model, test_multiclass_tree_model
# from trees.tuning import tune_multiclass_rf, tune_multiclass_tree
#
# if __name__ == "__main__":
#
#     # _, rf_params, _ = tune_multiclass_rf(save_model=False)
#     # _, tree_params, _ = tune_multiclass_tree(save_model=False)
#     # Train RF + Tree
#     # train_binary_model()
#     # train_multiclass_model()  # RF
#     train_multiclass_tree_model()  # DT
#     #
#     # # Test RF + Tree
#     # # test_binary_model()
#     # # test_multiclass_model()  # RF
#     test_multiclass_tree_model()  # DT

# main.py


from logreg.binary import run_binary_logreg
from logreg.multiclass import run_multiclass_logreg


def main():
    print("=== SENG 4610 Intrusion Detection Project ===")
    print("1) Run Solution 1 - Binary Logistic Regression")
    print("2) Run Solution 1 - Multiclass Logistic Regression (attacks only)")
    print("3) Run both (binary then multiclass)")
    print("0) Exit")

    choice = input("Select an option: ").strip()

    if choice == "1":
        print("\n[Running Solution 1: Binary Logistic Regression]\n")
        run_binary_logreg()
    elif choice == "2":
        print("\n[Running Solution 1: Multiclass Logistic Regression]\n")
        run_multiclass_logreg()
    elif choice == "3":
        print("\n[Running Solution 1: Binary Logistic Regression]\n")
        run_binary_logreg()
        print("\n[Running Solution 1: Multiclass Logistic Regression]\n")
        run_multiclass_logreg()
    elif choice == "0":
        print("Exiting.")
    else:
        print("Invalid choice.")


if __name__ == "__main__":
    main()


