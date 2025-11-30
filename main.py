# main.py

import time
import tracemalloc

from trees.binary import train_binary_models
from trees.multiclass import train_multiclass_models
from trees.tuning import tune_binary_rf, tune_multiclass_rf
from trees.predict import predict_multiclass, predict_binary


if __name__ == "__main__":

    tracemalloc.start()

    # -------------- TRAINING MODE --------------
    # Uncomment this block when you want to retrain and see training cost.

    print("=== Solution 2: Binary Models (training) ===")
    start_bin = time.perf_counter()
    train_binary_models()
    end_bin = time.perf_counter()
    print(f"Total binary training time: {end_bin - start_bin:.3f} s\n")

    print("=== Solution 2: Multiclass Models (training) ===")
    start_multi = time.perf_counter()
    train_multiclass_models()
    end_multi = time.perf_counter()
    print(f"Total multiclass training time: {end_multi - start_multi:.3f} s\n")

    # print("\n=== Hyperparameter Tuning (optional, off-device) ===")
    # tune_binary_rf()
    # tune_multiclass_rf()

    # -------------- PREDICT MODE --------------
    print("=== Binary predictions from saved model ===")
    preds_bin = predict_binary(
        data_path="data/training_data.csv",
        model_path="trees/models/rf_binary.joblib",
        evaluate=True
    )

    print("First 20 binary predictions:", preds_bin[:20])

    print("\n=== Multiclass predictions from saved model ===")

    preds_multi = predict_multiclass(
        data_path="data/training_data.csv",
        model_path="trees/models/rf_multiclass.joblib",
        evaluate=True
    )

    print("First 20 multiclass predictions:", preds_multi[:20])

    # -------------- MEMORY SUMMARY --------------
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print("\n=== Memory usage (Python process, rough) ===")
    print(f"Current: {current / (1024**2):.2f} MB")
    print(f"Peak   : {peak / (1024**2):.2f} MB")
