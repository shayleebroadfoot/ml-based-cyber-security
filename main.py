# main.py

from trees.binary import train_binary_models
from trees.multiclass import train_multiclass_models
from trees.tuning import tune_binary_rf, tune_multiclass_rf
from trees.predict import predict_multiclass, predict_binary

if __name__ == "__main__":

    # -------------- TRAINING MODE --------------
    # print("=== Solution 2: Binary Models ===")
    # train_binary_models()
    #
    # print("\n=== Solution 2: Multiclass Models ===")
    # train_multiclass_models()

    # print("\n=== Hyperparameter Tuning ===")
    # # tune_binary_rf()
    # # tune_multiclass_rf()

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


