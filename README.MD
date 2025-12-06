# Intrusion Detection System

This project implements two machine-learning solutions for detecting network intrusions using the provided dataset. The system performs both binary classification (normal vs attack) and multiclass classification (attack categories only).

The final version of the project focuses on two main solutions:

1. Kernel SVM (RBF) – nonlinear classifier  
2. Random Forest – tree-based model

Both solutions have separate training and testing pipelines and share a simple menu-based interface in `main.py`.

---

## Project Structure

~~~text
project/
│
├── main.py
│
├── kernelsvm/
│   ├── preprocess.py
│   ├── binary.py
│   └── multiclass.py
│
├── trees/
│   ├── preprocess.py
│   ├── binary.py
│   └── multiclass.py
│
└── data/
    ├── training_data.csv
    └── feature_description.csv
~~~

---

## Solution 1 — Kernel SVM (RBF)

This solution uses a nonlinear SVM with an RBF kernel.  
The preprocessing for this model includes:

- Cleaning and normalizing categorical string values  
- Grouping categorical features into reduced category sets  
- Log-transform on nonnegative numeric columns  
- Standard scaling for all numeric features  
- One-hot encoding for grouped categorical features  

### Notes

- The binary model uses all rows.  
- The multiclass model removes all Normal rows and trains only on attack categories.  
- This solution is more computationally expensive but captures nonlinear patterns well.

---

## Solution 2 — Random Forest

This solution uses Random Forest models for both binary and multiclass tasks.

Key preprocessing characteristics:

- Minimal preprocessing (trees handle unscaled features)
- One-hot encoding where needed
- Multiclass version trained on attack rows only
- Faster training and prediction compared to SVM

Random Forest provides solid baseline performance with low preprocessing cost.

---

## Running the Project

### Dependencies

Install required packages:

~~~bash
pip install numpy pandas scikit-learn joblib
~~~

### Start the Program

~~~bash
python main.py
~~~

You will see a menu like this:

~~~text
=== SENG 4610 Intrusion Detection Project ===

Solution 1 – Kernel SVM
 1) Train Binary Kernel SVM (RBF)
 2) Test  Binary Kernel SVM (RBF)
 3) Train Multiclass Kernel SVM (RBF)
 4) Test  Multiclass Kernel SVM (RBF)

Solution 2 – Random Forest
 5) Train Binary Random Forest
 6) Test  Binary Random Forest
 7) Train Multiclass Random Forest
 8) Test  Multiclass Random Forest

 0) Exit
~~~

---

## Model Outputs

When models are trained, they are saved automatically:

~~~text
kernelsvm/models/
trees/models/
~~~

Testing prints metrics such as:

- Accuracy  
- Precision, Recall, F1  
- Confusion matrix  
- RMSE  
- Model size and prediction memory usage  

---

## Data Requirements

Place the following files inside `data/`:

- `training_data.csv`  
- `feature_description.csv`

Both files are required by the preprocessing modules.

---

## Notes

- Stratified splits are used for train/test to preserve class balance.  
- Multiclass classification excludes the Normal class entirely.  
- Each solution has its own preprocessing pipeline and they are not interchangeable.
