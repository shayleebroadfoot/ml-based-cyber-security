# linsvm/preprocess.py

"""
Preprocessing and train/test splits for SVM models (Solution 2).

Right now this simply reuses the same cleaned features and splits
as Solution 1 (logistic regression), so there is a single source of
truth for:
- loading the CSVs
- feature engineering (log1p, bytes_total, grouping)
- ColumnTransformer (scaling + one-hot)

If we ever want SVM-specific tweaks (different scaling, feature
selection, etc.), we can change them here without touching Solution 1.
"""

from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

# Import the existing logic from Solution 1
from logreg.preprocess import (
    load_and_preprocess,       # returns (df, feature_names, preprocessor)
    get_binary_splits as _logreg_get_binary_splits,
    get_multiclass_splits as _logreg_get_multiclass_splits,
)


def get_binary_splits() -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, ColumnTransformer]:
    """
    Return (X_train, X_test, y_train, y_test, preprocessor) for
    binary SVM (attack vs normal).

    Currently identical to the logistic-regression splits; wrapped
    here so SVM has its own preprocessing module.
    """
    return _logreg_get_binary_splits()


def get_multiclass_splits() -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, ColumnTransformer]:
    """
    Return (X_train, X_test, y_train, y_test, preprocessor) for
    multiclass SVM (attack types, attacks only).

    Currently identical to the logistic-regression splits; wrapped
    here so SVM has its own preprocessing module.
    """
    return _logreg_get_multiclass_splits()
