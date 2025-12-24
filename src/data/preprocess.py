import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.impute import SimpleImputer

def convert_to_numeric(X):
    """Convert DataFrame to numeric, replacing '?' with NaN"""
    if isinstance(X, pd.DataFrame):
        X = X.copy()
        # Replace '?' with NaN and convert to numeric
        X = X.replace('?', np.nan)
        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors='coerce')
    return X

def build_preprocessor():
    return Pipeline([
        ("converter", FunctionTransformer(convert_to_numeric, validate=False)),
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

