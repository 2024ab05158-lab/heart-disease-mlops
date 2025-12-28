"""
Unit tests for model training and inference
"""
import sys
from pathlib import Path
import pytest
import pandas as pd
import numpy as np
import joblib

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.preprocess import build_preprocessor


def test_preprocessor_output_shape():
    """Test that preprocessor outputs correct shape"""
    preprocessor = build_preprocessor()
    
    # Create test data
    X = pd.DataFrame({
        'feature1': [1, 2, 3, np.nan],
        'feature2': [4, 5, 6, 7]
    })
    
    X_transformed = preprocessor.fit_transform(X)
    
    assert X_transformed.shape[0] == X.shape[0]
    assert X_transformed.shape[1] == X.shape[1]
    assert not np.isnan(X_transformed).any()


def test_preprocessor_handles_missing_values():
    """Test that preprocessor handles missing values"""
    preprocessor = build_preprocessor()
    
    X = pd.DataFrame({
        'feature1': [1, np.nan, 3],
        'feature2': [4, 5, np.nan]
    })
    
    X_transformed = preprocessor.fit_transform(X)
    
    # Should not have NaN after transformation
    assert not np.isnan(X_transformed).any()


def test_model_file_exists():
    """Test that model file exists and can be loaded"""
    model_path = project_root / "model.joblib"
    
    if model_path.exists():
        model = joblib.load(model_path)
        assert model is not None
        assert hasattr(model, 'predict')
        assert hasattr(model, 'predict_proba')
    else:
        pytest.skip("model.joblib not found. Run training first.")


def test_model_prediction_shape():
    """Test that model produces correct prediction shape"""
    model_path = project_root / "model.joblib"
    
    if not model_path.exists():
        pytest.skip("model.joblib not found. Run training first.")
    
    model = joblib.load(model_path)
    
    # Create test input (13 features for heart disease dataset)
    X_test = np.array([[63, 1, 1, 145, 233, 1, 2, 150, 0, 2.3, 3, 0, 6]])
    
    prediction = model.predict(X_test)
    probabilities = model.predict_proba(X_test)
    
    assert prediction.shape == (1,)
    assert probabilities.shape == (1, 2)
    assert prediction[0] in [0, 1]
    assert np.allclose(probabilities.sum(axis=1), 1.0)


def test_model_prediction_range():
    """Test that model predictions are in valid range"""
    model_path = project_root / "model.joblib"
    
    if not model_path.exists():
        pytest.skip("model.joblib not found. Run training first.")
    
    model = joblib.load(model_path)
    
    X_test = np.array([[63, 1, 1, 145, 233, 1, 2, 150, 0, 2.3, 3, 0, 6]])
    
    probabilities = model.predict_proba(X_test)
    
    assert np.all(probabilities >= 0)
    assert np.all(probabilities <= 1)

