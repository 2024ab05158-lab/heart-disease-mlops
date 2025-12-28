"""
Unit tests for the FastAPI application
"""
import sys
from pathlib import Path
import pytest
import numpy as np
from fastapi.testclient import TestClient

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from api.main import app

client = TestClient(app)


def test_health_endpoint():
    """Test health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] == "ok"


def test_metrics_endpoint():
    """Test Prometheus metrics endpoint"""
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]
    assert "api_requests_total" in response.text


def test_predict_endpoint_valid():
    """Test prediction endpoint with valid data"""
    test_data = {
        "age": 45.0,
        "sex": 0.0,
        "cp": 1.0,
        "trestbps": 120.0,
        "chol": 200.0,
        "fbs": 0.0,
        "restecg": 0.0,
        "thalach": 180.0,
        "exang": 0.0,
        "oldpeak": 0.5,
        "slope": 1.0,
        "ca": 0.0,
        "thal": 3.0
    }
    
    response = client.post("/predict", json=test_data)
    assert response.status_code == 200
    data = response.json()
    assert "risk" in data
    assert "confidence" in data
    assert data["risk"] in [0, 1]
    assert 0.0 <= data["confidence"] <= 1.0


def test_predict_endpoint_invalid():
    """Test prediction endpoint with invalid data"""
    test_data = {
        "age": "invalid",
        "sex": 0.0
    }
    
    response = client.post("/predict", json=test_data)
    assert response.status_code == 422  # Validation error


def test_predict_endpoint_missing_fields():
    """Test prediction endpoint with missing required fields"""
    test_data = {
        "age": 45.0
    }
    
    response = client.post("/predict", json=test_data)
    assert response.status_code == 422  # Validation error


def test_api_docs():
    """Test that API documentation is accessible"""
    response = client.get("/docs")
    assert response.status_code == 200


def test_openapi_schema():
    """Test OpenAPI schema endpoint"""
    response = client.get("/openapi.json")
    assert response.status_code == 200
    schema = response.json()
    assert "openapi" in schema
    assert "info" in schema
    assert "paths" in schema

