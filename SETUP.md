# Complete Setup & Deployment Guide


## 📋 Table of Contents

1. [Prerequisites](#prerequisites)
2. [Initial Setup](#initial-setup)
3. [Train the Model](#train-the-model)
4. [Run Locally](#run-locally)
5. [Deploy with Docker](#deploy-with-docker)
6. [Test the API](#test-the-api)
7. [Troubleshooting](#troubleshooting)



## 1. Prerequisites

- **Python 3.9+** installed
- **Docker Desktop** (for Docker deployment - optional but recommended)
- **Git** (if cloning from repository)


## 2. Initial Setup

### Step 1: Navigate to Project

```powershell
cd D:\Bits\Semester-3\MLOps\A-1\heart-disease-mlops
```

### Step 2: Create Virtual Environment

```powershell
python -m venv venv
```

### Step 3: Activate Virtual Environment

**PowerShell:**
```powershell
.\venv\Scripts\Activate.ps1
```

If you get execution policy error:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**Command Prompt:**
```cmd
venv\Scripts\activate.bat
```

### Step 4: Install Dependencies

```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 3. Train the Model

**Important:** You must train the model before running the API!

```powershell
python src\models\train.py
```

**What this does:**
- Downloads/loads the dataset
- Trains Logistic Regression and Random Forest models
- Evaluates with cross-validation
- Logs experiments to MLflow
- Saves the best model as `model.joblib`

**Expected output:**
- Model training progress
- Cross-validation scores
- Final metrics (Accuracy, Precision, Recall, F1, ROC-AUC)
- `model.joblib` file created in project root

**Verify model exists:**
```powershell
dir model.joblib
```

---

## 4. Run Locally

### Option A: Direct Python

```powershell
# Make sure virtual environment is activated
uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
```

### Option B: Run EDA First (Optional)

```powershell
python notebooks\01_eda.py
```

This generates visualization files in the `notebooks/` directory.

**Access:**
- API: http://127.0.0.1:8000
- Docs: http://127.0.0.1:8000/docs
- Health: http://127.0.0.1:8000/health
- Metrics: http://127.0.0.1:8000/metrics

---

## 5. Deploy with Docker

### Prerequisites
- Docker Desktop installed and running
- Model trained (`model.joblib` exists)

### Step 1: Verify Docker is Running

```powershell
docker ps
```

If error, start Docker Desktop and wait for it to fully start.

### Step 2: Build Docker Image

```powershell
docker build -f docker/Dockerfile -t heart-disease-api .
```

**Expected output:**
- Image building progress
- Success message with image ID

### Step 3: Run Container

```powershell
docker run -d -p 8000:8000 --name heart-disease-api heart-disease-api
```

**Verify container is running:**
```powershell
docker ps
```

You should see `heart-disease-api` in the list.

### Alternative: Docker Compose

```powershell
docker-compose up -d --build
```

### Step 4: View Logs 

```powershell
docker logs heart-disease-api
```

### Step 5: Stop Container (When Done)

```powershell
docker stop heart-disease-api
docker rm heart-disease-api
```

Or in one command:
```powershell
docker rm -f heart-disease-api
```

---

## 6. Test the API

### Health Check

```powershell
# Using curl
curl http://localhost:8000/health

# Using PowerShell
Invoke-WebRequest -Uri http://localhost:8000/health

# Expected: {"status":"ok","model_loaded":true}
```

### View API Documentation

Open in browser: **http://localhost:8000/docs**

### View Metrics

Open in browser: **http://localhost:8000/metrics**

### Test Prediction

**Using PowerShell:**
```powershell
$body = @{
    age = 45.0
    sex = 0.0
    cp = 1.0
    trestbps = 120.0
    chol = 200.0
    fbs = 0.0
    restecg = 0.0
    thalach = 180.0
    exang = 0.0
    oldpeak = 0.5
    slope = 1.0
    ca = 0.0
    thal = 3.0
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://localhost:8000/predict" `
  -Method Post `
  -ContentType "application/json" `
  -Body $body
```

**Expected response:**
```json
{
  "risk": 0,
  "confidence": 0.05
}
```

**Using curl:**
```powershell
curl -X POST "http://localhost:8000/predict" `
  -H "Content-Type: application/json" `
  -d '{"age":45.0,"sex":0.0,"cp":1.0,"trestbps":120.0,"chol":200.0,"fbs":0.0,"restecg":0.0,"thalach":180.0,"exang":0.0,"oldpeak":0.5,"slope":1.0,"ca":0.0,"thal":3.0}'
```
