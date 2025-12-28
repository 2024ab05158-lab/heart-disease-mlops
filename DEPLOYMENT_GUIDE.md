# Deployment Guide - Heart Disease MLOps API

This guide provides step-by-step instructions for deploying the Heart Disease Prediction API.

## Table of Contents

1. [Local Deployment](#local-deployment)
2. [Docker Deployment](#docker-deployment)
3. [Docker Compose Deployment](#docker-compose-deployment)
4. [Verification](#verification)

---

## 1. Local Deployment

### Prerequisites
- Python 3.9+
- Virtual environment

### Steps

```powershell
# 1. Create virtual environment
python -m venv venv

# 2. Activate virtual environment
.\venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install -r requirements.txt

# 4. Train the model
python src\models\train.py

# 5. Start the API
uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
```

**Access:**
- API: http://127.0.0.1:8000
- Docs: http://127.0.0.1:8000/docs
- Metrics: http://127.0.0.1:8000/metrics
- Health: http://127.0.0.1:8000/health

---

## 2. Docker Deployment

### Prerequisites
- Docker Desktop installed and running
- Model trained (`model.joblib` exists)

### Steps

```powershell
# 1. Ensure model is trained
python src\models\train.py

# 2. Verify model file exists
dir model.joblib

# 3. Build Docker image
docker build -f docker/Dockerfile -t heart-disease-api .

# 4. Run container
docker run -d -p 8000:8000 --name heart-disease-api heart-disease-api

# 5. Verify container is running
docker ps

# 6. Check logs (optional)
docker logs heart-disease-api
docker logs -f heart-disease-api
docker logs --tail 50 heart-disease-api 
```

### Stop and Remove Container

```powershell
# Stop container
docker stop heart-disease-api

# Remove container
docker rm heart-disease-api

# Or stop and remove in one command
docker rm -f heart-disease-api
```

---

## 3. Docker Compose Deployment

### Prerequisites
- Docker Desktop installed and running
- Docker Compose (included with Docker Desktop)

### Steps

```powershell
# 1. Ensure model is trained
python src\models\train.py

# 2. Build and start services
docker-compose up -d --build

# 3. View logs
docker-compose logs -f

# 4. Stop services
docker-compose down
```

---

## 4. Verification

After deployment, verify the API is working:

### Health Check

```powershell
# Using curl
curl http://localhost:8000/health

# Using PowerShell
Invoke-WebRequest -Uri http://localhost:8000/health

# Expected response:
# {"status":"ok","model_loaded":true}
```

### API Documentation

Visit in browser: http://localhost:8000/docs

### Metrics Endpoint

```powershell
# View Prometheus metrics
curl http://localhost:8000/metrics

# Or visit in browser
# http://localhost:8000/metrics
```

### Test Prediction

```powershell
# Using curl
curl -X POST "http://localhost:8000/predict" `
  -H "Content-Type: application/json" `
  -d '{
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
  }'

# Expected response:
# {"risk":0,"confidence":0.05}
```

### Using PowerShell Invoke-RestMethod

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

---

## Troubleshooting

### Container won't start

**Check if model file exists:**
```powershell
dir model.joblib
```

If missing, train the model:
```powershell
python src\models\train.py
```

**Check container logs:**
```powershell
docker logs heart-disease-api
```

### Port 8000 already in use

**Option 1: Stop other service using port 8000**

**Option 2: Use different port:**
```powershell
docker run -d -p 8001:8000 --name heart-disease-api heart-disease-api
# Then access at http://localhost:8001
```

### Image build fails

**Check Docker is running:**
```powershell
docker ps
```

**Check Dockerfile path:**
```powershell
# Make sure you're in project root
docker build -f docker/Dockerfile -t heart-disease-api .
```

### API returns errors

**Check logs:**
```powershell
docker logs -f heart-disease-api
```

**Verify model loaded:**
```powershell
curl http://localhost:8000/health
# Should show: {"status":"ok","model_loaded":true}
```

---

## Next Steps

- Set up monitoring with Prometheus (optional - metrics endpoint already available)
- Configure logging aggregation
- Set up CI/CD for automatic deployments
- Scale with multiple containers (using Docker Compose or orchestration)

---

## Quick Reference

### Docker Commands

```powershell
# Build
docker build -f docker/Dockerfile -t heart-disease-api .

# Run
docker run -d -p 8000:8000 --name heart-disease-api heart-disease-api

# Stop
docker stop heart-disease-api

# Remove
docker rm heart-disease-api

# View logs
docker logs -f heart-disease-api

# List containers
docker ps

# List images
docker images
```

### Docker Compose Commands

```powershell
# Start
docker-compose up -d --build

# Stop
docker-compose down

# View logs
docker-compose logs -f

# Restart
docker-compose restart
```

---

**For more details, see:**
- `DOCKER_SETUP.md` - Detailed Docker installation guide
- `DEPLOYMENT_OPTIONS.md` - Deployment options explained
- `README.md` - Main project documentation
