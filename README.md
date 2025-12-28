# Heart Disease Prediction 

### Prerequisites
- Python 3.9+
- Docker Desktop (for deployment)

### Installation

```powershell
# 1. Create virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train the model
python src\models\train.py

# 4. Deploy with Docker
docker build -f docker/Dockerfile -t heart-disease-api .
docker run -d -p 8000:8000 --name heart-disease-api heart-disease-api
```

**For detailed setup instructions, see [SETUP.md](SETUP.md)**


## 📁 Project Structure

```
heart-disease-mlops/
├── api/                  # FastAPI application
│   └── main.py
├── data/                 # Dataset
│   └── raw/
│       ├── download_data.py
│       └── heart.csv
├── docker/               # Docker configuration
│   └── Dockerfile
├── notebooks/            # EDA scripts
│   └── 01_eda.py
├── src/                  # Source code
│   ├── data/
│   │   └── preprocess.py
│   └── models/
│       └── train.py
├── tests/                # Unit tests
│   ├── test_preprocessing.py
│   ├── test_api.py
│   └── test_model.py
├── .github/              # CI/CD workflows
│   └── workflows/
│       └── ci.yml
├── requirements.txt
├── docker-compose.yml
└── model.joblib          # Trained model (after training)
```

## 🔌 API Endpoints

### Base URL
```
http://localhost:8000
```

### Endpoints

- **GET /health** - Health check
- **POST /predict** - Make prediction
- **GET /metrics** - Prometheus metrics
- **GET /docs** - Interactive API documentation

### Example Prediction

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
  -Method Post -ContentType "application/json" -Body $body
```

**Response:**
```json
{
  "risk": 0,
  "confidence": 0.05
}
```

---

## 📊 Model Performance

**Response:**
```json
{
  "risk": 0,
  "confidence": 0.44
}
```
*Interpretation: Moderate risk (44% confidence of heart disease, below 50% threshold)*

---

## 🐳 Docker Deployment

### Build and Run

```powershell
# Build image
docker build -f docker/Dockerfile -t heart-disease-api .

# Run container
docker run -d -p 8000:8000 --name heart-disease-api heart-disease-api

# Verify
curl http://localhost:8000/health
```

### Using Docker Compose

```powershell
docker-compose up -d --build
```

## 🧪 Testing

Run tests:
```powershell
pytest tests\ -v
```

Run with coverage:
```powershell
pytest tests\ -v --cov=src --cov=api
```

---

## 📈 MLflow Tracking

View experiment tracking:
```powershell
mlflow ui
```

Then visit http://127.0.0.1:5000

---

## 📋 Feature Descriptions

| Feature | Description |
|---------|-------------|
| age | Age in years |
| sex | 0 = Female, 1 = Male |
| cp | Chest pain type (1-4) |
| trestbps | Resting blood pressure (mm Hg) |
| chol | Serum cholesterol (mg/dl) |
| fbs | Fasting blood sugar > 120 (0/1) |
| restecg | Resting ECG results (0,1,2) |
| thalach | Maximum heart rate achieved |
| exang | Exercise induced angina (0/1) |
| oldpeak | ST depression |
| slope | Slope of peak exercise ST segment |
| ca | Number of major vessels (0-3) |
| thal | Thalassemia (3,6,7) |


## 🛠️ Technologies Used

- **Python 3.9+**
- **scikit-learn** - Machine learning
- **FastAPI** - API framework
- **MLflow** - Experiment tracking
- **Docker** - Containerization
- **Prometheus** - Metrics
- **pytest** - Testing
- **GitHub Actions** - CI/CD

---




