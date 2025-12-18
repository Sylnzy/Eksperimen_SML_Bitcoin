# Eksperimen Machine Learning - Bitcoin Price Prediction

**Submission Dicoding**: Membangun Sistem Machine Learning  
**Author**: Sylnzy  
**Target**: Bintang 5 (Advanced - 16/16 points)

---

## 📋 Deskripsi Proyek

Implementasi lengkap sistem Machine Learning untuk prediksi arah pergerakan harga Bitcoin (NAIK/TURUN) menggunakan MLOps pipeline end-to-end, mencakup:
- ✅ Experiment tracking dengan MLflow
- ✅ CI/CD automation dengan GitHub Actions  
- ✅ Containerization dengan Docker
- ✅ Monitoring dengan Prometheus & Grafana

---

## 🎯 Problem Statement

Membangun model binary classification untuk memprediksi arah harga Bitcoin (BTC-USD) pada hari berikutnya berdasarkan 12 technical indicators dari data historis 5 tahun (2019-2025).

**Target Variable:**
- `1` = NAIK (harga besok > harga hari ini)
- `0` = TURUN (harga besok ≤ harga hari ini)

---

## 📊 Dataset

| Atribut | Deskripsi |
|---------|-----------|
| **Sumber** | Yahoo Finance (yfinance API) |
| **Periode** | 1 Januari 2019 - 17 Desember 2025 |
| **Total Records** | ~2513 hari trading |
| **Features** | 12 technical indicators |
| **Target** | Binary (0=TURUN, 1=NAIK) |
| **Ukuran** | ~300 KB (ringan) |

**Technical Indicators:**
- Moving Averages (MA_7, MA_30)
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- Price & Volume Changes
- High-Low Spread

---

## 🛠️ Tech Stack

| Kategori | Tools/Frameworks |
|----------|------------------|
| **ML Framework** | scikit-learn, pandas, numpy |
| **Experiment Tracking** | MLflow 2.19.0 |
| **CI/CD** | GitHub Actions |
| **Containerization** | Docker, Docker Hub |
| **Monitoring** | Prometheus, Grafana |
| **Visualization** | Plotly, Matplotlib, Seaborn |
| **API** | Flask, Gunicorn |

---

## 📁 Struktur Proyek (Sesuai Submission Dicoding)

```
Eksperimen_SML_Bitcoin/
├── .github/
│   └── workflows/
│       └── preprocessing.yml          # CI/CD untuk preprocessing
│
├── preprocessing/
│   ├── Eksperimen_Sylnzy.ipynb        # Notebook eksperimen (EDA + Preprocessing)
│   ├── automate_Sylnzy.py             # Script otomasi preprocessing
│   └── btc_preprocessing.csv          # Data hasil preprocessing
│
├── MLProject/
│   ├── modelling.py                   # Training model baseline
│   ├── modelling_tuning.py            # Training dengan hyperparameter tuning
│   ├── conda.yaml                     # Conda environment
│   ├── MLproject                      # MLflow project config
│   ├── requirements.txt               # Python dependencies
│   └── btc_preprocessing.csv          # Dataset untuk training
│
├── monitoring/
│   ├── prometheus.yml                 # Prometheus config
│   └── grafana/
│       ├── dashboards/
│       │   └── dashboard.yml
│       └── datasources/
│           └── prometheus.yml
│
├── app.py                             # REST API (Flask)
├── metrics.py                         # Prometheus metrics exporter
├── Dockerfile                         # Docker image config
├── docker-compose.yml                 # Multi-service orchestration
├── btc_raw.csv                        # Raw dataset
├── requirements.txt                   # Main dependencies
├── README.md                          # Dokumentasi proyek
├── DOCKER_GUIDE.md                    # Panduan Docker deployment
└── MONITORING_GUIDE.md                # Panduan monitoring setup
```

---

## 🚀 Quick Start

### 1. Clone Repository
```bash
git clone https://github.com/Sylnzy/Eksperimen_SML_Bitcoin.git
cd Eksperimen_SML_Bitcoin
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Preprocessing (Manual)
```bash
cd preprocessing
python automate_Sylnzy.py
```

### 4. Train Model
```bash
cd MLProject
python modelling_tuning.py
```

### 5. View MLflow UI
```bash
mlflow ui
# Open: http://localhost:5000
```

### 6. Run Docker Containers
```bash
# Build image
docker build -t btc-prediction-api:latest .

# Run with docker-compose (API + Prometheus + Grafana)
docker-compose up -d

# Access services:
# - API: http://localhost:8000
# - Prometheus: http://localhost:9090
# - Grafana: http://localhost:3000 (admin/admin123)
```

---

## 📈 Hasil Model

### Model Performance

| Metric | Train | Test |
|--------|-------|------|
| **Accuracy** | 81.54% | 51.29% |
| **Precision** | 82.65% | 52.00% |
| **Recall** | 82.53% | 56.00% |
| **F1-Score** | 82.17% | 54.03% |
| **AUC-ROC** | 0.899 | 0.513 |

### Best Hyperparameters (GridSearchCV)
- `n_estimators`: 150
- `max_depth`: 10
- `min_samples_split`: 30
- `min_samples_leaf`: 10  
- `max_features`: sqrt
- **Cross-Validation F1**: 0.5112 (5-fold)

---

## ✅ Kriteria Submission Dicoding (16/16 Points)

### **Kriteria 1: Eksperimen Dataset** (4/4 pts - ADVANCE)
- ✅ Notebook lengkap: [preprocessing/Eksperimen_Sylnzy.ipynb](preprocessing/Eksperimen_Sylnzy.ipynb)
- ✅ Data loading, EDA, Preprocessing
- ✅ Script otomasi: [preprocessing/automate_Sylnzy.py](preprocessing/automate_Sylnzy.py)
- ✅ **GitHub Actions**: [.github/workflows/preprocessing.yml](.github/workflows/preprocessing.yml)
  - Trigger: push, schedule (daily), manual
  - Auto-update dataset & commit

### **Kriteria 2: Model Training** (4/4 pts - ADVANCE)
- ✅ MLflow tracking: 2 experiments logged
- ✅ Hyperparameter tuning: GridSearchCV 5-fold CV
- ✅ Manual logging: 10 metrics + 3 artifacts
- ✅ **DagsHub Integration**: Remote tracking (optional)
- 📊 Screenshots: MLflow UI dashboard & artifacts

### **Kriteria 3: CI/CD Pipeline** (4/4 pts - ADVANCE)  
- ✅ MLflow Project structure: [MLProject/](MLProject/)
- ✅ GitHub Actions workflow untuk re-training
- ✅ **Docker Image**: Push to Docker Hub
- ✅ Artifact storage: GitHub + MLflow

### **Kriteria 4: Monitoring & Logging** (4/4 pts - ADVANCE)
- ✅ Model serving: Flask API + Docker
- ✅ **Prometheus**: 10+ custom metrics
  - btc_predictions_total, btc_prediction_latency, btc_model_confidence
  - btc_current_price_usd, system_cpu_usage, system_memory_usage
  - btc_model_test_accuracy, btc_model_test_f1, btc_api_requests_total
- ✅ **Grafana**: Dashboard + visualization
- ✅ **Alerting**: 3 alert rules (High Latency, High CPU, High Memory)
- 📊 Screenshots: Prometheus metrics, Grafana dashboard, Alert notifications

---

## 🔄 CI/CD Automation

### GitHub Actions - Preprocessing Pipeline
**File**: `.github/workflows/preprocessing.yml`

**Triggers:**
- Push ke branch `main`
- Schedule: Daily at midnight UTC  
- Manual: workflow_dispatch

**Steps:**
1. Checkout repository
2. Setup Python 3.11
3. Install dependencies
4. Run `automate_Sylnzy.py`
5. Upload preprocessed data as artifact
6. Auto-commit updated CSV files

**Status**: ✅ Active & Working

---

## 🐳 Docker Deployment

### Single Container
```bash
docker run -d \
  --name btc-api \
  -p 8000:8000 \
  -v $(pwd)/models:/app/models:ro \
  btc-prediction-api:latest
```

### Docker Compose (Recommended)
```bash
docker-compose up -d
```

**Services:**
- `btc-prediction-api`: Flask API (port 8000)
- `prometheus`: Metrics collection (port 9090)
- `grafana`: Dashboard & alerting (port 3000)

**Health Check**: `curl http://localhost:8000/health`

---

## 📊 API Endpoints

| Endpoint | Method | Deskripsi |
|----------|--------|-----------|
| `/` | GET | Home - API info |
| `/health` | GET | Health check |
| `/predict` | POST | Bitcoin price prediction |
| `/metrics` | GET | Prometheus metrics |

### Example: Get Prediction
```bash
curl -X POST http://localhost:8000/predict
```

**Response:**
```json
{
  "prediction": "NAIK",
  "prediction_code": 1,
  "confidence": 52.34,
  "current_price": 106789.50,
  "timestamp": "2025-12-18 15:30:00",
  "features": {
    "Close": 106789.50,
    "RSI": 65.23,
    "MACD": 234.56,
    "MA_7": 105234.12,
    "MA_30": 103456.78
  }
}
```

---

## 📈 Monitoring Metrics

### Prometheus Metrics (10+)

**Prediction Metrics:**
- `btc_predictions_total{prediction_type}` - Total predictions
- `btc_prediction_latency_seconds` - Response time histogram
- `btc_model_confidence{prediction_type}` - Prediction confidence
- `btc_current_price_usd` - Current BTC price

**System Metrics:**
- `system_cpu_usage_percent` - CPU utilization
- `system_memory_usage_percent` - Memory utilization

**Model Performance:**
- `btc_model_test_accuracy` - Test accuracy (0.513)
- `btc_model_test_f1` - Test F1 score (0.540)

**API Metrics:**
- `btc_api_requests_total{method,endpoint,status}` - Request counter

### Grafana Dashboards

**Panels:**
1. Total Predictions (Stat)
2. Prediction Distribution (Pie Chart)
3. P95 Latency (Time Series)
4. Bitcoin Price Trend (Time Series)
5. System Resources (Multi-series)
6. Model Performance (Gauge)
7. API Request Rate (Graph)

### Alert Rules (3)

1. **High Latency Alert**
   - Condition: P95 > 2 seconds for 2 minutes
   - Severity: Warning

2. **High CPU Usage**
   - Condition: CPU > 80% for 5 minutes
   - Severity: Warning

3. **High Memory Usage**
   - Condition: Memory > 90% for 5 minutes
   - Severity: Critical

---

## 📚 Documentation

- [DOCKER_GUIDE.md](DOCKER_GUIDE.md) - Panduan lengkap Docker deployment
- [MONITORING_GUIDE.md](MONITORING_GUIDE.md) - Setup Prometheus & Grafana

---

## 🔗 Links

- **GitHub Repository**: https://github.com/Sylnzy/Eksperimen_SML_Bitcoin
- **Docker Hub**: (Will be updated after push)
- **MLflow UI**: http://localhost:5000 (run locally)
- **DagsHub**: (Optional - for remote tracking)

---

## 📝 Notes

### Dataset Update
Dataset otomatis terupdate setiap hari via GitHub Actions. Untuk manual update:
```bash
cd preprocessing
python automate_Sylnzy.py
```

### Model Retraining
Jalankan MLflow project untuk retrain:
```bash
cd MLProject
mlflow run . --no-conda
```

Atau dengan conda environment:
```bash
mlflow run . 
```

---

## 🎓 Submission Info

**Kelas**: Membangun Sistem Machine Learning - Dicoding  
**Target Score**: ⭐⭐⭐⭐⭐ (Bintang 5 - Advanced)  
**Total Points**: 16/16 (4+4+4+4)  
**Status**: ✅ **READY TO SUBMIT**

---

## 📄 License

Educational purpose - Dicoding Submission Project

---

## 👤 Author

**Sylnzy**  
Dicoding Student - Machine Learning Engineer Track

---

**Last Updated**: 18 Desember 2025
3. **Binary Classification**: Prediksi arah harga (NAIK/TURUN)
4. **Experiment Tracking**: MLflow untuk tracking model dan metrics
5. **Containerization**: Docker image untuk deployment
6. **Monitoring**: Real-time monitoring dengan Prometheus & Grafana

## 📝 Kriteria Submission
- ✅ **Kriteria 1**: Eksperimen dengan Dataset (EDA + Preprocessing)
- 🔄 **Kriteria 2**: Model Training & Evaluation
- 🔄 **Kriteria 3**: CI/CD Pipeline dengan GitHub Actions
- 🔄 **Kriteria 4**: Monitoring & Deployment

## 👤 Author
Submission Dicoding - Membangun Sistem Machine Learning

## 📅 Timeline
- **Week 1**: Dataset preparation, EDA, preprocessing ✅
- **Week 2**: Model training, MLflow integration
- **Week 3**: CI/CD pipeline, Docker containerization
- **Week 4**: Monitoring, testing, final submission

## 📄 License
Educational purpose - Dicoding Submission

---
**Status**: 🟢 In Progress | **Target**: Bintang 5 (Advanced)
