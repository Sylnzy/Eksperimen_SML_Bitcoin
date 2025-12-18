# Eksperimen Machine Learning - Bitcoin Price Prediction

## 📋 Deskripsi Proyek
Proyek ini merupakan implementasi sistem Machine Learning untuk memprediksi arah pergerakan harga Bitcoin (NAIK/TURUN) menggunakan MLOps pipeline yang lengkap.

## 🎯 Tujuan
- Membangun model prediksi harga Bitcoin dengan binary classification
- Mengimplementasikan MLOps best practices (CI/CD, monitoring, containerization)
- Mencapai target "Bintang 5 (Advanced)" pada submission Dicoding

## 📊 Dataset
- **Sumber**: Yahoo Finance (via yfinance)
- **Periode**: 2019-2025 (5 tahun data)
- **Total Records**: ~2500 hari trading
- **Features**: 12 technical indicators (MA_7, MA_30, RSI, MACD, dll)
- **Target**: Binary (0=TURUN, 1=NAIK)

## 🛠️ Tech Stack
- **ML Framework**: scikit-learn, pandas, numpy
- **Experiment Tracking**: MLflow + DagsHub
- **CI/CD**: GitHub Actions
- **Containerization**: Docker + Docker Hub
- **Monitoring**: Prometheus + Grafana
- **Visualization**: Plotly, Matplotlib, Seaborn

## 📁 Struktur Proyek
```
├── Salinan_dari_Template_Eksperimen_MSML.ipynb  # Notebook eksperimen (EDA + Preprocessing)
├── automate_preprocessing.py                     # Script otomasi preprocessing
├── requirements.txt                              # Python dependencies
├── btc_raw.csv                                   # Data mentah Bitcoin
├── btc_preprocessing.csv                         # Data hasil preprocessing
└── README.md                                     # Dokumentasi
```

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Preprocessing
```bash
python automate_preprocessing.py
```

### 3. Explore Notebook
```bash
jupyter notebook Salinan_dari_Template_Eksperimen_MSML.ipynb
```

## 📈 Fitur Utama
1. **Automated Data Pipeline**: Download & preprocessing otomatis via GitHub Actions
2. **Technical Indicators**: 7 indikator teknikal (MA, RSI, MACD, dll)
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
