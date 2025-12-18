"""
Bitcoin Price Prediction API Service
Author: Sylnzy
Date: 2025-12-18

REST API for serving Bitcoin price predictions using trained model.
"""

from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# Load model and scaler
MODEL_PATH = 'models/btc_model_tuned.pkl'
SCALER_PATH = 'models/scaler_tuned.pkl'

try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print(f"Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None
    scaler = None


def calculate_rsi(data, window=14):
    """Calculate RSI indicator"""
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def get_latest_btc_data():
    """Download latest Bitcoin data and calculate features"""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=60)  # Get 60 days for indicators
    
    btc_data = yf.download('BTC-USD', 
                          start=start_date.strftime('%Y-%m-%d'),
                          end=end_date.strftime('%Y-%m-%d'),
                          progress=False)
    
    if btc_data.empty:
        raise ValueError("Failed to download Bitcoin data")
    
    # Calculate technical indicators
    df = btc_data.copy()
    df['MA_7'] = df['Close'].rolling(window=7).mean()
    df['MA_30'] = df['Close'].rolling(window=30).mean()
    df['RSI'] = calculate_rsi(df['Close'])
    df['Price_Change'] = df['Close'].pct_change() * 100
    df['Volume_Change'] = df['Volume'].pct_change() * 100
    df['HL_Spread'] = df['High'] - df['Low']
    
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    
    # Get latest complete row (no NaN)
    df = df.dropna()
    latest = df.iloc[-1]
    
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 
                'MA_7', 'MA_30', 'RSI', 'Price_Change', 
                'Volume_Change', 'HL_Spread', 'MACD']
    
    return latest[features], latest['Close'], latest.name


@app.route('/')
def home():
    """API home endpoint"""
    return jsonify({
        'service': 'Bitcoin Price Prediction API',
        'version': '1.0',
        'status': 'running',
        'model': 'Random Forest Classifier',
        'endpoints': {
            '/predict': 'POST - Get price direction prediction',
            '/health': 'GET - Health check',
            '/metrics': 'GET - Model metrics'
        }
    })


@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    })


@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict Bitcoin price direction
    
    Request body (optional):
    {
        "custom_data": {...}  // Optional: provide custom feature values
    }
    
    Returns:
    {
        "prediction": "NAIK" or "TURUN",
        "confidence": float,
        "current_price": float,
        "timestamp": str
    }
    """
    if model is None or scaler is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        data = request.get_json() if request.is_json else {}
        
        if 'custom_data' in data:
            # Use custom provided data
            features_dict = data['custom_data']
            features = pd.Series(features_dict)
            current_price = features_dict.get('Close', 0)
            timestamp = datetime.now()
        else:
            # Download latest data
            features, current_price, timestamp = get_latest_btc_data()
        
        # Prepare features for prediction
        features_array = features.values.reshape(1, -1)
        features_scaled = scaler.transform(features_array)
        
        # Make prediction
        prediction = model.predict(features_scaled)[0]
        prediction_proba = model.predict_proba(features_scaled)[0]
        
        # Get confidence
        confidence = float(prediction_proba[int(prediction)])
        
        # Format response
        result = {
            'prediction': 'NAIK' if prediction == 1 else 'TURUN',
            'prediction_code': int(prediction),
            'confidence': round(confidence * 100, 2),
            'current_price': float(current_price),
            'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'features': {
                'Close': float(features['Close']),
                'RSI': float(features['RSI']),
                'MACD': float(features['MACD']),
                'MA_7': float(features['MA_7']),
                'MA_30': float(features['MA_30'])
            }
        }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/metrics')
def metrics():
    """Get model performance metrics (from training)"""
    metrics_data = {
        'model_type': 'Random Forest Classifier',
        'training_date': '2025-12-18',
        'test_metrics': {
            'accuracy': 0.513,
            'f1_score': 0.540,
            'precision': 0.52,
            'recall': 0.56
        },
        'hyperparameters': {
            'n_estimators': 150,
            'max_depth': 10,
            'min_samples_split': 30,
            'min_samples_leaf': 10,
            'max_features': 'sqrt'
        },
        'dataset': {
            'total_samples': 2513,
            'features': 12,
            'train_size': 2010,
            'test_size': 503
        }
    }
    
    return jsonify(metrics_data)


if __name__ == '__main__':
    print("="*50)
    print("Bitcoin Price Prediction API")
    print("="*50)
    print(f"Starting server at http://0.0.0.0:8000")
    print("Press CTRL+C to stop")
    print("="*50 + "\n")
    
    app.run(host='0.0.0.0', port=8000, debug=False)
