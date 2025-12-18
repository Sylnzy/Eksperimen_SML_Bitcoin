"""
Improved Bitcoin Price Prediction with Hyperparameter Tuning
Author: Sylnzy
Date: 2025-12-18

This script includes:
- Grid search for hyperparameter tuning
- Cross-validation
- Regularization to prevent overfitting
"""

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from datetime import datetime
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)


def load_preprocessed_data(filepath='btc_preprocessing.csv'):
    """Load and prepare data"""
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath, index_col=0, parse_dates=True)
    df = df.dropna()
    
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 
                'MA_7', 'MA_30', 'RSI', 'Price_Change', 
                'Volume_Change', 'HL_Spread', 'MACD']
    
    X = df[features]
    y = df['Target']
    
    print(f"Data loaded: {X.shape[0]} samples, {X.shape[1]} features")
    return X, y


def train_improved_model():
    """Train improved model with cross-validation"""
    
    print("="*50)
    print("IMPROVED MODEL TRAINING WITH HYPERPARAMETER TUNING")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*50 + "\n")
    
    mlflow.set_experiment("Bitcoin_Price_Prediction_Improved")
    
    with mlflow.start_run(run_name=f"RF_Tuned_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        
        # Load data
        X, y = load_preprocessed_data()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        print(f"Train set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples\n")
        
        # Define parameter grid for tuning
        param_grid = {
            'n_estimators': [50, 100, 150],
            'max_depth': [10, 15, 20],
            'min_samples_split': [10, 20, 30],
            'min_samples_leaf': [5, 10, 15],
            'max_features': ['sqrt', 'log2']
        }
        
        print("Starting Grid Search...")
        print(f"Parameter grid: {param_grid}\n")
        
        # Grid search with cross-validation
        rf = RandomForestClassifier(random_state=42, n_jobs=-1)
        grid_search = GridSearchCV(
            rf, param_grid, cv=5, scoring='f1', n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X_train_scaled, y_train)
        
        print(f"\nBest parameters found: {grid_search.best_params_}")
        print(f"Best CV F1-Score: {grid_search.best_score_:.4f}")
        
        # Use best model
        best_model = grid_search.best_estimator_
        
        # Evaluate
        y_train_pred = best_model.predict(X_train_scaled)
        y_test_pred = best_model.predict(X_test_scaled)
        
        y_train_proba = best_model.predict_proba(X_train_scaled)[:, 1]
        y_test_proba = best_model.predict_proba(X_test_scaled)[:, 1]
        
        # Metrics
        metrics = {
            'train_accuracy': accuracy_score(y_train, y_train_pred),
            'test_accuracy': accuracy_score(y_test, y_test_pred),
            'train_precision': precision_score(y_train, y_train_pred),
            'test_precision': precision_score(y_test, y_test_pred),
            'train_recall': recall_score(y_train, y_train_pred),
            'test_recall': recall_score(y_test, y_test_pred),
            'train_f1': f1_score(y_train, y_train_pred),
            'test_f1': f1_score(y_test, y_test_pred),
            'train_auc': roc_auc_score(y_train, y_train_proba),
            'test_auc': roc_auc_score(y_test, y_test_proba),
            'cv_f1_mean': grid_search.best_score_
        }
        
        print("\n" + "="*50)
        print("EVALUATION RESULTS")
        print("="*50)
        print(f"\nTRAIN - Accuracy: {metrics['train_accuracy']:.4f}, F1: {metrics['train_f1']:.4f}")
        print(f"TEST  - Accuracy: {metrics['test_accuracy']:.4f}, F1: {metrics['test_f1']:.4f}")
        print(f"CV F1 (5-fold): {metrics['cv_f1_mean']:.4f}")
        
        print("\nConfusion Matrix (Test):")
        cm = confusion_matrix(y_test, y_test_pred)
        print(cm)
        
        print("\nClassification Report (Test):")
        print(classification_report(y_test, y_test_pred, 
                                    target_names=['TURUN', 'NAIK']))
        
        # Log to MLflow
        mlflow.log_params(grid_search.best_params_)
        mlflow.log_metrics(metrics)
        
        # Save model
        import os
        os.makedirs('models', exist_ok=True)
        model_path = 'models/btc_model_tuned.pkl'
        scaler_path = 'models/scaler_tuned.pkl'
        
        joblib.dump(best_model, model_path)
        joblib.dump(scaler, scaler_path)
        
        mlflow.sklearn.log_model(best_model, "model")
        mlflow.log_artifact(scaler_path)
        
        print(f"\nModel saved: {model_path}")
        print(f"Scaler saved: {scaler_path}")
        
        print("\n" + "="*50)
        print("TRAINING COMPLETED!")
        print("="*50)
        
        return best_model, metrics


if __name__ == "__main__":
    model, metrics = train_improved_model()
