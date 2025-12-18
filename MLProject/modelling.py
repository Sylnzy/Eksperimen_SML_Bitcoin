"""
Bitcoin Price Prediction Model Training Script
Author: Sylnzy
Date: 2025-12-18

This script trains a Random Forest classifier to predict Bitcoin price movements
with MLflow experiment tracking integration.
"""

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from datetime import datetime
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns


def load_preprocessed_data(filepath='btc_preprocessing.csv'):
    """
    Load preprocessed Bitcoin data
    
    Args:
        filepath: Path to preprocessed CSV file
    
    Returns:
        X, y: Features and target variable
    """
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath, index_col=0, parse_dates=True)
    
    # Remove any remaining NaN values
    df = df.dropna()
    
    # Define features and target
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 
                'MA_7', 'MA_30', 'RSI', 'Price_Change', 
                'Volume_Change', 'HL_Spread', 'MACD']
    
    X = df[features]
    y = df['Target']
    
    print(f"Data loaded: {X.shape[0]} samples, {X.shape[1]} features")
    return X, y


def prepare_train_test_split(X, y, test_size=0.2, random_state=42):
    """
    Split data into train and test sets with scaling
    
    Args:
        X: Features
        y: Target variable
        test_size: Proportion of test set
        random_state: Random seed
    
    Returns:
        X_train_scaled, X_test_scaled, y_train, y_test, scaler
    """
    print(f"\nSplitting data: {int((1-test_size)*100)}% train, {int(test_size*100)}% test")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Scale features
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Train set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    print(f"Class distribution - Train: {dict(y_train.value_counts())}")
    print(f"Class distribution - Test: {dict(y_test.value_counts())}")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def train_random_forest(X_train, y_train, params=None):
    """
    Train Random Forest classifier
    
    Args:
        X_train: Training features
        y_train: Training target
        params: Model hyperparameters
    
    Returns:
        Trained model
    """
    if params is None:
        params = {
            'n_estimators': 100,
            'max_depth': 20,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'random_state': 42,
            'n_jobs': -1
        }
    
    print(f"\nTraining Random Forest with parameters:")
    for key, value in params.items():
        print(f"  {key}: {value}")
    
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    
    print("Training completed!")
    return model


def evaluate_model(model, X_train, X_test, y_train, y_test):
    """
    Evaluate model performance
    
    Args:
        model: Trained model
        X_train, X_test: Features
        y_train, y_test: Target variables
    
    Returns:
        Dictionary of metrics
    """
    print("\n" + "="*50)
    print("MODEL EVALUATION")
    print("="*50)
    
    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Prediction probabilities
    y_train_proba = model.predict_proba(X_train)[:, 1]
    y_test_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
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
        'test_auc': roc_auc_score(y_test, y_test_proba)
    }
    
    # Print metrics
    print("\nTRAIN METRICS:")
    print(f"  Accuracy:  {metrics['train_accuracy']:.4f}")
    print(f"  Precision: {metrics['train_precision']:.4f}")
    print(f"  Recall:    {metrics['train_recall']:.4f}")
    print(f"  F1-Score:  {metrics['train_f1']:.4f}")
    print(f"  AUC-ROC:   {metrics['train_auc']:.4f}")
    
    print("\nTEST METRICS:")
    print(f"  Accuracy:  {metrics['test_accuracy']:.4f}")
    print(f"  Precision: {metrics['test_precision']:.4f}")
    print(f"  Recall:    {metrics['test_recall']:.4f}")
    print(f"  F1-Score:  {metrics['test_f1']:.4f}")
    print(f"  AUC-ROC:   {metrics['test_auc']:.4f}")
    
    # Confusion Matrix
    print("\nCONFUSION MATRIX (Test Set):")
    cm = confusion_matrix(y_test, y_test_pred)
    print(cm)
    
    # Classification Report
    print("\nCLASSIFICATION REPORT (Test Set):")
    print(classification_report(y_test, y_test_pred, 
                                target_names=['TURUN (0)', 'NAIK (1)']))
    
    return metrics, cm, y_test_pred, y_test_proba


def plot_confusion_matrix(cm, save_path='confusion_matrix.png'):
    """Plot and save confusion matrix"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['TURUN (0)', 'NAIK (1)'],
                yticklabels=['TURUN (0)', 'NAIK (1)'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved: {save_path}")


def plot_feature_importance(model, feature_names, save_path='feature_importance.png'):
    """Plot and save feature importance"""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(12, 8))
    plt.bar(range(len(importances)), importances[indices], alpha=0.7)
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45)
    plt.title('Feature Importance')
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Feature importance saved: {save_path}")


def plot_roc_curve(y_test, y_test_proba, save_path='roc_curve.png'):
    """Plot and save ROC curve"""
    fpr, tpr, _ = roc_curve(y_test, y_test_proba)
    auc = roc_auc_score(y_test, y_test_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"ROC curve saved: {save_path}")


def save_model_artifacts(model, scaler, save_dir='models'):
    """Save model and scaler"""
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    model_path = f"{save_dir}/btc_model.pkl"
    scaler_path = f"{save_dir}/scaler.pkl"
    
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    
    print(f"\nModel saved: {model_path}")
    print(f"Scaler saved: {scaler_path}")
    
    return model_path, scaler_path


def train_with_mlflow(experiment_name="Bitcoin_Price_Prediction"):
    """
    Main training pipeline with MLflow tracking
    
    Args:
        experiment_name: Name of MLflow experiment
    """
    print("="*50)
    print("BITCOIN PRICE PREDICTION - MODEL TRAINING")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*50 + "\n")
    
    # Set MLflow experiment
    mlflow.set_experiment(experiment_name)
    
    # Start MLflow run
    with mlflow.start_run(run_name=f"RandomForest_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        
        # Load data
        X, y = load_preprocessed_data()
        
        # Split and scale data
        X_train, X_test, y_train, y_test, scaler = prepare_train_test_split(X, y)
        
        # Model parameters
        params = {
            'n_estimators': 100,
            'max_depth': 20,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'random_state': 42,
            'n_jobs': -1
        }
        
        # Log parameters
        mlflow.log_params(params)
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("features", X.shape[1])
        mlflow.log_param("samples", X.shape[0])
        
        # Train model
        model = train_random_forest(X_train, y_train, params)
        
        # Evaluate model
        metrics, cm, y_test_pred, y_test_proba = evaluate_model(
            model, X_train, X_test, y_train, y_test
        )
        
        # Log metrics
        mlflow.log_metrics(metrics)
        
        # Plot and log artifacts
        plot_confusion_matrix(cm)
        mlflow.log_artifact("confusion_matrix.png")
        
        plot_feature_importance(model, X.columns)
        mlflow.log_artifact("feature_importance.png")
        
        plot_roc_curve(y_test, y_test_proba)
        mlflow.log_artifact("roc_curve.png")
        
        # Save model
        model_path, scaler_path = save_model_artifacts(model, scaler)
        
        # Log model
        mlflow.sklearn.log_model(model, "model")
        mlflow.log_artifact(scaler_path)
        
        print("\n" + "="*50)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("="*50)
        print(f"MLflow Run ID: {mlflow.active_run().info.run_id}")
        print(f"Test Accuracy: {metrics['test_accuracy']:.4f}")
        print(f"Test F1-Score: {metrics['test_f1']:.4f}")
        print(f"Test AUC-ROC: {metrics['test_auc']:.4f}")
        
        return model, metrics


if __name__ == "__main__":
    # Train model with MLflow tracking
    model, metrics = train_with_mlflow()
