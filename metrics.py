"""
Prometheus Metrics Exporter for BTC Prediction API
Author: Sylnzy
Date: 2025-12-18

Exposes custom metrics for Prometheus monitoring
"""

from flask import Flask, Response
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
import time
import psutil
import os

# Define custom metrics
prediction_counter = Counter(
    'btc_predictions_total',
    'Total number of predictions made',
    ['prediction_type']
)

prediction_latency = Histogram(
    'btc_prediction_latency_seconds',
    'Latency of prediction requests',
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0]
)

api_requests = Counter(
    'btc_api_requests_total',
    'Total API requests',
    ['method', 'endpoint', 'status']
)

model_confidence = Gauge(
    'btc_model_confidence',
    'Latest prediction confidence',
    ['prediction_type']
)

btc_price = Gauge(
    'btc_current_price_usd',
    'Current Bitcoin price in USD'
)

cpu_usage = Gauge(
    'system_cpu_usage_percent',
    'System CPU usage percentage'
)

memory_usage = Gauge(
    'system_memory_usage_percent',
    'System memory usage percentage'
)

model_accuracy = Gauge(
    'btc_model_test_accuracy',
    'Model test accuracy score'
)

model_f1_score = Gauge(
    'btc_model_test_f1',
    'Model test F1 score'
)


def setup_metrics_app(main_app):
    """Add metrics endpoint to Flask app"""
    
    @main_app.route('/metrics')
    def metrics():
        """Prometheus metrics endpoint"""
        # Update system metrics
        cpu_usage.set(psutil.cpu_percent(interval=1))
        memory_usage.set(psutil.virtual_memory().percent)
        
        # Set model performance metrics (from training)
        model_accuracy.set(0.513)
        model_f1_score.set(0.540)
        
        return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)
    
    return main_app


def track_prediction(prediction_type, confidence, price):
    """Track prediction metrics"""
    prediction_counter.labels(prediction_type=prediction_type).inc()
    model_confidence.labels(prediction_type=prediction_type).set(confidence)
    btc_price.set(price)


def track_request(method, endpoint, status):
    """Track API request metrics"""
    api_requests.labels(method=method, endpoint=endpoint, status=status).inc()


def measure_latency(func):
    """Decorator to measure function latency"""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        duration = time.time() - start_time
        prediction_latency.observe(duration)
        return result
    return wrapper
