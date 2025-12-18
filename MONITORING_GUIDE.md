# Monitoring Setup Guide

## Overview
This setup includes Prometheus for metrics collection and Grafana for visualization.

## Components

### 1. Prometheus
- **Port**: 9090
- **Purpose**: Scrapes metrics from API every 10 seconds
- **Config**: `monitoring/prometheus.yml`

### 2. Grafana
- **Port**: 3000
- **Username**: admin
- **Password**: admin123
- **Purpose**: Visualize metrics with dashboards

### 3. Custom Metrics
The API exposes the following metrics at `/metrics` endpoint:

#### Prediction Metrics
- `btc_predictions_total` - Total predictions made (NAIK/TURUN)
- `btc_prediction_latency_seconds` - Response time histogram
- `btc_model_confidence` - Latest prediction confidence
- `btc_current_price_usd` - Current Bitcoin price

#### System Metrics
- `system_cpu_usage_percent` - CPU usage
- `system_memory_usage_percent` - Memory usage

#### Model Performance
- `btc_model_test_accuracy` - Model accuracy score
- `btc_model_test_f1` - Model F1 score

#### API Metrics
- `btc_api_requests_total` - Total API requests by endpoint

## Setup Instructions

### 1. Start Monitoring Stack

```bash
# Start all services (API, Prometheus, Grafana)
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f grafana
docker-compose logs -f prometheus
```

### 2. Access Dashboards

**Prometheus UI**: http://localhost:9090
- View metrics and run PromQL queries
- Check targets: http://localhost:9090/targets

**Grafana UI**: http://localhost:3000
- Login: admin / admin123
- Prometheus datasource auto-configured

### 3. Create Grafana Dashboard

1. Login to Grafana (http://localhost:3000)
2. Click **+** → **Dashboard** → **Add new panel**
3. Use PromQL queries:

```promql
# Total Predictions
sum(btc_predictions_total)

# Prediction Rate (per minute)
rate(btc_predictions_total[1m])

# Average Latency
histogram_quantile(0.95, rate(btc_prediction_latency_seconds_bucket[5m]))

# Current BTC Price
btc_current_price_usd

# CPU Usage
system_cpu_usage_percent

# Memory Usage  
system_memory_usage_percent

# Model Accuracy
btc_model_test_accuracy

# API Request Rate
rate(btc_api_requests_total[1m])
```

### 4. Sample Dashboard Panels

#### Panel 1: Total Predictions (Stat)
- Query: `sum(btc_predictions_total)`
- Visualization: Stat
- Title: "Total Predictions"

#### Panel 2: Prediction Distribution (Pie Chart)
- Query: `btc_predictions_total`
- Visualization: Pie chart
- Legend: {{prediction_type}}

#### Panel 3: Prediction Latency (Graph)
- Query: `histogram_quantile(0.95, rate(btc_prediction_latency_seconds_bucket[5m]))`
- Visualization: Time series
- Title: "P95 Latency"

#### Panel 4: Bitcoin Price (Graph)
- Query: `btc_current_price_usd`
- Visualization: Time series
- Title: "BTC Price (USD)"

#### Panel 5: System Resources (Graph)
- Query 1: `system_cpu_usage_percent`
- Query 2: `system_memory_usage_percent`
- Visualization: Time series
- Title: "System Resources"

### 5. Alerting Rules (Optional)

Create alert in Prometheus (`monitoring/prometheus.yml`):

```yaml
rule_files:
  - 'alerts.yml'

alerting:
  alertmanagers:
    - static_configs:
        - targets: ['alertmanager:9093']
```

Create `monitoring/alerts.yml`:

```yaml
groups:
  - name: btc_predictions
    interval: 30s
    rules:
      - alert: HighLatency
        expr: histogram_quantile(0.95, rate(btc_prediction_latency_seconds_bucket[5m])) > 2
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "High prediction latency"
          description: "P95 latency is {{ $value }}s"
      
      - alert: HighCPU
        expr: system_cpu_usage_percent > 80
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High CPU usage"
          description: "CPU usage is {{ $value }}%"
      
      - alert: HighMemory
        expr: system_memory_usage_percent > 90
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "High memory usage"
          description: "Memory usage is {{ $value }}%"
```

## Testing Metrics

```bash
# Generate some predictions
for i in {1..10}; do
  curl -X POST http://localhost:8000/predict
  sleep 1
done

# View metrics
curl http://localhost:8000/metrics

# Query Prometheus
curl 'http://localhost:9090/api/v1/query?query=btc_predictions_total'
```

## Troubleshooting

### Metrics not showing in Prometheus
1. Check targets: http://localhost:9090/targets
2. Verify API is running: `docker ps`
3. Check Prometheus logs: `docker-compose logs prometheus`

### Grafana not connecting to Prometheus
1. Verify datasource config
2. Test connection in Grafana → Configuration → Data Sources
3. Ensure containers are on same network: `docker network inspect submission_btc-network`

## Clean Up

```bash
# Stop services
docker-compose down

# Remove volumes (warning: deletes all data)
docker-compose down -v
```

## Production Recommendations

1. **Security**:
   - Change default Grafana password
   - Enable authentication in Prometheus
   - Use SSL/TLS certificates

2. **Persistence**:
   - Configure volume backups
   - Use external storage for Prometheus data

3. **Scaling**:
   - Increase Prometheus retention period
   - Configure remote storage (e.g., Thanos, Cortex)
   - Add multiple API replicas

4. **Alerting**:
   - Setup AlertManager
   - Configure notification channels (email, Slack, PagerDuty)
   - Define SLOs and SLIs
