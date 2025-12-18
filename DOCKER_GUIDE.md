# Docker Deployment Guide

## Prerequisites
- Docker installed
- Docker Hub account
- Models trained (models/btc_model_tuned.pkl and scaler_tuned.pkl)

## Build Docker Image

```bash
# Build image
docker build -t btc-prediction-api:latest .

# Tag for Docker Hub
docker tag btc-prediction-api:latest YOUR_DOCKERHUB_USERNAME/btc-prediction-api:latest

# Push to Docker Hub
docker login
docker push YOUR_DOCKERHUB_USERNAME/btc-prediction-api:latest
```

## Run Container

### Option 1: Run single container
```bash
docker run -d \
  --name btc-prediction-api \
  -p 8000:8000 \
  -v $(pwd)/models:/app/models:ro \
  btc-prediction-api:latest
```

### Option 2: Run with docker-compose (recommended)
```bash
# Start all services (API + Prometheus + Grafana)
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

## Test API

```bash
# Health check
curl http://localhost:8000/health

# Get prediction
curl -X POST http://localhost:8000/predict

# View metrics
curl http://localhost:8000/metrics
```

## Access Services

- **API**: http://localhost:8000
- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000
  - Username: admin
  - Password: admin123

## Update and Redeploy

```bash
# Rebuild image
docker-compose build

# Restart services
docker-compose restart
```

## Clean Up

```bash
# Remove containers
docker-compose down

# Remove volumes
docker-compose down -v

# Remove images
docker rmi btc-prediction-api:latest
```
