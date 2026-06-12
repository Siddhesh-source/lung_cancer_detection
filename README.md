# KUB Risk Detector

A production-ready FastAPI-based X-ray analysis service for KUB (Kidney, Ureter, Bladder) risk detection using ONNX models.

## Features

- **FastAPI** web server with async endpoints
- **ONNX Runtime** for efficient model inference
- **CLAHE** preprocessing for lung region enhancement
- **Batch prediction** support (up to 10 images)
- **Docker** containerized deployment
- **Health checks** and model info endpoints

## Quick Start

### 1. Build Docker Image

```bash
cd kub-risk-detector
docker build -t kub-risk-detector .
```

### 2. Run the Service

```bash
# With local model and data directories
docker run -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/data:/app/data \
  kub-risk-detector
```

### 3. Test the API

```bash
# Health check
curl http://localhost:8000/

# Detailed health
curl http://localhost:8000/health

# Model info
curl http://localhost:8000/model_info

# Single prediction
curl -X POST http://localhost:8000/predict \
  -F "file=@path/to/xray.png"

# Batch prediction (up to 10 files)
curl -X POST http://localhost:8000/predict_batch \
  -F "files=@image1.png" \
  -F "files=@image2.png"
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check |
| `/health` | GET | Detailed health status with model info |
| `/model_info` | GET | Model metadata (classes, input size) |
| `/predict` | POST | Single image prediction |
| `/predict_batch` | POST | Batch prediction (max 10 images) |

## Response Format

```json
{
  "filename": "xray.png",
  "predictions": {
    "Atelectasis": 0.12,
    "Cardiomegaly": 0.85,
    "Effusion": 0.23,
    ...
  },
  "top_disease": "Cardiomegaly",
  "confidence": 0.85,
  "processing_time_ms": 45.2
}
```

## Project Structure

```
kub-risk-detector/
├── config.yaml          # All configuration (no hardcoding)
├── Dockerfile           # Multi-stage production Docker build
├── requirements.txt     # Strict version-locked dependencies
├── .dockerignore        # Excludes data, cache, notebooks
├── README.md            # This file
├── src/                 # [THE ENGINE] - Production inference code
│   ├── api.py           # FastAPI application and endpoints
│   ├── inference.py     # ONNX model loading and prediction
│   └── preprocessor.py  # Image preprocessing (CLAHE, normalize)
├── pipeline/            # [THE FACTORY] - Training code
│   ├── train.py         # Model training and ONNX export
│   └── data_prep.py     # Dataset cleaning and splitting
├── models/             # ONNX model files
│   └── kub_model.onnx   # Trained model (add your file here)
└── data/                # Training/test data
    ├── train/normal/
    ├── train/abnormal/
    ├── val/normal/
    └── val/abnormal/
```

## Configuration

All settings are managed in `config.yaml`:

```yaml
model:
  weights_path: "models/kub_model.onnx"
  input_size: [224, 224]
  confidence_threshold: 0.5

preprocessing:
  image_size: 224
  clahe_clip_limit: 2.0

api:
  host: "0.0.0.0"
  port: 8000
```

## Training Your Own Model

### 1. Prepare Data

Organize your X-ray images:
```
data/
├── train/normal/     # Normal X-ray images
├── train/abnormal/   # Abnormal X-ray images
├── val/normal/       # Validation normal
└── val/abnormal/     # Validation abnormal
```

### 2. Train Model

```bash
cd kub-risk-detector
python -m pipeline.train \
  --train_normal data/train/normal \
  --train_abnormal data/train/abnormal \
  --val_normal data/val/normal \
  --val_abnormal data/val/abnormal \
  --output models/kub_model.pth \
  --onnx_output models/kub_model.onnx \
  --epochs 20 \
  --batch_size 16
```

### 3. Deploy

Copy your trained `kub_model.onnx` to the `models/` directory and rebuild the Docker image.

## Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run locally (without Docker)
python -m uvicorn src.api:app --reload --port 8000

# Run tests (if available)
pytest
```

## Requirements

- Python 3.9+
- Docker (for containerized deployment)
- ONNX model file (`kub_model.onnx`)

## License

MIT
