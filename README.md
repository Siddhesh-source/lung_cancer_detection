# Clinical Validation of AI-Based Diagnostic Screening System for Lung Cancer Detection on Chest X-Rays

A production-ready FastAPI-based X-ray analysis service for **lung cancer risk screening** using ChestX-ray14 multi-label classification with ONNX models.

## Features

- **Lung Cancer Risk Assessment** - Focused risk scoring using Mass and Nodule detection
- **Multi-label Classification** - 14 thoracic conditions detected
- **FastAPI** web server with async endpoints
- **ONNX Runtime** for efficient model inference
- **CLAHE** preprocessing for lung region enhancement
- **Grad-CAM Visualization** - Explainable AI heatmaps
- **Batch prediction** support (up to 10 images)
- **Docker** containerized deployment

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
  "prediction": "Low Risk",
  "confidence_score": 0.2578,
  "inference_time_ms": 115.07,
  "visualizations": {
    "gradcam_overlay_base64": "/9j/4AAQSkZJRg...",
    "segmentation_mask_base64": "/9j/4AAQSkZJRg..."
  },
  "additional_inference_metadata": {
    "filename": "xray.png",
    "top_disease": "Pneumothorax",
    "mass_probability": 0.2578,
    "nodule_probability": 0.1141,
    "lung_cancer_risk_score": 0.2578,
    "all_probabilities": {
      "Atelectasis": 0.021,
      "Cardiomegaly": 0.001,
      "Effusion": 0.062,
      "Infiltration": 0.161,
      "Mass": 0.258,
      "Nodule": 0.114,
      "Pneumonia": 0.022,
      "Pneumothorax": 0.814,
      "Consolidation": 0.051,
      "Edema": 0.022,
      "Emphysema": 0.456,
      "Fibrosis": 0.028,
      "Pleural_Thickening": 0.068,
      "Hernia": 0.0001
    },
    "model_version": "1.0.0",
    "preprocessing_applied": ["CLAHE", "resize", "normalize"]
  }
}
```

### Response Fields Explained

| Field | Type | Description |
|-------|------|-------------|
| `prediction` | string | Risk category: "High Risk", "Moderate Risk", or "Low Risk" |
| `confidence_score` | float | Lung cancer risk score (0-1), based on max(Mass, Nodule) |
| `inference_time_ms` | float | Model inference time in milliseconds |
| `visualizations` | object | Base64-encoded visualization images |
| `gradcam_overlay_base64` | string | Grad-CAM heatmap overlaid on X-ray (JPEG format) |
| `segmentation_mask_base64` | string | Segmentation mask with contours (JPEG format) |
| `additional_inference_metadata` | object | Detailed model outputs and metadata |
| `top_disease` | string | Highest probability disease across all 14 classes |
| `mass_probability` | float | Probability of Mass (primary lung cancer indicator) |
| `nodule_probability` | float | Probability of Nodule (primary lung cancer indicator) |
| `lung_cancer_risk_score` | float | max(Mass, Nodule) - used for risk classification |

### Risk Classification Thresholds

- **High Risk**: score ≥ 0.70
- **Moderate Risk**: 0.30 ≤ score < 0.70
- **Low Risk**: score < 0.30

**Note:** The lung cancer risk score is calculated using only Mass and Nodule probabilities (primary indicators for lung cancer). Other conditions (Pneumothorax, Effusion, etc.) are reported in `all_probabilities` but do not influence the risk prediction.

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
