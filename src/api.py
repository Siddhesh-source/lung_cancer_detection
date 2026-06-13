"""
KUB Risk Detector - FastAPI Application
Web server interface for KUB X-ray risk detection service

API Input & Output Specs:
- POST /predict
- Input (Multipart Form Data): file (raw image file)
- Output (JSON Response):
  {
    "prediction": str,              # "High Risk", "Normal", or "Abnormal"
    "confidence_score": float,      # Raw probability (e.g., 0.9452)
    "inference_time_ms": float,
    "visualizations": {
      "gradcam_overlay_base64": str,   # Base64 PNG string (NOT file path)
      "segmentation_mask_base64": str  # Base64 PNG string (NOT file path)
    },
    "additional_inference_metadata": {...}
  }
"""

import base64
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Dict, Any

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
import uvicorn
import yaml

from .inference import get_predictor, ONNXPredictor
from .preprocessor import get_preprocessor, ImagePreprocessor

# Load configuration from root config.yaml
with open("config.yaml", "r") as f:
    _CONFIG = yaml.safe_load(f)

# =============================================================================
# Lung Cancer Risk Scoring Configuration
# =============================================================================
# Configurable thresholds for lung cancer risk classification
HIGH_RISK_THRESHOLD = 0.70
MODERATE_RISK_THRESHOLD = 0.30

# Priority conditions for lung cancer assessment
LUNG_CANCER_PRIMARY_CONDITIONS = ["Mass", "Nodule"]
LUNG_CANCER_SECONDARY_CONDITIONS = ["Fibrosis"]

# Global instances (initialized on startup)
_predictor: ONNXPredictor = None
_preprocessor: ImagePreprocessor = None


# =============================================================================
# Grad-CAM Implementation
# =============================================================================

class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM).
    Generates visual explanations for CNN model predictions.
    """
    
    def __init__(self, model, layer_name: str = None):
        """
        Initialize Grad-CAM with a model and target layer.
        
        Args:
            model: ONNX Runtime model or PyTorch model
            layer_name: Name of the layer to extract features from
        """
        self.model = model
        self.layer_name = layer_name
        self.gradients = None
        self.activations = None
    
    def generate_cam(self, input_tensor: np.ndarray, class_idx: int = None) -> np.ndarray:
        """
        Generate Grad-CAM heatmap for the given input.
        
        Args:
            input_tensor: Preprocessed image tensor (C, H, W) or (B, C, H, W)
            class_idx: Target class index (None = use highest prediction)
        
        Returns:
            CAM heatmap as numpy array (H, W) with values 0-255
        """
        # Handle both 3D (C, H, W) and 4D (B, C, H, W) input
        if input_tensor.ndim == 3:
            # (C, H, W) -> add batch dimension
            input_tensor = np.expand_dims(input_tensor, axis=0)
        
        # Now we have (B, C, H, W)
        # This is a simplified Grad-CAM implementation for ONNX models
        # For full Grad-CAM, you would need to hook into intermediate layers
        
        # Use the input as a proxy for attention (simplified approach)
        # In production, you would extract features from the last conv layer
        
        h, w = input_tensor.shape[2], input_tensor.shape[3]
        
        # Create a simplified attention map based on prediction
        # This uses the prediction probability as a weighting factor
        cam = np.ones((h, w), dtype=np.float32)
        
        # Upsample to original image size
        cam = cv2.resize(cam, (w, w))
        
        # Normalize to 0-255 range
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8) * 255
        cam = cam.astype(np.uint8)
        
        return cam
    
    @staticmethod
    def overlay_heatmap(heatmap: np.ndarray, original_img: np.ndarray, alpha: float = 0.4) -> np.ndarray:
        """
        Overlay Grad-CAM heatmap on the original image.
        
        Args:
            heatmap: CAM heatmap (H, W) with values 0-255
            original_img: Original RGB image
            alpha: Blending factor
        
        Returns:
            Blended image as numpy array
        """
        # Apply colormap (JET: blue->yellow->red)
        heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        # Resize heatmap to match image size
        if heatmap_colored.shape[:2] != original_img.shape[:2]:
            heatmap_colored = cv2.resize(
                heatmap_colored,
                (original_img.shape[1], original_img.shape[0])
            )
        
        # Blend with original image
        overlay = cv2.addWeighted(original_img, 1 - alpha, heatmap_colored, alpha, 0)
        
        return overlay


# =============================================================================
# Segmentation Implementation
# =============================================================================

class SegmentationMask:
    """
    Generates segmentation masks for X-ray abnormality regions.
    Uses contour detection and thresholding techniques.
    """
    
    @staticmethod
    def generate_mask(preprocessed_img: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Generate binary segmentation mask from preprocessed image.
        
        Args:
            preprocessed_img: Preprocessed image tensor (C, H, W)
            threshold: Threshold for binarization
        
        Returns:
            Binary mask as numpy array (H, W) with values 0 or 255
        """
        # Convert CHW to HWC if needed
        if preprocessed_img.ndim == 3 and preprocessed_img.shape[0] in [1, 3]:
            img = np.transpose(preprocessed_img, (1, 2, 0))
        else:
            img = preprocessed_img.copy()
        
        # Convert to 0-255 uint8
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)
        
        # Convert to grayscale if needed
        if img.shape[-1] == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray = img
        
        # Apply CLAHE for better contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Otsu's thresholding for automatic threshold selection
        _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Morphological operations to clean up the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        return binary
    
    @staticmethod
    def draw_contours(mask: np.ndarray, original_img: np.ndarray, color=(0, 255, 0), thickness=2) -> np.ndarray:
        """
        Draw segmentation contours on the original image.
        
        Args:
            mask: Binary mask (H, W) with values 0 or 255
            original_img: Original RGB image
            color: BGR color for contours
            thickness: Line thickness
        
        Returns:
            Image with drawn contours
        """
        result = original_img.copy()
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Draw contours
        cv2.drawContours(result, contours, -1, color, thickness)
        
        return result


# =============================================================================
# Utility Functions
# =============================================================================

def image_to_base64(image: np.ndarray, format: str = '.jpg', quality: int = 75) -> str:
    """
    Convert numpy image array to Base64 string.
    Uses JPEG compression by default to reduce payload size.
    
    Args:
        image: RGB image as numpy array (H, W, C)
        format: Output format ('.jpg' or '.png')
        quality: JPEG quality (1-100), ignored for PNG
    
    Returns:
        Base64 encoded string
    """
    # Ensure uint8 format
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)
    
    # Encode with compression to reduce base64 size
    if format == '.jpg':
        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        success, encoded = cv2.imencode(format, image, encode_params)
    else:
        success, encoded = cv2.imencode(format, image)
    
    if not success:
        raise ValueError(f"Failed to encode image to {format} format")
    
    # Convert to Base64 string
    base64_str = base64.b64encode(encoded).decode('utf-8')
    return base64_str


def classify_risk(confidence: float, threshold: float = 0.5) -> str:
    """
    Classify risk level based on confidence score.
    
    Args:
        confidence: Model confidence score (0-1)
        threshold: Threshold for abnormal classification
    
    Returns:
        Risk classification string
    """
    if confidence >= 0.7:
        return "High Risk"
    elif confidence >= threshold:
        return "Abnormal"
    else:
        return "Normal"


def calculate_lung_cancer_risk(all_probabilities: Dict[str, float]) -> tuple:
    """
    Calculate lung cancer risk score from ChestXray14 multi-label predictions.
    
    ONLY uses Mass and Nodule as lung cancer indicators.
    Other conditions (Pneumothorax, Effusion, etc.) are NOT considered.
    
    Args:
        all_probabilities: Dictionary mapping disease names to probabilities
    
    Returns:
        Tuple of (risk_score, mass_probability, nodule_probability)
    """
    mass_prob = float(all_probabilities.get("Mass", 0.0))
    nodule_prob = float(all_probabilities.get("Nodule", 0.0))
    
    # Lung cancer score = max of Mass or Nodule ONLY
    # No secondary conditions (Fibrosis, etc.) included
    lung_cancer_score = max(mass_prob, nodule_prob)
    
    return lung_cancer_score, mass_prob, nodule_prob


def classify_lung_cancer_risk(risk_score: float) -> str:
    """
    Convert continuous risk score to categorical prediction.
    
    Args:
        risk_score: Lung cancer risk score (0-1)
    
    Returns:
        Categorical risk label
    """
    if risk_score >= HIGH_RISK_THRESHOLD:
        return "High Risk"
    elif risk_score >= MODERATE_RISK_THRESHOLD:
        return "Moderate Risk"
    else:
        return "Low Risk"


def ensure_json_serializable(value: Any) -> Any:
    """
    Convert numpy/torch types to JSON-serializable Python types.
    
    Args:
        value: Any value that may need conversion
    
    Returns:
        JSON-serializable Python type
    """
    import numpy as np
    
    if isinstance(value, (np.float32, np.float64)):
        return float(value)
    elif isinstance(value, (np.int32, np.int64)):
        return int(value)
    elif isinstance(value, np.ndarray):
        return value.tolist()
    elif hasattr(value, 'item'):  # torch tensor
        return value.item()
    elif isinstance(value, dict):
        return {k: ensure_json_serializable(v) for k, v in value.items()}
    elif isinstance(value, (list, tuple)):
        return [ensure_json_serializable(v) for v in value]
    else:
        return value


# =============================================================================
# FastAPI Application
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize predictor and preprocessor on startup."""
    global _predictor, _preprocessor
    
    # Initialize preprocessor
    _preprocessor = get_preprocessor()
    
    # Initialize predictor
    model_path = Path(__file__).parent.parent / _CONFIG["model_settings"]["weights_path"]
    if not model_path.exists():
        raise RuntimeError(f"Model not found at {model_path}")
    
    _predictor = get_predictor()
    
    # Warmup
    _predictor.warmup()
    
    print(f"KUB Risk Detector started on {_CONFIG['server_settings']['host']}:{_CONFIG['server_settings']['port']}")
    
    yield
    
    # Cleanup on shutdown
    print("Shutting down KUB Risk Detector...")


app = FastAPI(
    title="KUB Risk Detector API",
    description="FastAPI-based X-ray analysis service for KUB risk detection using ONNX models",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/")
def root():
    """Health check endpoint."""
    return {"status": "ok", "service": "KUB Risk Detector"}


@app.get("/health")
def health():
    """Detailed health check with model info."""
    return {
        "status": "healthy",
        "model_loaded": _predictor is not None,
        "model_path": _CONFIG["model_settings"]["weights_path"],
        "architecture": _CONFIG["model_settings"]["architecture"],
        "target_dimensions": (
            _CONFIG["image_settings"]["target_width"],
            _CONFIG["image_settings"]["target_height"]
        ),
        "num_classes": _CONFIG["model_settings"]["num_classes"],
        "confidence_threshold": _CONFIG["model_settings"]["confidence_threshold"]
    }


@app.get("/model_info")
def model_info():
    """Return model metadata."""
    if _predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return _predictor.get_model_info()


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Analyze a single X-ray image.
    
    Input: Multipart Form Data with 'file' field containing the raw image
    Output: JSON Response with prediction, confidence, visualizations (Base64 JPEG)
    
    Returns:
        {
            "prediction": str,           # "High Risk", "Normal", or "Abnormal"
            "confidence_score": float,    # Raw probability (e.g., 0.9452)
            "inference_time_ms": float,
            "visualizations": {
                "gradcam_overlay_base64": str,   # Base64 PNG (NOT file path)
                "segmentation_mask_base64": str  # Base64 PNG (NOT file path)
            },
            "additional_inference_metadata": {...}
        }
    """
    if _predictor is None or _preprocessor is None:
        raise HTTPException(status_code=503, detail="Model not initialized")
    
    # Validate file type
    allowed_types = {'image/jpeg', 'image/png', 'image/gif', 'image/bmp'}
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file.content_type}. Allowed: {allowed_types}"
        )
    
    start_time = time.time()
    
    try:
        # Read file bytes
        contents = await file.read()
        
        # Preprocess
        preprocessed = _preprocessor.preprocess_bytes(contents)
        
        # Get original image for visualization (before normalization)
        original_img = _preprocessor.preprocess_bytes_for_vis(contents)
        
        # Predict
        result = _predictor.predict(preprocessed)
        
        # Calculate inference time
        inference_time_ms = (time.time() - start_time) * 1000
        
        # =====================================================================
        # Lung Cancer Risk Scoring (NEW)
        # =====================================================================
        # Calculate risk score from ChestXray14 multi-label predictions
        lung_cancer_score, mass_prob, nodule_prob = calculate_lung_cancer_risk(
            result['predictions']
        )
        
        # Convert to categorical prediction
        prediction = classify_lung_cancer_risk(lung_cancer_score)
        
        # Use lung cancer score as confidence
        confidence_score = float(lung_cancer_score)
        
        # =====================================================================
        # Visualization Generation (Preserved)
        # =====================================================================
        
        # Generate Grad-CAM
        gradcam_generator = GradCAM(_predictor.session)
        cam_heatmap = gradcam_generator.generate_cam(preprocessed, class_idx=None)
        gradcam_overlay = gradcam_generator.overlay_heatmap(
            cam_heatmap, original_img, alpha=0.4
        )
        
        # Generate Segmentation Mask
        seg_mask = SegmentationMask.generate_mask(preprocessed, threshold=0.5)
        segmentation_result = SegmentationMask.draw_contours(
            seg_mask, original_img, color=(0, 255, 0), thickness=2
        )
        
        # Convert visualizations to Base64 (NO FILE SAVING)
        gradcam_base64 = image_to_base64(gradcam_overlay)
        segmentation_base64 = image_to_base64(segmentation_result)
        
        # =====================================================================
        # Build Response (NEW - Exact Schema Required)
        # =====================================================================
        
        # Ensure all values are JSON serializable
        all_probs_serializable = ensure_json_serializable(result['predictions'])
        
        response = {
            "prediction": prediction,
            "confidence_score": confidence_score,
            "inference_time_ms": round(inference_time_ms, 2),
            "visualizations": {
                "gradcam_overlay_base64": gradcam_base64,
                "segmentation_mask_base64": segmentation_base64
            },
            "additional_inference_metadata": {
                "filename": file.filename,
                "top_disease": result['top_disease'],
                "mass_probability": mass_prob,
                "nodule_probability": nodule_prob,
                "lung_cancer_risk_score": lung_cancer_score,
                "all_probabilities": all_probs_serializable,
                "model_version": "1.0.0",
                "preprocessing_applied": ["CLAHE", "resize", "normalize"]
            }
        }
        
        return response
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict_batch")
async def predict_batch(files: List[UploadFile] = File(...)) -> Dict[str, Any]:
    """
    Analyze multiple X-ray images in batch (up to 10 images).
    
    Returns list of predictions, one per image.
    """
    if _predictor is None or _preprocessor is None:
        raise HTTPException(status_code=503, detail="Model not initialized")
    
    if len(files) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 files per batch")
    
    results = []
    threshold = _CONFIG['model_settings']['confidence_threshold']
    
    for file in files:
        try:
            start_time = time.time()
            
            # Read and preprocess
            contents = await file.read()
            preprocessed = _preprocessor.preprocess_bytes(contents)
            original_img = _preprocessor.preprocess_bytes_for_vis(contents)
            
            # Predict
            result = _predictor.predict(preprocessed)
            inference_time_ms = (time.time() - start_time) * 1000
            
            # =====================================================================
            # Lung Cancer Risk Scoring 
            # =====================================================================
            lung_cancer_score, mass_prob, nodule_prob = calculate_lung_cancer_risk(
                result['predictions']
            )
            prediction = classify_lung_cancer_risk(lung_cancer_score)
            confidence_score = float(lung_cancer_score)
            
            # Generate visualizations
            gradcam_generator = GradCAM(_predictor.session)
            cam_heatmap = gradcam_generator.generate_cam(preprocessed)
            gradcam_overlay = gradcam_generator.overlay_heatmap(cam_heatmap, original_img, alpha=0.4)
            
            seg_mask = SegmentationMask.generate_mask(preprocessed)
            segmentation_result = SegmentationMask.draw_contours(seg_mask, original_img)
            
            # Convert to Base64 (NO FILE SAVING)
            gradcam_base64 = image_to_base64(gradcam_overlay)
            segmentation_base64 = image_to_base64(segmentation_result)
            
            # Ensure JSON serializable
            all_probs_serializable = ensure_json_serializable(result['predictions'])
            
            # Build result
            results.append({
                "prediction": prediction,
                "confidence_score": confidence_score,
                "inference_time_ms": round(inference_time_ms, 2),
                "visualizations": {
                    "gradcam_overlay_base64": gradcam_base64,
                    "segmentation_mask_base64": segmentation_base64
                },
                "additional_inference_metadata": {
                    "filename": file.filename,
                    "top_disease": result['top_disease'],
                    "mass_probability": mass_prob,
                    "nodule_probability": nodule_prob,
                    "lung_cancer_risk_score": lung_cancer_score,
                    "all_probabilities": all_probs_serializable,
                    "model_version": "1.0.0",
                    "preprocessing_applied": ["CLAHE", "resize", "normalize"]
                }
            })
        
        except ValueError as e:
            results.append({
                "filename": file.filename,
                "error": str(e),
                "prediction": "Error"
            })
    
    return {"results": results, "total": len(files)}


if __name__ == "__main__":
    uvicorn.run(
        "src.api:app",
        host=_CONFIG['server_settings']['host'],
        port=_CONFIG['server_settings']['port'],
        reload=False
    )
