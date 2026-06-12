"""
KUB Risk Detector - Inference Engine
Model loading and mathematical execution using ONNX Runtime
"""

import time
from pathlib import Path
from typing import Dict, List
import numpy as np
import onnxruntime as ort
import yaml

# Load configuration from root config.yaml
with open("config.yaml", "r") as f:
    _CONFIG = yaml.safe_load(f)


class ONNXPredictor:
    """ONNX model inference engine for KUB risk detection."""
    
    def __init__(self, model_path: str = None, providers: List[str] = None):
        # Dynamically extract parameters from config
        self.model_path = model_path or _CONFIG["model_settings"]["weights_path"]
        self.num_classes = _CONFIG["model_settings"]["num_classes"]
        self.threshold = _CONFIG["model_settings"]["confidence_threshold"]
        self.class_names = _CONFIG["class_names"]
        self.target_dimensions = (
            _CONFIG["image_settings"]["target_width"],
            _CONFIG["image_settings"]["target_height"]
        )
        
        # Default providers (CPU only in Docker container)
        if providers is None:
            providers = ['CPUExecutionProvider']
        
        self._session = None
        self._providers = providers
        self._warmup_done = False
    
    @property
    def session(self) -> ort.InferenceSession:
        """Lazy load ONNX session on first access."""
        if self._session is None:
            self._session = self._create_session()
        return self._session
    
    def _create_session(self) -> ort.InferenceSession:
        """Create ONNX Runtime session with providers."""
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        session = ort.InferenceSession(
            str(self.model_path),
            sess_options=sess_options,
            providers=self._providers
        )
        
        # Verify input/output names
        self._input_name = session.get_inputs()[0].name
        self._output_name = session.get_outputs()[0].name
        
        return session
    
    def warmup(self, iterations: int = 3) -> None:
        """Run dummy inference to warm up the model."""
        if self._warmup_done:
            return
        
        # Create dummy input matching target dimensions
        dummy_input = np.random.randn(
            1, 3, self.target_dimensions[0], self.target_dimensions[1]
        ).astype(np.float32)
        
        for _ in range(iterations):
            self.session.run([self._output_name], {self._input_name: dummy_input})
        
        self._warmup_done = True
    
    def predict(self, preprocessed_image: np.ndarray) -> Dict:
        """
        Run inference on a single preprocessed image.
        
        Returns:
            Dict with predictions, top disease, confidence, and processing time
        """
        start_time = time.time()

        # Ensure batch dimension
        if preprocessed_image.ndim == 3:
            preprocessed_image = np.expand_dims(preprocessed_image, axis=0)

        # Run inference
        outputs = self.session.run(
            [self._output_name],
            {self._input_name: preprocessed_image}
        )

        if len(outputs) == 0:
            raise ValueError("Model returned empty outputs")

        output_data = outputs[0]
        if isinstance(output_data, tuple):
            output_data = output_data[0]

        # Apply sigmoid (multi-label classification)
        probs = self._sigmoid(output_data)[0]
        
        # Build predictions dict
        predictions = {
            self.class_names[i]: float(probs[i])
            for i in range(len(self.class_names))
        }
        
        # Get top prediction
        top_idx = int(np.argmax(probs))
        
        processing_time = (time.time() - start_time) * 1000
        
        return {
            "predictions": predictions,
            "top_disease": self.class_names[top_idx],
            "confidence": float(probs[top_idx]),
            "processing_time_ms": round(processing_time, 2)
        }
    
    def predict_batch(self, images: List[np.ndarray]) -> List[Dict]:
        """Run inference on a batch of preprocessed images."""
        if not images:
            return []
        
        batch = np.stack(images, axis=0)
        
        start_time = time.time()
        
        outputs = self.session.run(
            [self._output_name],
            {self._input_name: batch}
        )
        
        probs = self._sigmoid(outputs[0])
        
        results = []
        for i in range(len(images)):
            top_idx = int(np.argmax(probs[i]))
            results.append({
                "predictions": {
                    self.class_names[j]: float(probs[i][j])
                    for j in range(len(self.class_names))
                },
                "top_disease": self.class_names[top_idx],
                "confidence": float(probs[i][top_idx])
            })
        
        total_time = (time.time() - start_time) * 1000
        for r in results:
            r["processing_time_ms"] = round(total_time / len(images), 2)
        
        return results
    
    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        """Numerically stable sigmoid."""
        return np.where(
            x >= 0,
            1 / (1 + np.exp(-x)),
            np.exp(x) / (1 + np.exp(x))
        )
    
    def get_model_info(self) -> Dict:
        """Return model metadata."""
        return {
            "model_path": str(self.model_path),
            "architecture": _CONFIG["model_settings"]["architecture"],
            "num_classes": self.num_classes,
            "target_dimensions": self.target_dimensions,
            "class_names": self.class_names,
            "confidence_threshold": self.threshold,
            "providers": self.session.get_providers()
        }


def get_predictor() -> ONNXPredictor:
    """Factory function to get predictor instance."""
    return ONNXPredictor()
