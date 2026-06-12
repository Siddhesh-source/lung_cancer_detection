"""
KUB Risk Detector - Image Preprocessor
On-the-fly image resizing, CLAHE enhancement, and normalization (1 image at a time)
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Union
import yaml

# Load configuration from root config.yaml
with open("config.yaml", "r") as f:
    _CONFIG = yaml.safe_load(f)


class ImagePreprocessor:
    """Preprocesses chest X-ray images for inference pipeline."""
    
    def __init__(
        self,
        target_width: int = None,
        target_height: int = None,
        channels: int = None,
        mean: list = None,
        std: list = None,
        clahe_clip_limit: float = None,
        clahe_tile_size: int = None
    ):
        # Dynamically extract parameters from config
        self.target_width = target_width or _CONFIG["image_settings"]["target_width"]
        self.target_height = target_height or _CONFIG["image_settings"]["target_height"]
        self.channels = channels or _CONFIG["image_settings"]["channels"]
        self.mean = np.array(mean or _CONFIG["image_settings"]["mean"])
        self.std = np.array(std or _CONFIG["image_settings"]["std"])
        self.clahe_clip_limit = clahe_clip_limit or _CONFIG["image_settings"]["clahe_clip_limit"]
        self.clahe_tile_size = clahe_tile_size or _CONFIG["image_settings"]["clahe_tile_size"]
        
        # Initialize CLAHE
        self.clahe = cv2.createCLAHE(
            clipLimit=self.clahe_clip_limit,
            tileGridSize=(self.clahe_tile_size, self.clahe_tile_size)
        )
    
    def load_image(self, image_path: Union[str, Path]) -> np.ndarray:
        """Load image from file path."""
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Failed to load image: {image_path}")
        return img
    
    def apply_clahe(self, image: np.ndarray) -> np.ndarray:
        """Apply CLAHE for lung region enhancement."""
        return self.clahe.apply(image)
    
    def resize(self, image: np.ndarray) -> np.ndarray:
        """Resize image to target dimensions using INTER_AREA interpolation."""
        return cv2.resize(
            image,
            (self.target_width, self.target_height),
            interpolation=cv2.INTER_AREA
        )
    
    def normalize(self, image: np.ndarray) -> np.ndarray:
        """Normalize image to [0, 1] then apply z-score normalization."""
        img_float = image.astype(np.float32) / 255.0
        return (img_float - self.mean) / self.std
    
    def to_rgb(self, image: np.ndarray) -> np.ndarray:
        """Convert grayscale to 3-channel RGB."""
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    
    def preprocess(self, image_path: Union[str, Path]) -> np.ndarray:
        """
        Full preprocessing pipeline:
        load -> CLAHE -> RGB convert -> resize -> normalize
        
        Returns:
            np.ndarray: Preprocessed image with shape (C, H, W) = (3, 224, 224)
        """
        # 1. Load
        img = self.load_image(image_path)
        
        # 2. CLAHE enhancement
        img = self.apply_clahe(img)
        
        # 3. Convert to RGB (3 channels for pretrained models)
        img = self.to_rgb(img)
        
        # 4. Resize
        img = self.resize(img)
        
        # 5. Normalize
        img = self.normalize(img)
        
        # 6. Convert to CHW format for ONNX
        img = img.transpose(2, 0, 1)  # HWC -> CHW
        
        return img.astype(np.float32)
    
    def preprocess_bytes(self, image_bytes: bytes) -> np.ndarray:
        """Preprocess image from raw bytes (for API uploads)."""
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError("Failed to decode image from bytes")
        
        img = self.apply_clahe(img)
        img = self.to_rgb(img)
        img = self.resize(img)
        img = self.normalize(img)
        img = img.transpose(2, 0, 1)
        
        return img.astype(np.float32)
    
    def preprocess_bytes_for_vis(self, image_bytes: bytes) -> np.ndarray:
        """
        Preprocess image from raw bytes for visualization purposes.
        Returns RGB image (resized, with CLAHE) without normalization.
        Used for overlaying Grad-CAM and segmentation visualizations.
        
        Returns:
            np.ndarray: RGB image with shape (H, W, 3) and values 0-255 (uint8)
        """
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError("Failed to decode image from bytes")
        
        # Apply CLAHE for enhancement
        img = self.apply_clahe(img)
        
        # Convert to RGB (3 channels)
        img = self.to_rgb(img)
        
        # Resize to target dimensions
        img = self.resize(img)
        
        return img


def get_preprocessor() -> ImagePreprocessor:
    """Factory function to get preprocessor instance with config values."""
    return ImagePreprocessor()
