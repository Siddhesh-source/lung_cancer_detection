"""
KUB Risk Detector - PyTorch to ONNX Conversion Script
Converts trained CheXNet DenseNet-121 weights to ONNX format

This script is part of the pipeline (THE FACTORY) for model export.
Run from project root: python pipeline/convert_to_onnx.py
"""

import os
import sys
import torch
import torchvision.models as models

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# Load configuration from root config.yaml
with open("config.yaml", "r") as f:
    _CONFIG = yaml.safe_load(f)

# Paths
PYTORCH_WEIGHTS = os.path.join(PROJECT_ROOT, "chexnet-multilabel-14", "pytorch_model.bin")
ONNX_OUTPUT = os.path.join(PROJECT_ROOT, "kub-risk-detector", "models", "kub_model.onnx")

# Model constants (from config)
NUM_CLASSES = _CONFIG["model_settings"]["num_classes"]
IMAGE_SIZE = _CONFIG["image_settings"]["target_width"]
INPUT_CHANNELS = _CONFIG["image_settings"]["channels"]


class CheXNetDenseNet121(torch.nn.Module):
    """
    CheXNet DenseNet-121 model for 14-class pathology detection.
    Matches the architecture used during training.
    """
    
    def __init__(self, num_classes: int = 14):
        super().__init__()
        
        # Load pretrained DenseNet-121
        self.backbone = models.densenet121(pretrained=False)
        
        # Get the classifier input features
        in_features = self.backbone.classifier.in_features
        
        # Replace classifier for 14-class output
        self.backbone.classifier = torch.nn.Sequential(
            torch.nn.Linear(in_features, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)


def load_trained_weights(model: torch.nn.Module, weights_path: str) -> None:
    """Load trained weights with proper state dict key handling."""
    print(f"Loading trained weights from: {weights_path}")
    
    # Load state dict
    state_dict = torch.load(weights_path, map_location='cpu', weights_only=True)
    
    # Handle state dict format:
    # 1. Remove "backbone." prefix if present
    # 2. Convert "classifier.1" to "classifier" (Sequential vs Linear)
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key
        
        # Remove "backbone." prefix
        if new_key.startswith("backbone."):
            new_key = new_key[9:]  # Remove "backbone." prefix
        
        # Convert "classifier.1." to "classifier." (Sequential to Linear)
        if "classifier.1." in new_key:
            new_key = new_key.replace("classifier.1.", "classifier.")
        
        new_state_dict[new_key] = value
    
    # Load into model
    missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
    
    print(f"  Missing keys: {len(missing_keys)} (expected for aux classifiers)")
    print(f"  Unexpected keys: {len(unexpected_keys)} (ignored)")
    print(f"  Loaded {len(new_state_dict)} parameter tensors")


def export_to_onnx(model: torch.nn.Module, output_path: str) -> None:
    """Export PyTorch model to ONNX format."""
    print(f"\nExporting model to ONNX: {output_path}")
    
    # Create dummy input (batch=1, channels=3, height=224, width=224)
    # Using RGB since DenseNet-121 was pretrained on ImageNet with 3 channels
    dummy_input = torch.randn(1, INPUT_CHANNELS, IMAGE_SIZE, IMAGE_SIZE)
    
    # Ensure model is in eval mode
    model.eval()
    
    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,          # Store trained weights inside the file
        opset_version=12,            # Standard, highly compatible version
        do_constant_folding=True,   # Optimize graph for faster inference
        input_names=['input'],       # Input layer key string
        output_names=['output'],     # Output layer key string
        dynamic_axes={
            'input': {0: 'batch_size'},   # Allow variable batch sizes
            'output': {0: 'batch_size'}
        }
    )
    
    print(f"  ONNX export complete: {output_path}")


def verify_onnx(onnx_path: str, expected_min_size_mb: float = 20) -> bool:
    """Verify ONNX file was created properly with weights included."""
    import os
    
    file_size = os.path.getsize(onnx_path)
    file_size_mb = file_size / (1024 * 1024)
    
    print(f"\nVerification:")
    print(f"  File: {onnx_path}")
    print(f"  Size: {file_size_mb:.2f} MB")
    
    if file_size_mb < expected_min_size_mb:
        print(f"  ERROR: File too small ({file_size_mb:.2f} MB < {expected_min_size_mb} MB)")
        print(f"  This indicates weights were NOT properly saved!")
        return False
    
    # Verify with onnx
    try:
        import onnx
        model = onnx.load(onnx_path)
        onnx.checker.check_model(model)
        print(f"  ONNX structure verified: {len(model.graph.node)} nodes")
        print(f"  Inputs: {[i.name for i in model.graph.input]}")
        print(f"  Outputs: {[o.name for o in model.graph.output]}")
        print(f"  SUCCESS: Model is valid and contains weights!")
        return True
    except Exception as e:
        print(f"  ERROR: ONNX verification failed: {e}")
        return False


def main():
    print("=" * 60)
    print("KUB Risk Detector - PyTorch to ONNX Conversion")
    print("=" * 60)
    
    # Check if source weights exist
    if not os.path.exists(PYTORCH_WEIGHTS):
        print(f"ERROR: Source weights not found at: {PYTORCH_WEIGHTS}")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(os.path.dirname(ONNX_OUTPUT), exist_ok=True)
    
    # 1. Initialize model architecture
    print("\n[1/4] Initializing CheXNet DenseNet-121 architecture...")
    model = CheXNetDenseNet121(num_classes=NUM_CLASSES)
    print(f"  Model created with {NUM_CLASSES} output classes")
    
    # 2. Load trained weights
    print("\n[2/4] Loading trained weights...")
    load_trained_weights(model, PYTORCH_WEIGHTS)
    
    # 3. Export to ONNX
    print("\n[3/4] Exporting to ONNX format...")
    export_to_onnx(model, ONNX_OUTPUT)
    
    # 4. Verify
    print("\n[4/4] Verifying ONNX file...")
    success = verify_onnx(ONNX_OUTPUT)
    
    print("\n" + "=" * 60)
    if success:
        print(f"SUCCESS! Model exported to: {ONNX_OUTPUT}")
    else:
        print("FAILED: ONNX conversion failed verification!")
        sys.exit(1)
    print("=" * 60)


if __name__ == "__main__":
    main()
