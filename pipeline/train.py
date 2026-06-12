"""
KUB Risk Detector - Training Script
Main training execution script for KUB X-ray classification model
"""

import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import yaml
import onnx

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load configuration from root config.yaml
with open("config.yaml", "r") as f:
    _CONFIG = yaml.safe_load(f)


class KUBDataset(Dataset):
    """Dataset class for KUB X-ray images."""
    
    def __init__(
        self,
        normal_dir: str,
        abnormal_dir: str,
        image_size: int = 224,
        augment: bool = False
    ):
        self.image_size = image_size
        self.augment = augment
        
        # Collect image paths
        self.images = []
        self.labels = []
        
        normal_path = Path(normal_dir)
        abnormal_path = Path(abnormal_dir)
        
        for img_path in normal_path.glob("*"):
            if img_path.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'}:
                self.images.append(img_path)
                self.labels.append(0)  # Normal = 0
        
        for img_path in abnormal_path.glob("*"):
            if img_path.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'}:
                self.images.append(img_path)
                self.labels.append(1)  # Abnormal = 1
        
        # Transforms
        if augment:
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
    
    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path = self.images[idx]
        label = self.labels[idx]
        
        try:
            img = Image.open(img_path).convert('RGB')
            img = self.transform(img)
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            img = torch.zeros(3, self.image_size, self.image_size)
        
        return img, label


class KUBClassifier(nn.Module):
    """Simple CNN classifier for KUB X-ray binary classification."""
    
    def __init__(self, num_classes: int = 1):
        super().__init__()
        
        # Use a lightweight CNN architecture
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 112x112
            
            # Block 2
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 56x56
            
            # Block 3
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 28x28
            
            # Block 4
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 14x14
            
            # Block 5
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)  # 1x1
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class Trainer:
    """Training manager for KUB classifier."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str = None,
        learning_rate: float = 0.001
    ):
        config = _load_config()
        
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3, verbose=True
        )
        
        self.best_val_loss = float('inf')
        self.best_model_state = None
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (images, labels) in enumerate(self.train_loader):
            images = images.to(self.device)
            labels = labels.float().to(self.device)
            
            self.optimizer.zero_grad()
            
            outputs = self.model(images).squeeze()
            loss = self.criterion(outputs, labels)
            
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            
            # Calculate accuracy
            preds = (torch.sigmoid(outputs) > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = correct / total
        
        return {'loss': epoch_loss, 'accuracy': epoch_acc}
    
    def validate(self) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in self.val_loader:
                images = images.to(self.device)
                labels = labels.float().to(self.device)
                
                outputs = self.model(images).squeeze()
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                
                preds = (torch.sigmoid(outputs) > 0.5).float()
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        
        val_loss = running_loss / len(self.val_loader)
        val_acc = correct / total
        
        return {'loss': val_loss, 'accuracy': val_acc}
    
    def train(self, num_epochs: int = 20, save_path: str = None) -> Dict:
        """Full training loop."""
        print(f"Training on {self.device}")
        print(f"Epochs: {num_epochs}")
        print("-" * 50)
        
        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        
        for epoch in range(num_epochs):
            epoch_start = time.time()
            
            # Train
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate()
            
            epoch_time = time.time() - epoch_start
            
            # Update scheduler
            self.scheduler.step(val_metrics['loss'])
            
            # Save best model
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.best_model_state = self.model.state_dict().copy()
                best_marker = " *BEST*"
            else:
                best_marker = ""
            
            # Log
            print(
                f"Epoch {epoch+1}/{num_epochs} "
                f"({epoch_time:.1f}s) - "
                f"Train Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f} - "
                f"Val Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}"
                f"{best_marker}"
            )
            
            # Record history
            history['train_loss'].append(train_metrics['loss'])
            history['train_acc'].append(train_metrics['accuracy'])
            history['val_loss'].append(val_metrics['loss'])
            history['val_acc'].append(val_metrics['accuracy'])
        
        # Load best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
        
        # Save model if path provided
        if save_path:
            torch.save(self.best_model_state, save_path)
            print(f"\nBest model saved to {save_path}")
        
        return history


def export_to_onnx(model: nn.Module, output_path: str, input_size: Tuple = (1, 3, 224, 224)):
    """Export PyTorch model to ONNX format."""
    model.eval()
    
    dummy_input = torch.randn(*input_size)
    
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    print(f"Model exported to ONNX: {output_path}")
    
    # Verify the ONNX model
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX model verified successfully")


def train_model(
    train_normal: str,
    train_abnormal: str,
    val_normal: str,
    val_abnormal: str,
    output_model: str = "models/kub_model.pth",
    output_onnx: str = "models/kub_model.onnx",
    num_epochs: int = 20,
    batch_size: int = 16,
    learning_rate: float = 0.001
) -> Dict:
    """
    Full training pipeline.
    
    Returns:
        Training history dict
    """
    config = _load_config()
    image_size = _CONFIG['image_settings']['target_width']
    
    print("KUB Risk Detector - Training Pipeline")
    print("=" * 50)
    
    # Create datasets
    train_dataset = KUBDataset(
        normal_dir=train_normal,
        abnormal_dir=train_abnormal,
        image_size=image_size,
        augment=True
    )
    
    val_dataset = KUBDataset(
        normal_dir=val_normal,
        abnormal_dir=val_abnormal,
        image_size=image_size,
        augment=False
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    # Create model
    model = KUBClassifier(num_classes=1)
    
    # Train
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=learning_rate
    )
    
    history = trainer.train(num_epochs=num_epochs, save_path=output_model)
    
    # Export to ONNX
    if output_onnx:
        export_to_onnx(model, output_onnx, input_size=(1, 3, image_size, image_size))
    
    return history


if __name__ == "__main__":
    print("KUB Risk Detector - Training Script")
    print("=" * 50)
    
    # Default paths from config
    normal_dir = _CONFIG['data_paths']['normal']
    abnormal_dir = _CONFIG['data_paths']['abnormal']
    
    print(f"Normal images: {normal_dir}")
    print(f"Abnormal images: {abnormal_dir}")
    print()
    print("To run training programmatically:")
    print("  from pipeline.train import train_model")
    print("  history = train_model(...)")
