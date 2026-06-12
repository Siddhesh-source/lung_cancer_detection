"""
KUB Risk Detector - Data Preparation Pipeline
Dataset cleaning, loading, and preparation logic for training
"""

import os
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import yaml
from sklearn.model_selection import train_test_split

# Load configuration from root config.yaml
with open("config.yaml", "r") as f:
    _CONFIG = yaml.safe_load(f)


class DatasetPreparator:
    """Handles dataset preparation, cleaning, and splitting."""
    
    SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    
    def __init__(
        self,
        normal_dir: str = None,
        abnormal_dir: str = None
    ):
        # Dynamically extract paths from config
        self.normal_dir = Path(normal_dir or _CONFIG['data_paths']['normal'])
        self.abnormal_dir = Path(abnormal_dir or _CONFIG['data_paths']['abnormal'])
        
        self.normal_dir.mkdir(parents=True, exist_ok=True)
        self.abnormal_dir.mkdir(parents=True, exist_ok=True)
    
    def validate_image(self, image_path: Path) -> bool:
        """Check if image is valid and not corrupted."""
        try:
            img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            return img is not None and img.size > 0
        except Exception:
            return False
    
    def scan_directory(self, directory: Path) -> List[Path]:
        """Scan directory for valid image files."""
        images = []
        if not directory.exists():
            return images
        
        for ext in self.SUPPORTED_FORMATS:
            images.extend(directory.glob(f"*{ext}"))
            images.extend(directory.glob(f"*{ext.upper()}"))
        
        return sorted(set(images))
    
    def clean_dataset(self) -> Dict[str, List[Path]]:
        """
        Clean dataset by removing corrupted images.
        
        Returns:
            Dict with 'normal' and 'abnormal' lists of valid image paths
        """
        print("Scanning and cleaning dataset...")
        
        cleaned = {'normal': [], 'abnormal': []}
        
        for label, dir_path in [('normal', self.normal_dir), ('abnormal', self.abnormal_dir)]:
            images = self.scan_directory(dir_path)
            
            valid_count = 0
            invalid_count = 0
            
            for img_path in images:
                if self.validate_image(img_path):
                    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                    h, w = img.shape[:2]
                    if min(h, w) >= 32:
                        cleaned[label].append(img_path)
                        valid_count += 1
                    else:
                        invalid_count += 1
                        print(f"  {label} - Too small ({h}x{w}): {img_path.name}")
                else:
                    invalid_count += 1
                    print(f"  {label} - Corrupted: {img_path.name}")
            
            print(f"  {label}: {valid_count} valid, {invalid_count} invalid")
        
        return cleaned
    
    def split_dataset(
        self,
        normal_images: List[Path],
        abnormal_images: List[Path],
        train_ratio: float = 0.8,
        random_seed: int = 42
    ) -> Tuple[Dict, Dict]:
        """
        Split dataset into train/val sets while maintaining balance.
        
        Returns:
            Tuple of (train_dict, val_dict) with 'normal' and 'abnormal' lists
        """
        print(f"Splitting dataset (train_ratio={train_ratio})...")
        
        train_normal, val_normal = train_test_split(
            normal_images,
            train_size=train_ratio,
            random_state=random_seed
        )
        
        train_abnormal, val_abnormal = train_test_split(
            abnormal_images,
            train_size=train_ratio,
            random_state=random_seed
        )
        
        train_set = {'normal': list(train_normal), 'abnormal': list(train_abnormal)}
        val_set = {'normal': list(val_normal), 'abnormal': list(val_abnormal)}
        
        print(f"  Train: {len(train_normal)} normal, {len(train_abnormal)} abnormal")
        print(f"  Val:   {len(val_normal)} normal, {len(val_abnormal)} abnormal")
        
        return train_set, val_set
    
    def get_dataset_stats(self) -> Dict:
        """Get statistics about the prepared dataset."""
        normal_count = len(self.scan_directory(self.normal_dir))
        abnormal_count = len(self.scan_directory(self.abnormal_dir))
        
        return {
            'normal': normal_count,
            'abnormal': abnormal_count,
            'total': normal_count + abnormal_count
        }


if __name__ == "__main__":
    print("KUB Risk Detector - Data Preparation")
    print("=" * 50)
    
    preparator = DatasetPreparator()
    stats = preparator.get_dataset_stats()
    
    print(f"\nDataset Statistics:")
    print(f"  Normal images: {stats['normal']}")
    print(f"  Abnormal images: {stats['abnormal']}")
    print(f"  Total: {stats['total']}")
    print("\nUse DatasetPreparator to clean and split the dataset.")
