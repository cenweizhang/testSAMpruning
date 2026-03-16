# -*- coding: utf-8 -*-
"""
CVC-ColonDB Dataset Loader for MedSAM Pruning Pilot Study.

Loads PNG images and masks, resizes to 1024x1024, normalizes to [0,1],
and extracts bounding box prompts from masks.
"""

import os
import random
import numpy as np
from skimage import io, transform as sk_transform
import torch
from torch.utils.data import Dataset, DataLoader


class PolypDataset(Dataset):
    """
    Dataset for polyp segmentation.
    
    Directory structure:
        data_root/
            images/   -> *.png (RGB colonoscopy images)
            masks/    -> *.png (binary segmentation masks, same filenames)
    """
    
    def __init__(self, data_root, image_size=1024, bbox_shift=5):
        """
        Args:
            data_root: path to dataset root (e.g., assert/CVC-ColonDB)
            image_size: target size for SAM input (default 1024)
            bbox_shift: random perturbation range for bounding box (pixels in 1024 space)
        """
        self.data_root = data_root
        self.image_size = image_size
        self.bbox_shift = bbox_shift
        
        img_dir = os.path.join(data_root, "images")
        mask_dir = os.path.join(data_root, "masks")
        
        # Get sorted file list (ensure image-mask correspondence)
        self.img_files = sorted(
            [f for f in os.listdir(img_dir) if f.endswith(".png")],
            key=lambda x: int(os.path.splitext(x)[0])
        )
        self.mask_files = sorted(
            [f for f in os.listdir(mask_dir) if f.endswith(".png")],
            key=lambda x: int(os.path.splitext(x)[0])
        )
        
        assert len(self.img_files) == len(self.mask_files), \
            f"Image/mask count mismatch: {len(self.img_files)} vs {len(self.mask_files)}"
        
        self.img_dir = img_dir
        self.mask_dir = mask_dir
    
    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, index):
        # --- Load image ---
        img_path = os.path.join(self.img_dir, self.img_files[index])
        img = io.imread(img_path)  # (H, W, 3) uint8
        
        if len(img.shape) == 2:
            img = np.stack([img] * 3, axis=-1)
        elif img.shape[2] == 4:
            img = img[:, :, :3]  # drop alpha
        
        H_orig, W_orig = img.shape[:2]
        
        # Resize to 1024x1024 and normalize to [0, 1]
        img_1024 = sk_transform.resize(
            img, (self.image_size, self.image_size),
            order=3, preserve_range=True, anti_aliasing=True
        ).astype(np.float64)
        img_1024 = (img_1024 - img_1024.min()) / np.clip(
            img_1024.max() - img_1024.min(), a_min=1e-8, a_max=None
        )
        img_tensor = torch.tensor(img_1024).float().permute(2, 0, 1)  # (3, 1024, 1024)
        
        # --- Load mask ---
        mask_path = os.path.join(self.mask_dir, self.mask_files[index])
        mask = io.imread(mask_path)  # could be (H, W) or (H, W, C)
        
        if len(mask.shape) == 3:
            mask = mask[:, :, 0]  # take first channel
        
        # Binarize
        mask = (mask > 127).astype(np.uint8)
        
        # Resize mask to 256x256 (mask decoder output resolution)
        mask_256 = sk_transform.resize(
            mask, (256, 256), order=0, preserve_range=True, anti_aliasing=False
        ).astype(np.uint8)
        
        # Also keep a 1024x1024 mask for evaluation
        mask_1024 = sk_transform.resize(
            mask, (self.image_size, self.image_size),
            order=0, preserve_range=True, anti_aliasing=False
        ).astype(np.uint8)
        
        # --- Extract bounding box from 1024x1024 mask ---
        y_indices, x_indices = np.where(mask_1024 > 0)
        if len(y_indices) == 0:
            # Empty mask fallback
            bbox = np.array([0, 0, self.image_size, self.image_size], dtype=np.float32)
        else:
            x_min, x_max = np.min(x_indices), np.max(x_indices)
            y_min, y_max = np.min(y_indices), np.max(y_indices)
            # Add random shift
            x_min = max(0, x_min - random.randint(0, self.bbox_shift))
            x_max = min(self.image_size, x_max + random.randint(0, self.bbox_shift))
            y_min = max(0, y_min - random.randint(0, self.bbox_shift))
            y_max = min(self.image_size, y_max + random.randint(0, self.bbox_shift))
            bbox = np.array([x_min, y_min, x_max, y_max], dtype=np.float32)
        
        return {
            "image": img_tensor,                                    # (3, 1024, 1024) float [0,1]
            "mask_256": torch.tensor(mask_256).long().unsqueeze(0), # (1, 256, 256)
            "mask_1024": torch.tensor(mask_1024).long(),            # (1024, 1024)
            "bbox": torch.tensor(bbox).float(),                     # (4,)
            "name": self.img_files[index],
            "original_size": (H_orig, W_orig),
        }


def build_dataloaders(data_root, n_calibration=128, batch_size=4, seed=42):
    """
    Split dataset into calibration and test sets.
    
    Args:
        data_root: path to CVC-ColonDB
        n_calibration: number of samples for gradient computation
        batch_size: batch size for data loading
        seed: random seed for reproducibility
    
    Returns:
        cal_loader, test_loader, cal_dataset, test_dataset
    """
    full_dataset = PolypDataset(data_root)
    
    n_total = len(full_dataset)
    assert n_calibration < n_total, \
        f"Calibration size {n_calibration} >= total {n_total}"
    
    # Fixed split for reproducibility
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(n_total, generator=generator).tolist()
    
    cal_indices = indices[:n_calibration]
    test_indices = indices[n_calibration:]
    
    cal_dataset = torch.utils.data.Subset(full_dataset, cal_indices)
    test_dataset = torch.utils.data.Subset(full_dataset, test_indices)
    
    cal_loader = DataLoader(
        cal_dataset, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=1, shuffle=False,
        num_workers=0, pin_memory=True
    )
    
    print(f"Dataset split: {n_calibration} calibration, {len(test_indices)} test")
    return cal_loader, test_loader, cal_dataset, test_dataset
