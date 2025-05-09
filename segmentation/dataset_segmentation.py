import os
import cv2
import torch
from torch.utils.data import Dataset

class SegmentationDataset(Dataset):
    """
    PyTorch Dataset for loading images and binary masks for segmentation.
    Expects masks to have a '_mask' suffix before the extension.
    Example: image: 'benign (1).png' → mask: 'benign (1)_mask.png'
    """
    def __init__(self, images_dir, masks_dir, transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform

        # List image files
        self.image_names = sorted([f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])
        if len(self.image_names) == 0:
            raise RuntimeError(f"No images found in {images_dir}")
        
    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        # Load image
        img_name = self.image_names[idx]
        img_path = os.path.join(self.images_dir, img_name)
        image = cv2.imread(img_path)
        if image is None:
            raise RuntimeError(f"Failed to read image {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Modify mask name to include '_mask' suffix
        mask_base = os.path.splitext(img_name)[0] + '_mask'
        mask_path = None
        for ext in ['.png', '.jpg', '.jpeg']:
            candidate = os.path.join(self.masks_dir, mask_base + ext)
            if os.path.exists(candidate):
                mask_path = candidate
                break
        if mask_path is None:
            raise RuntimeError(f"Mask not found for {img_name} in {self.masks_dir}")

        # Load mask
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"Failed to read mask {mask_path}")
        mask = mask.astype('float32') / 255.0
        mask = mask.clip(0, 1)

        # Apply transforms
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
            if mask.ndim == 2:
                mask = mask.unsqueeze(0)
        else:
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
            mask = torch.from_numpy(mask).unsqueeze(0).float()
        return image, mask
