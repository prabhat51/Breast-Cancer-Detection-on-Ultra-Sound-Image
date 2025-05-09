import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

class ClassificationDataset(Dataset):
    """
    PyTorch Dataset for loading image, mask and label for classification.
    Expects images and masks directories with class subdirectories for each label.
    """
    def __init__(self, images_dir, masks_dir, transform_img=None, transform_mask=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform_img = transform_img
        self.transform_mask = transform_mask

        # Find class subdirectories
        self.classes = sorted([d for d in os.listdir(images_dir) if os.path.isdir(os.path.join(images_dir, d))])
        if len(self.classes) == 0:
            raise RuntimeError(f"No class directories found in {images_dir}")
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}

        # List all (image_path, mask_path, label) tuples
        self.samples = []
        for cls_name in self.classes:
            cls_idx = self.class_to_idx[cls_name]
            cls_img_dir = os.path.join(images_dir, cls_name)
            cls_mask_dir = os.path.join(masks_dir, cls_name)
            if not os.path.isdir(cls_mask_dir):
                raise RuntimeError(f"Mask directory for class {cls_name} not found at {cls_mask_dir}")
            for fname in os.listdir(cls_img_dir):
                if fname.endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(cls_img_dir, fname)
                    # Assume mask has same filename in masks directory
                    mask_path = os.path.join(cls_mask_dir, fname)
                    if not os.path.exists(mask_path):
                        raise RuntimeError(f"Mask {mask_path} not found")
                    self.samples.append((img_path, mask_path, cls_idx))

        if len(self.samples) == 0:
            raise RuntimeError("No image-mask pairs found.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path, label = self.samples[idx]
        # Load image and mask
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        if self.transform_img:
            image = self.transform_img(image)
        if self.transform_mask:
            mask = self.transform_mask(mask)
        # Ensure mask is tensor
        import torch
        if not isinstance(mask, torch.Tensor):
            mask = torch.from_numpy(np.array(mask, dtype='float32')).unsqueeze(0)
        # Concatenate image and mask to create 4-channel input
        image_mask = torch.cat([image, mask], dim=0)  # shape: 4 x H x W
        label_tensor = torch.tensor(label, dtype=torch.long)
        return image_mask, label_tensor
