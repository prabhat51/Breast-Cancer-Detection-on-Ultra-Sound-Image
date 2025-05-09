import os
import cv2
import numpy as np
from tqdm import tqdm

def resize_image_and_mask(image, mask, size=(256, 256)):
    return cv2.resize(image, size), cv2.resize(mask, size)

def load_paths(base_path):
    image_paths, mask_paths = [], []
    for class_name in os.listdir(base_path):
        class_path = os.path.join(base_path, class_name)
        if not os.path.isdir(class_path):
            continue
        for img_file in os.listdir(class_path):
            if "mask" in img_file:
                mask_paths.append(os.path.join(class_path, img_file))
            else:
                image_paths.append(os.path.join(class_path, img_file))
    return sorted(image_paths), sorted(mask_paths)
