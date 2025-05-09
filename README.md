# Breast Cancer Detection in Ultrasound Images

DeepLabV3+ model for segmenting breast cancer lesions in ultrasound images using PyTorch.

## Project Overview

This project implements a DeepLabV3+ model with a MIT-B2 encoder to segment breast lesions in ultrasound images from the Dataset_BUSI_with_GT dataset. The model achieves segmentation with Dice scores over 0.88.

## Dataset

The Dataset_BUSI_with_GT contains breast ultrasound images with three categories:
- Benign
- Malignant
- Normal

Each image has a corresponding mask for ground truth segmentation.

## Model Architecture

- Encoder: MIT-B2 pretrained on ImageNet
- Model: DeepLabV3+
- Loss: Combined Dice Loss and BCEWithLogitsLoss

## Results

After 200 epochs of training:
- Train Dice: 0.8896
- Validation Dice: 0.8861
- Validation IoU: 0.8188

## Requirements

- Python 3.8+
- PyTorch 1.10+
- torchvision
- segmentation-models-pytorch
- albumentations
- opencv-python
- numpy
- pandas
- matplotlib

## Usage

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Breast-Cancer-Detection.git
cd Breast-Cancer-Detection
