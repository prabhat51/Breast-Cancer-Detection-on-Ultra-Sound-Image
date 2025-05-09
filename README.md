# BCDet - Breast Cancer Detection using Ultrasound Imaging

This repository contains a modular deep learning pipeline for breast cancer detection using ultrasound imaging.

## Features
- Segmentation (UNet/FPN + EfficientNet)
- Classification (EfficientNet/ConvNeXt)
- Modular & scalable

## Installation
```bash
git clone https://github.com/<your-username>/BCDet-BreastCancer-Detection.git
cd BCDet-BreastCancer-Detection
pip install -r requirements.txt
```

## Usage
Place images and masks in `data/` folder, organized by class.
Train models using scripts in `src/` folder.

## Directory Structure
```
project_root/
├── src/: core modules
├── utils/: helper utilities
├── data/: dataset folder
├── outputs/: logs, models, and figures
└── notebook/: notebook (for reference)
```
