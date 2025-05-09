# Breast Ultrasound Segmentation and Classification

## Project Overview
This repository implements a pipeline for **breast ultrasound image analysis** using deep learning. It includes two main tasks: (1) **segmentation** of the ultrasound image to identify regions of interest (lesions) using a DeepLabV3+ model, and (2) **classification** of the lesions as benign, malignant, or normal using a ConvNeXt-Base model. The classification model takes a 4-channel input (the original RGB image plus the predicted mask from the segmentation model) to improve accuracy. A prediction script (`predict_pipeline.py`) ties both tasks together: it loads a trained segmentation model to predict a mask for a given image, then feeds the image and mask into the trained classification model to output the final prediction.

## Dataset Format
The repository expects the **BUSI (Breast Ultrasound) dataset** with corresponding ground truth masks. The data directory should have the following structure:
- `Dataset_BUSI_with_GT/`
  - `benign/` - containing benign ultrasound images (e.g., `benign-01.png`)
  - `malignant/` - containing malignant ultrasound images (e.g., `malignant-01.png`)
  - `normal/` - containing normal ultrasound images (e.g., `normal-01.png`)
  - Each image file should have a corresponding mask file with the same base name and `_mask` suffix (e.g., `benign-01_mask.png`).

The images are assumed to be color (RGB) ultrasound scans, and masks are binary or multi-class PNG images indicating lesion regions.

## Model Architecture
- **Segmentation Model:** A DeepLabV3+ network with a MiT-B2 (Vision Transformer) encoder from `segmentation_models_pytorch`. This model segments the breast ultrasound image to highlight potential lesion areas.
- **Classification Model:** A ConvNeXt-Base network from the `timm` library, modified to accept a 4-channel input. The three standard channels are the original image, and the fourth channel is the predicted (or ground truth) mask. The classifier outputs one of three classes: benign, malignant, or normal.

## Installation
Ensure you have Python (>=3.8) installed. Install the required packages:
```bash
pip install torch torchvision timm segmentation_models_pytorch albumentations Pillow
```
You may optionally set up a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows use venv\Scripts\activate
pip install -r requirements.txt  # if a requirements file is provided
```

## Training

### Segmentation Model Training
Train the segmentation model using `segmentation/train_segmentation.py`. You need to specify the dataset directory, number of epochs, batch size, and where to save the model. For example:
```bash
python segmentation/train_segmentation.py \
  --images-dir /path/to/images \
  --masks-dir /path/to/masks \
  --epochs 30 \
  --lr 0.001 \
  --batch-size 4 \
  --img-size 256 \
  --val-split 0.2 \
  --save-path segmentation_model.pth
```
```
segmentation_dataset/
├── images/
│   ├── benign (1).png
│   ├── malignant (2).png
│   └── normal (3).png
├── masks/
│   ├── benign (1)_mask.png
│   ├── malignant (2)_mask.png
│   └── normal (3)_mask.png

```

### Classification Model Training
Train the classification model using `classification/train_classification.py`. The model uses the ultrasound image plus a mask as input. You can use the ground truth masks or generate masks by running the trained segmentation model on the images. Example:
```bash
python classification/train_classification.py \
  --images-dir /path/to/images \
  --masks-dir /path/to/masks \
  --epochs 30 \
  --lr 0.001 \
  --batch-size 16 \
  --img-size 256 \
  --val-split 0.2 \
  --save-path classification_model.pth
```
```
classification_dataset/
├── images/
│   ├── benign/
│   │   ├── benign (1).png
│   │   ├── benign (2).png
│   ├── malignant/
│   │   ├── malignant (1).png
│   │   ├── malignant (2).png
│   └── normal/
│       ├── normal (1).png
│       ├── normal (2).png
├── masks/
│   ├── benign/
│   │   ├── benign (1)_mask.png
│   │   ├── benign (2)_mask.png
│   ├── malignant/
│   │   ├── malignant (1)_mask.png
│   │   ├── malignant (2)_mask.png
│   └── normal/
│       ├── normal (1)_mask.png
│       ├── normal (2)_mask.png

```
## Prediction Pipeline
Once both models are trained, use `predict_pipeline.py` to classify new images. The script will:
1. Load the trained segmentation model and predict a mask for the input image.
2. Combine the input image and the predicted mask into a 4-channel tensor.
3. Load the trained classification model and predict the class label.
4. Print the predicted class and confidence score.

### Usage
```bash
python predict_pipeline.py \
    --image "/path/to/image.png" \
    --seg_model "/path/to/segmentation_model.pth" \
    --cls_model "/path/to/classification_model.pth" \
    --save_mask "/path/to/predicted_mask.png"
```

### Expected Outputs
```
Predicted class: benign (0.92 confidence)
```


## Credits and References
This implementation uses:
- [segmentation_models_pytorch](https://github.com/qubvel/segmentation_models.pytorch) for DeepLabV3+.
- [timm](https://github.com/huggingface/pytorch-image-models) for ConvNeXt-Base.
- [Dataset](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset)
