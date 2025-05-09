import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode
from PIL import Image
import numpy as np
import argparse
import timm
import segmentation_models_pytorch as smp
import os

# Define class labels
CLASS_NAMES = ['benign', 'malignant', 'normal']

def load_image(image_path, image_size=256):
    """Load and preprocess image"""
    image = Image.open(image_path).convert('RGB')
    transform = T.Compose([
        T.Resize((image_size, image_size), interpolation=InterpolationMode.BILINEAR),
        T.ToTensor()
    ])
    return transform(image).unsqueeze(0)  # Shape: (1, 3, H, W)

def load_segmentation_model(model_path):
    model = smp.DeepLabV3Plus(
        encoder_name="mit_b2",
        encoder_weights=None,  # already trained
        in_channels=3,
        classes=1
    )
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model

def load_classification_model(model_path, num_classes=3):
    model = timm.create_model(
        'convnext_base',
        pretrained=False,
        in_chans=4,
        num_classes=num_classes
    )
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model

def predict_mask(seg_model, image_tensor):
    with torch.no_grad():
        mask_logits = seg_model(image_tensor)
        mask = torch.sigmoid(mask_logits)
        mask = (mask > 0.5).float()
    return mask  # Shape: (1, 1, H, W)

def predict_class(cls_model, image_tensor, mask_tensor):
    x = torch.cat([image_tensor, mask_tensor], dim=1)  # (1, 4, H, W)
    with torch.no_grad():
        logits = cls_model(x)
        probs = F.softmax(logits, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()
    return pred_class, probs[0][pred_class].item()

def save_mask(mask_tensor, path):
    mask_np = mask_tensor.squeeze().numpy() * 255
    Image.fromarray(mask_np.astype(np.uint8)).save(path)

def main(args):
    image_tensor = load_image(args.image, image_size=args.image_size)
    
    print("Loading segmentation model...")
    seg_model = load_segmentation_model(args.seg_model)
    print("Predicting mask...")
    mask_tensor = predict_mask(seg_model, image_tensor)

    if args.save_mask:
        save_mask(mask_tensor, args.save_mask)
        print(f"Saved predicted mask to: {args.save_mask}")
    
    print("Loading classification model...")
    cls_model = load_classification_model(args.cls_model)
    print("Predicting class...")
    pred_class, confidence = predict_class(cls_model, image_tensor, mask_tensor)
    
    print(f"Prediction: {CLASS_NAMES[pred_class]} ({confidence*100:.2f}%)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Segmentation + Classification Pipeline")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--seg_model", type=str, required=True, help="Path to trained segmentation model (.pt)")
    parser.add_argument("--cls_model", type=str, required=True, help="Path to trained classification model (.pt)")
    parser.add_argument("--image_size", type=int, default=256, help="Resize image to this size (default: 256)")
    parser.add_argument("--save_mask", type=str, default=None, help="Path to save predicted mask image")
    args = parser.parse_args()
    main(args)
