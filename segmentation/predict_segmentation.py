import argparse
import cv2
import torch
import numpy as np
import segmentation_models_pytorch as smp
from transforms_segmentation import get_val_transform

def predict_segmentation(args):
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load model
    model = smp.DeepLabV3Plus(encoder_name='mit_b2', encoder_weights=None, in_channels=3, classes=1)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    model.eval()

    # Read and preprocess image
    image = cv2.imread(args.image_path)
    if image is None:
        raise RuntimeError(f"Failed to read image {args.image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = image.shape[:2]
    transform = get_val_transform(args.img_size)
    augmented = transform(image=image)
    input_image = augmented['image'].unsqueeze(0).to(device)  # add batch dimension

    # Predict mask
    with torch.no_grad():
        output = model(input_image)
        prob = torch.sigmoid(output)[0,0]  # remove batch and channel dims
        mask = (prob > args.threshold).float().cpu().numpy()
    # Resize mask back to original size
    mask = cv2.resize(mask, (orig_w, orig_h))
    mask = (mask * 255).astype(np.uint8)

    # Save mask
    cv2.imwrite(args.output_path, mask)
    print(f"Saved predicted mask to {args.output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict segmentation mask for an image')
    parser.add_argument('--image-path', type=str, required=True, help='Path to input image')
    parser.add_argument('--model-path', type=str, required=True, help='Path to trained segmentation model (.pth)')
    parser.add_argument('--output-path', type=str, required=True, help='Path to save output mask')
    parser.add_argument('--img-size', type=int, default=256, help='Image size for model input (square)')
    parser.add_argument('--threshold', type=float, default=0.5, help='Threshold to convert probability to binary mask')
    args = parser.parse_args()
    predict_segmentation(args)
