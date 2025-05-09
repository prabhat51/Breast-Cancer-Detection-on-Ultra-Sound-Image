import argparse
import torch
from PIL import Image
import timm
from transforms_classification import get_val_transform

def predict_classification(args):
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load model (ConvNeXt-Base, 4 input channels, 3 classes)
    model = timm.create_model('convnext_base', pretrained=False, in_chans=4, num_classes=3)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    model.eval()

    # Prepare transforms
    img_transform, mask_transform = get_val_transform(args.img_size)

    # Read image and mask
    image = Image.open(args.image_path).convert('RGB')
    mask = Image.open(args.mask_path).convert('L')

    # Apply transforms
    image = img_transform(image).to(device)
    mask = mask_transform(mask).to(device)
    input_tensor = torch.cat([image, mask], dim=0).unsqueeze(0)  # shape: 1 x 4 x H x W

    # Predict
    with torch.no_grad():
        output = model(input_tensor)
        _, pred = torch.max(output, 1)
        pred = pred.item()
    # Map to class label
    classes = ['benign', 'malignant', 'normal']
    label = classes[pred] if pred < len(classes) else str(pred)
    print(f"Predicted class: {label}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict class label from image and mask')
    parser.add_argument('--image-path', type=str, required=True, help='Path to input image')
    parser.add_argument('--mask-path', type=str, required=True, help='Path to corresponding mask')
    parser.add_argument('--model-path', type=str, required=True, help='Path to trained classification model (.pth)')
    parser.add_argument('--img-size', type=int, default=224, help='Image size (square)')
    args = parser.parse_args()
    predict_classification(args)
