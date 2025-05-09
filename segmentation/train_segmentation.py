import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import segmentation_models_pytorch as smp
from dataset_segmentation import SegmentationDataset
from transforms_segmentation import get_train_transform, get_val_transform

def train_segmentation(args):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Prepare dataset and dataloaders
    dataset = SegmentationDataset(args.images_dir, args.masks_dir, transform=get_train_transform(args.img_size))
    # Split into train and validation
    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    # Update transforms for validation dataset
    val_dataset.dataset.transform = get_val_transform(args.img_size)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")

    # Initialize model
    model = smp.DeepLabV3Plus(encoder_name='mit_b2', encoder_weights='imagenet', in_channels=3, classes=1)
    model.to(device)

    # Loss functions
    dice_loss = smp.losses.DiceLoss(mode='binary')
    bce_loss = nn.BCEWithLogitsLoss()

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_loss = float('inf')
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images)
            loss = dice_loss(outputs, masks) + bce_loss(outputs, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
        train_loss /= len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)
                outputs = model(images)
                loss = dice_loss(outputs, masks) + bce_loss(outputs, masks)
                val_loss += loss.item() * images.size(0)
        val_loss /= len(val_loader.dataset)

        print(f"Epoch [{epoch}/{args.epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), args.save_path)
            print(f"Saved best model to {args.save_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train segmentation model')
    parser.add_argument('--images-dir', type=str, required=True, help='Path to input images directory')
    parser.add_argument('--masks-dir', type=str, required=True, help='Path to input masks directory')
    parser.add_argument('--epochs', type=int, default=25, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--img-size', type=int, default=256, help='Image size (will be squared)')
    parser.add_argument('--val-split', type=float, default=0.2, help='Fraction of data for validation')
    parser.add_argument('--save-path', type=str, default='best_segmentation.pth', help='Path to save best model')
    args = parser.parse_args()
    train_segmentation(args)
