import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import timm
from dataset_classification import ClassificationDataset
from transforms_classification import get_train_transform

def train_classification(args):
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Prepare dataset and dataloaders (using training transforms for all for simplicity)
    img_transform, mask_transform = get_train_transform(args.img_size)
    dataset = ClassificationDataset(args.images_dir, args.masks_dir,
                                    transform_img=img_transform, transform_mask=mask_transform)
    # Split into train and validation
    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")

    # Model (ConvNeXt-Base with 4 input channels)
    num_classes = len(dataset.class_to_idx)
    model = timm.create_model('convnext_base', pretrained=True, in_chans=4, num_classes=num_classes)
    model.to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_correct = 0
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            total_correct += (preds == labels).sum().item()
        train_loss = total_loss / len(train_loader.dataset)
        train_acc = total_correct / len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        val_correct = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
        val_loss = val_loss / len(val_loader.dataset) if len(val_loader.dataset) > 0 else 0
        val_acc = val_correct / len(val_loader.dataset) if len(val_loader.dataset) > 0 else 0

        print(f"Epoch [{epoch}/{args.epochs}], Train Loss: {train_loss:.4f}, "
              f"Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), args.save_path)
            print(f"Saved best model to {args.save_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train classification model')
    parser.add_argument('--images-dir', type=str, required=True, help='Path to input images directory')
    parser.add_argument('--masks-dir', type=str, required=True, help='Path to input masks directory')
    parser.add_argument('--epochs', type=int, default=25, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--img-size', type=int, default=224, help='Image size (square)')
    parser.add_argument('--val-split', type=float, default=0.2, help='Fraction of data for validation')
    parser.add_argument('--save-path', type=str, default='best_classification.pth', help='Path to save best model')
    args = parser.parse_args()
    train_classification(args)
