import torch
from sklearn.metrics import f1_score

def evaluate_model(model, val_loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            preds = torch.sigmoid(outputs).cpu().numpy() > 0.5
            all_preds.extend(preds.flatten())
            all_labels.extend(masks.cpu().numpy().flatten())

    f1 = f1_score(all_labels, all_preds)
    print(f"F1 Score: {f1:.4f}")
