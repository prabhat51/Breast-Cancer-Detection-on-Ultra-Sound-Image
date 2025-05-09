import segmentation_models_pytorch as smp

def get_segmentation_model(arch='Unet', encoder='efficientnet-b0', num_classes=1):
    return getattr(smp, arch)(encoder_name=encoder, in_channels=3, classes=num_classes)


def get_classification_model(num_classes=3):
    from torchvision import models
    import torch.nn as nn
    model = models.efficientnet_b0(pretrained=True)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model
