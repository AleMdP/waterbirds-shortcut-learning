import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

def get_waterbirds_model(num_classes=2, pretrained=True):
    """
    Returns a ResNet-50 architecture modified for Waterbirds classification.
    """
    weights = ResNet50_Weights.DEFAULT if pretrained else None
    model = resnet50(weights=weights)
    
    # Replace the final fully-connected layer for 2 classes (Landbird vs Waterbird)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    return model