import torch.nn as nn

def get_waterbirds_model(num_classes=2, pretrained=True):
    """
    Returns a ResNet-50 architecture modified for Waterbirds classification.
    """
    from torchvision.models import resnet50, ResNet50_Weights

    weights = ResNet50_Weights.DEFAULT if pretrained else None
    model = resnet50(weights=weights)
    
    # Replace the final fully-connected layer for 2 classes (Landbird vs Waterbird)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    return model

def get_transformer_model(num_classes=2, pretrained=True):
    """
    Returns a Vision Transformer architecture modified for Waterbirds classification.
    """
    from torchvision.models import vit_b_16, ViT_B_16_Weights
    
    weights = ViT_B_16_Weights.DEFAULT if pretrained else None
    model = vit_b_16(weights=weights)
    
    # Replace the final fully-connected layer for 2 classes (Landbird vs Waterbird)
    model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    
    return model