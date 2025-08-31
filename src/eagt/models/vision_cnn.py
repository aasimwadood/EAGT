import torch.nn as nn
import torchvision.models as models

class VisionCNN(nn.Module):
    """
    CNN backbone for facial embeddings.
    
    Defaults to ResNet18 pretrained on ImageNet, with
    the final FC removed → output (B, 512).
    """
    def __init__(self, backbone: str = "resnet18", pretrained: bool = True):
        super().__init__()
        if backbone == "resnet18":
            base = models.resnet18(pretrained=pretrained)
            self.feat_dim = base.fc.in_features
            base.fc = nn.Identity()
            self.model = base
        else:
            raise ValueError(f"Unsupported vision backbone: {backbone}")

    def forward(self, x):
        return self.model(x)  # (B, feat_dim)
