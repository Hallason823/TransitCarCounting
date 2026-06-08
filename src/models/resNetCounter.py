import torch.nn as nn
from torchvision import models

from .baseCounter import BaseCounter

class ResNetCounter(BaseCounter):
    """ResNet18 backbone fine-tuned for car count classification."""

    def __init__(self, num_classes, freeze_backbone=True):
        self._freeze = freeze_backbone
        super().__init__(num_classes, freeze_backbone)

    def _build_backbone(self, freeze_backbone):
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        if freeze_backbone:
            for param in backbone.parameters():
                param.requires_grad = False
        return backbone

    def _get_in_features(self):
        return self.backbone.fc.in_features

    def _replace_head(self):
        self.backbone.fc = nn.Identity()