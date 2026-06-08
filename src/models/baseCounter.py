import torch.nn as nn

class BaseCounter(nn.Module):
    """Abstract base class for car count classifiers."""

    def __init__(self, num_classes, freeze_backbone=True):
        super().__init__()
        self.backbone = self._build_backbone(freeze_backbone)
        in_features = self._get_in_features()
        self._replace_head()
        self.head = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def _build_backbone(self, freeze_backbone):
        raise NotImplementedError

    def _get_in_features(self):
        raise NotImplementedError

    def _replace_head(self):
        raise NotImplementedError

    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)

    def get_embeddings(self, x):
        return self.backbone(x)