import timm
import torch.nn as nn

class ConvNeXt(nn.Module):
    def __init__(self, model_name='convnext_tiny', num_classes=2, in_chans=6, pretrained=True):
        """
        ConvNeXt Classifier
        Args:
            model_name (str): ConvNeXt model variant, e.g., 'convnext_tiny', 'convnext_small', 'convnext_base'
            num_classes (int): Number of output classes for classification
            in_chans (int): Number of input channels (default=6 for MRI data)
            pretrained (bool): Whether to load pretrained weights
        """
        super(ConvNeXt, self).__init__()
        # Load pretrained ConvNeXt model from timm
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            in_chans=in_chans,  # Custom input channels
            num_classes=0  # Remove default classification head
        )

        # Add custom classification head
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.backbone.num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # Extract features using ConvNeXt backbone
        features = self.backbone(x)
        # Pass features through custom classifier
        output = self.classifier(features)
        return output