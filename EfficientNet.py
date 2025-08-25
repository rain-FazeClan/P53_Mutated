import timm
import torch.nn as nn

class EfficientNet(nn.Module):
    def __init__(self, model_name='tf_efficientnet_b0', num_classes=2, in_chans=6, pretrained=True):
        """
        EfficientNet Classifier
        Args:
            model_name (str): EfficientNet model variant, e.g., 'tf_efficientnet_b0'
            num_classes (int): Number of output classes for classification
            in_chans (int): Number of input channels (default=6 for MRI data)
            pretrained (bool): Whether to load pretrained weights
        """
        super(EfficientNet, self).__init__()
        # Load pretrained EfficientNet model from timm
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            in_chans=in_chans,  # Custom input channels
            num_classes=0  # Remove default classification head
        )

        # Add custom classification head
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),  # Dropout for regularization
            nn.Linear(self.backbone.num_features, 256),  # Intermediate layer
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)  # Output layer with num_classes outputs
        )

    def forward(self, x):
        # Extract features using EfficientNet backbone
        features = self.backbone(x)
        # Pass features through custom classifier
        output = self.classifier(features)
        return output