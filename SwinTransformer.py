import timm
import torch.nn as nn

class SwinTransformer(nn.Module):
    def __init__(self, model_name='swin_tiny_patch4_window7_224', num_classes=2, in_chans=6, pretrained=True):
        """
        Swin Transformer Classifier
        Args:
            model_name (str): Swin Transformer variant, e.g., 'swin_tiny_patch4_window7_224'
            num_classes (int): Number of output classes
            in_chans (int): Number of input channels (default=6 for MRI data)
            pretrained (bool): Whether to load pretrained weights
        """
        super(SwinTransformer, self).__init__()
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
        features = self.backbone(x)
        output = self.classifier(features)
        return output