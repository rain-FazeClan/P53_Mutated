import timm
import torch
import torch.nn as nn

class ResNet(nn.Module):
    def __init__(self, model_name='resnet50', num_classes=2, in_chans=6, pretrained=True):
        """
        ResNet50 Classifier
        Args:
            model_name (str): ResNet model variant, e.g., 'resnet50', 'resnet34', 'resnet101'
            num_classes (int): Number of output classes for classification
            in_chans (int): Number of input channels (default=6 for MRI data)
            pretrained (bool): Whether to load pretrained weights
        """
        super(ResNet, self).__init__()

        # Load pretrained ResNet model from timm
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0  # Remove default classification head
        )

        # Manually modify first conv layer for 6-channel input if needed
        if in_chans != 3:
            self._modify_first_conv(in_chans)

        # Add custom classification head
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),  # Dropout for regularization
            nn.Linear(self.backbone.num_features, 512),  # Intermediate layer
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)  # Output layer with num_classes outputs
        )

    def _modify_first_conv(self, in_chans):
        """
        Modify the first convolutional layer to accept custom number of input channels
        """
        old_conv = self.backbone.conv1

        # Create new conv layer with custom input channels
        new_conv = nn.Conv2d(
            in_chans,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None
        )

        # Initialize weights for the new conv layer
        with torch.no_grad():
            if in_chans == 6:
                # For 6-channel input, duplicate the original 3-channel weights
                new_conv.weight[:, :3] = old_conv.weight
                new_conv.weight[:, 3:] = old_conv.weight
            elif in_chans == 1:
                # For 1-channel input, average the original 3-channel weights
                new_conv.weight[:, 0] = old_conv.weight.mean(dim=1)
            else:
                # For other cases, use Xavier initialization
                nn.init.xavier_uniform_(new_conv.weight)

            if old_conv.bias is not None:
                new_conv.bias = old_conv.bias

        # Replace the first conv layer
        self.backbone.conv1 = new_conv

    def forward(self, x):
        """
        Forward pass
        Args:
            x: Input tensor of shape (batch_size, in_chans, height, width)
        Returns:
            output: Classification logits of shape (batch_size, num_classes)
        """
        # Extract features using ResNet backbone
        features = self.backbone(x)
        # Pass features through custom classifier
        output = self.classifier(features)
        return output

    def get_feature_extractor(self):
        """
        Return the backbone for feature extraction
        """
        return self.backbone