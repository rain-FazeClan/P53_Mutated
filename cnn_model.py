import torch
import torch.nn as nn
import torch.nn.functional as F

# --- 3D ResNet ---
# BasicBlock for ResNet18/34
class BasicBlock3D(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(BasicBlock3D, self).__init__()
        self.conv1 = nn.Conv3d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet3D(nn.Module):
    # ResNet-18 3D variant
    def __init__(self, block=BasicBlock3D, layers=[2, 2, 2, 2], num_classes=1, input_channels=4):
        super(ResNet3D, self).__init__()
        self.in_planes = 64

        # Initial convolution - adjust kernel size/stride/padding as needed for 3D
        # Input: (Batch, Channels, Depth, Height, Width) e.g., (N, 4, 32, 224, 224)
        self.conv1 = nn.Conv3d(input_channels, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3), bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1) # Check stride/padding

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1)) # Global Average Pooling
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # Initialize weights (as mentioned Glorot in paper, but Kaiming is common for ResNets)
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                 nn.init.xavier_uniform_(m.weight) # Glorot uniform for Linear
                 if m.bias is not None:
                     nn.init.constant_(m.bias, 0)


    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.in_planes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.in_planes, planes, stride, downsample))
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        # x shape: (N, C, D, H, W)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x) # Shape potentially changes significantly here

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x) # Shape: (N, 512*expansion, 1, 1, 1)
        features = torch.flatten(x, 1) # Shape: (N, 512*expansion)
        out = self.fc(features) # Shape: (N, num_classes)

        # Apply sigmoid for binary classification output probability
        if self.fc.out_features == 1:
            out = torch.sigmoid(out)

        return out, features # Return both output and features before final FC

# --- 3D VGGNet (VGG11 variant with BatchNorm) ---
# Configuration for VGG11: [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
}

class VGGNet3D(nn.Module):
    def __init__(self, vgg_name='VGG11', num_classes=1, input_channels=4, init_weights=True):
        super(VGGNet3D, self).__init__()
        self.features = self._make_layers(cfg[vgg_name], input_channels)
        # Calculate the size after feature extraction - this depends heavily on input size and pooling
        # For input (N, C, 32, 224, 224) and VGG11 pooling, it gets complex.
        # Using AdaptiveAvgPool avoids needing to calculate the exact size.
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1)) # Output size (512, 1, 1, 1)
        self.classifier = nn.Sequential(
            nn.Linear(512 * 1 * 1 * 1, 4096), # Adjust 512 if VGG config changes
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        features = torch.flatten(x, 1)
        out = self.classifier(features)
        # Apply sigmoid for binary classification output probability
        if self.classifier[-1].out_features == 1:
             out = torch.sigmoid(out)
        return out, features # Return output and flattened features before classifier

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01) # VGG often uses normal init for linear
                nn.init.constant_(m.bias, 0)

    def _make_layers(self, cfg, input_channels, batch_norm=True):
        layers = []
        in_channels = input_channels
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool3d(kernel_size=2, stride=2)] # Standard VGG max pool
            else:
                conv3d = nn.Conv3d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv3d, nn.BatchNorm3d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv3d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)