import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

class ResNet(nn.Module):
    def __init__(self, num_classes=1000, fixed_dim=512, feature_extraction=True):
        super(ResNet, self).__init__()
        # Load the pretrained ResNet50 model
        self.resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        
        # Linear layers to project features to a fixed dim
        self.fc1 = nn.Linear(256, fixed_dim)  # From layer1
        self.fc2 = nn.Linear(512, fixed_dim)  # From layer2
        self.fc3 = nn.Linear(1024, fixed_dim) # From layer3
        self.fc4 = nn.Linear(2048, fixed_dim) # From layer4

        # Final classification head for the full classification mode
        self.fc_classification = nn.Linear(2048, num_classes)

        self.feature_extraction = feature_extraction

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        # Extract features at different levels
        x1 = self.resnet.layer1(x)  # Low-level features
        x2 = self.resnet.layer2(x1)  # Mid-level features
        x3 = self.resnet.layer3(x2)  # Higher-level features
        x4 = self.resnet.layer4(x3)  # Even higher-level features

        # If in feature extraction mode, return intermediate features
        if self.feature_extraction:
            # Apply global average pooling (squeeze spatial dimensions)
            x1 = self.resnet.avgpool(x1).squeeze(-1).squeeze(-1)
            x2 = self.resnet.avgpool(x2).squeeze(-1).squeeze(-1)
            x3 = self.resnet.avgpool(x3).squeeze(-1).squeeze(-1)
            x4 = self.resnet.avgpool(x4).squeeze(-1).squeeze(-1)
            # Project each feature map to a fixed dimension
            x1 = self.fc1(x1)  # Shape: (batch_size, fixed_dim)
            x2 = self.fc2(x2)
            x3 = self.fc3(x3)
            x4 = self.fc4(x4)

            # Concatenate features along the sequence dimension (as context)
            # Shape: (batch_size, seq_len, fixed_dim)
            context = torch.stack([x1, x2, x3, x4], dim=1)
            return context

        else:
            return self.resent(x)
