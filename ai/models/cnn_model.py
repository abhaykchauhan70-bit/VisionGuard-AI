"""
ai/models/cnn_model.py
-----------------------
CNN FEATURE EXTRACTOR
======================
We use transfer learning: a ResNet18 pretrained on ImageNet, with its final
classification layer removed. What's left is a "feature extractor" - it
turns any 224x224 image into a 512-dimensional vector that summarizes
what's visually in that frame (shapes, textures, objects, motion blur, etc).

Why transfer learning?
  Training a CNN from scratch needs millions of labelled images. ResNet18
  was already trained on 1.2M ImageNet images and learned general-purpose
  visual features (edges -> textures -> shapes -> objects). We reuse those
  frozen weights and only train the small layers we add on top - this needs
  far less data and trains much faster.

Key concepts (for your viva/interview):
  - Convolution: a small filter slides over the image detecting local
    patterns (edges, corners). Many filters = many pattern detectors.
  - Pooling: downsamples the feature map (e.g. max-pooling) to reduce size
    and add translation invariance.
  - Feature extraction: passing an image through the conv layers to get a
    compact vector representation instead of raw pixels.
  - Feature vector: the 512-dim output we feed into the LSTM as one
    timestep of the sequence.
"""
import torch
import torch.nn as nn
import torchvision.models as models


class CNNFeatureExtractor(nn.Module):
    def __init__(self, freeze_backbone: bool = True):
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        # Drop the final fully-connected classification layer.
        # Everything up to (and including) global average pooling stays.
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.output_dim = 512  # resnet18's feature dimension

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, x):
        """
        x: (batch, 3, 224, 224) normalized image tensor
        returns: (batch, 512) feature vector
        """
        features = self.backbone(x)          # (batch, 512, 1, 1)
        features = torch.flatten(features, 1)  # (batch, 512)
        return features


if __name__ == "__main__":
    # quick self-test
    model = CNNFeatureExtractor()
    dummy = torch.randn(2, 3, 224, 224)
    out = model(dummy)
    print("Feature vector shape:", out.shape)  # torch.Size([2, 512])
