import torch
import torch.nn as nn
import torch.nn.functional as F


class MiddleFusionModel(nn.Module):

    def __init__(self, architecture_class, num_classes, num_channels, hyperparameters):
        super().__init__()
        num_filters = hyperparameters["num_filters"]
        self.encoder = architecture_class(num_channels, num_filters)
        self.classifier = nn.Linear(num_filters, num_classes)
        self.fusion_weights = nn.Parameter(torch.zeros(num_channels))

    def fuse(self, all_features):
        weights = F.softmax(self.fusion_weights, dim=0).view(1, -1, 1)
        # weights: (1, C, 1)
        return (weights * all_features).sum(dim=1)

    def forward(self, x):
        # x: (B, C, L)
        all_features = self.encoder(x)
        # all_features: (B, C, F)
        fused_features = self.fuse(all_features)
        # fused_features: (B, F)
        logits = self.classifier(fused_features)
        # logits: (B, K)
        return {"logits": logits, "features": fused_features, "all_features": all_features}


class MiddleEqualFusionModel(MiddleFusionModel):

    def fuse(self, all_features):
        return all_features.mean(dim=1)


class MiddleNormFusionModel(MiddleFusionModel):

    def fuse(self, all_features):
        all_features = F.normalize(all_features, p=2, dim=2)
        return super().fuse(all_features)


class MiddleEqualNormFusionModel(MiddleEqualFusionModel):

    def fuse(self, all_features):
        all_features = F.normalize(all_features, p=2, dim=2)
        return super().fuse(all_features)
