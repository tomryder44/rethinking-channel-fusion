import torch
import torch.nn as nn
import torch.nn.functional as F


class MiddleFusionModel(nn.Module):

    def __init__(self, encoder_class, num_classes, num_channels, hyperparameters):
        super().__init__()
        num_filters = hyperparameters["num_filters"]
        self.num_channels = num_channels
        self.encoders = nn.ModuleList([encoder_class(1, num_filters) for _ in range(num_channels)])
        self.fusion_weights = nn.Parameter(torch.zeros(num_channels))
        self.classifier = nn.Linear(num_filters, num_classes)

    def forward(self, x):
        # x: (B, C, L)
        features_list = []
        for i in range(self.num_channels):
            channel_input = x[:, i, :].unsqueeze(1)
            # channel_input: (B, 1, L)
            feats = self.encoders[i](channel_input)
            # feats: (B, F)
            features_list.append(feats)
        all_features = torch.stack(features_list, dim=1)
        # all_features: (B, C, F)
        weights = F.softmax(self.fusion_weights, dim=0).view(1, -1, 1)
        # weights: (1, C, 1)
        fused_features = (weights * all_features).sum(dim=1)
        # fused_features: (B, F)
        logits = self.classifier(fused_features)
        # logits: (B, K)
        return {"logits": logits, "features": fused_features, "all_features": all_features}
