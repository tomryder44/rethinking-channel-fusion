import torch
import torch.nn as nn
import torch.nn.functional as F


class LateFusionModel(nn.Module):

    def __init__(self, encoder_class, num_classes, num_channels, hyperparameters):
        super().__init__()
        num_filters = hyperparameters["num_filters"]
        self.num_channels = num_channels
        self.encoders = nn.ModuleList([encoder_class(1, num_filters) for _ in range(num_channels)])
        self.classifiers = nn.ModuleList([nn.Linear(num_filters, num_classes) for _ in range(num_channels)])

    def forward(self, x):
        # x: (B, C, L)
        features_list = []
        logits_list = []
        for i in range(self.num_channels):
            channel_input = x[:, i, :].unsqueeze(1)
            # channel_input: (B, 1, L)
            feats = self.encoders[i](channel_input)
            # feats: (B, F)
            features_list.append(feats)
            logits = self.classifiers[i](feats)
            # logits: (B, K)
            logits_list.append(logits)
        all_features = torch.stack(features_list, dim=1)
        # all_features: (B, C, F)
        all_logits = torch.stack(logits_list, dim=1)
        # all_logits: (B, C, K)
        all_probs = F.softmax(all_logits, dim=2)
        # all_probs: (B, C, K)
        fused_probs = all_probs.mean(dim=1)
        # fused_probs: (B, K)
        return {"probs": fused_probs, "all_probs": all_probs, "all_logits": all_logits, "all_features": all_features}
