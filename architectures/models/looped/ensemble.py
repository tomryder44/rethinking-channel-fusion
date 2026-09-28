import torch
import torch.nn as nn
import torch.nn.functional as F


class EarlyFusionEnsemble(nn.Module):

    def __init__(self, architecture_class, num_classes, num_channels, hyperparameters):
        super().__init__()
        num_models = hyperparameters["num_models"]
        if num_models == "C":
            num_models = num_channels
        self.num_models = num_models
        num_filters = hyperparameters["num_filters"]
        self.encoders = nn.ModuleList([architecture_class(num_channels, num_filters) for _ in range(num_models)])
        self.classifiers = nn.ModuleList([nn.Linear(num_filters, num_classes) for _ in range(num_models)])

    def forward(self, x):
        # x: (B, C, L)
        features_list = []
        logits_list = []
        for i in range(self.num_models):
            feats = self.encoders[i](x)
            # feats: (B, F)
            features_list.append(feats)
            logits = self.classifiers[i](feats)
            # logits: (B, K)
            logits_list.append(logits)
        all_features = torch.stack(features_list, dim=1)
        # all_features: (B, N, F)
        all_logits = torch.stack(logits_list, dim=1)
        # all_logits: (B, N, K)
        all_probs = F.softmax(all_logits, dim=2)
        # all_probs: (B, N, K)
        fused_probs = all_probs.mean(dim=1)
        # fused_probs: (B, K)
        return {"probs": fused_probs, "all_probs": all_probs, "all_logits": all_logits, "all_features": all_features}
