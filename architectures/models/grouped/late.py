import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class LateFusionModel(nn.Module):

    def __init__(self, architecture_class, num_classes, num_channels, hyperparameters):
        super().__init__()
        num_filters = hyperparameters["num_filters"]
        self.encoder = architecture_class(num_channels, num_filters)
        self.num_channels = num_channels
        self.head_weight = nn.Parameter(torch.empty(num_channels, num_classes, num_filters))
        self.head_bias = nn.Parameter(torch.empty(num_channels, num_classes))
        self.reset_head_parameters()

    def reset_head_parameters(self):
        for c in range(self.num_channels):
            nn.init.kaiming_uniform_(self.head_weight[c], a=math.sqrt(5))
            fan_in = self.head_weight.size(2)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.head_bias[c], -bound, bound)

    def forward(self, x):
        # x: (B, C, L)
        all_features = self.encoder(x)
        # all_features: (B, C, F)
        all_logits = torch.einsum("bcf,ckf->bck", all_features, self.head_weight)  # (B, C, F) x (C, K, F) -> (B, C, K)
        # all_logits: (B, C, K)
        all_logits = all_logits + self.head_bias
        all_probs = F.softmax(all_logits, dim=2)
        # all_probs: (B, C, K)
        fused_probs = all_probs.mean(dim=1)
        # fused_probs: (B, K)
        return {"probs": fused_probs, "all_probs": all_probs, "all_logits": all_logits, "all_features": all_features}


class LateFusionWeightModel(LateFusionModel):

    def forward(self, x, weights=None):
        # x: (B, C, L)
        if weights is None:
            return super().forward(x)
        all_features = self.encoder(x)
        # all_features: (B, C, F)
        all_logits = torch.einsum("bcf,ckf->bck", all_features, self.head_weight)  # (B, C, F) x (C, K, F) -> (B, C, K)
        # all_logits: (B, C, K)
        all_logits = all_logits + self.head_bias
        all_probs = F.softmax(all_logits, dim=2)
        # all_probs: (B, C, K)
        fused_probs = (weights.view(1, -1, 1) * all_probs).sum(dim=1)
        # fused_probs: (B, K)
        return {"probs": fused_probs, "all_probs": all_probs, "all_logits": all_logits, "all_features": all_features}
