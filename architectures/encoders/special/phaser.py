import torch
from torch import nn


class PhaserEncoder(nn.Module):

    def __init__(self, num_channels, num_filters):
        super().__init__()
        self.num_channels = num_channels
        self.num_sub_features = 3
        c = int(num_filters // 2)
        self.mag_encoder = nn.Conv2d(num_channels, 2*c, kernel_size=(5, 5), padding=2)
        self.phase_encoder = nn.Conv2d(num_channels, 2*c, kernel_size=(5, 5), padding=2)
        self.fusion_encoder = nn.Sequential(nn.Conv2d(4*c, 2*c, kernel_size=5, padding=2), nn.BatchNorm2d(2*c), nn.SiLU())
        self.depth_encoder = nn.Sequential(nn.Conv2d(2*c, 2*c, kernel_size=(5, 5), padding=2), nn.BatchNorm2d(2*c), nn.SiLU(), nn.AdaptiveAvgPool2d((1, None)))
        self.temporal_encoder = nn.Sequential(nn.Conv2d(2*c, 2*c, kernel_size=(1, 3), padding=(0, 1)), nn.BatchNorm2d(2*c), nn.SiLU())
        self.phase_residual = nn.Sequential(nn.Conv2d(2*c, 2*c, kernel_size=1), nn.AdaptiveAvgPool2d((1, None)))
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2*c, num_filters)

    def sub_feature_normalize(self, x):
        B, C, F, T = x.shape
        num_groups = 3
        split_size = F // num_groups
        remainder = F % num_groups
        outputs = []
        start = 0
        for i in range(num_groups):
            end = start + split_size + (1 if i < remainder else 0)
            sub_band = x[:, :, start:end, :]
            mean = sub_band.mean(dim=[2, 3], keepdim=True)
            std = sub_band.std(dim=[2, 3], keepdim=True) + 1e-5
            outputs.append((sub_band - mean) / std)
            start = end
        return torch.cat(outputs, dim=2)

    def forward(self, x):
        magnitude = x[0]  # (B, C, F, L)
        phase = x[1]  # (B, C, F, L)
        mag_features = self.sub_feature_normalize(self.mag_encoder(magnitude))  # (B, 2c, F, L)
        phase_features = self.sub_feature_normalize(self.phase_encoder(phase)) # (B, 2c, F, L)
        fused = self.fusion_encoder(torch.cat([mag_features, phase_features], dim=1))  # (B, 2c, F, L)
        depth_features = self.depth_encoder(fused)  # (B, 2c, 1, L)
        temporal_features = self.temporal_encoder(depth_features)  # (B, 2c, 1, L)
        phase_residual = self.phase_residual(phase_features)  # (B, 2c, 1, L)
        residual_features = temporal_features + phase_residual  # (B, 2c, 1, L)
        features = self.gap(residual_features).squeeze(-1).squeeze(-1)  # (B, 2c)
        features = self.fc(features)  # (B, num_filters)
        return features
