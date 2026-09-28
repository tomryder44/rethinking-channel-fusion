import torch
from torch import nn
import math


class TransformerEncoder(nn.Module):

    def __init__(self, num_channels, num_filters):
        super().__init__()
        self.d_model = num_filters
        n_head = 2
        n_layers = 2
        self.embedding = nn.Linear(num_channels, self.d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=n_head, dim_feedforward=2*self.d_model,
                                                   dropout=0.1, batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers, norm=nn.LayerNorm(self.d_model),
                                                 enable_nested_tensor=False)

    @staticmethod
    def sinusoidal_encoding(seq_len, d_model, device):
        position = torch.arange(seq_len, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, device=device) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, seq_len, d_model, device=device)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, x):
        # x: (B, C, L)
        x = x.permute(0, 2, 1)
        x = self.embedding(x)
        # x: (B, L, F)
        x = x + self.sinusoidal_encoding(x.size(1), self.d_model, x.device)
        x = self.transformer(x)
        # x: (B, L, F)
        x = x.mean(dim=1)
        # x: (B, F)
        return x
