import torch
from torch import nn


class CNNLSTMEncoder(nn.Module):

    def __init__(self, num_channels, num_filters):
        super().__init__()
        num_features = num_filters
        self.conv1 = nn.Conv1d(in_channels=num_channels, out_channels=num_features, kernel_size=3, bias=False)
        self.conv2 = nn.Conv1d(in_channels=num_features, out_channels=num_features, kernel_size=3, bias=False)
        self.conv3 = nn.Conv1d(in_channels=num_features, out_channels=num_features, kernel_size=3, bias=False)
        self.bn1 = nn.BatchNorm1d(num_features)
        self.bn2 = nn.BatchNorm1d(num_features)
        self.bn3 = nn.BatchNorm1d(num_features)
        self.relu = nn.LeakyReLU()
        hidden_size = num_features//2
        self.lstm = nn.LSTM(input_size=num_features, hidden_size=hidden_size, num_layers=1,
                            batch_first=True, bidirectional=True)

    def forward(self, x):
        # x: (B, C, L)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        # x: (B, F, L-6)
        x = x.transpose(1, 2)
        _, (h_n, _) = self.lstm(x)
        # h_n: (2, B, hidden_size)
        x = torch.cat([h_n[-2], h_n[-1]], dim=1)
        # x: (B, F)
        return x
