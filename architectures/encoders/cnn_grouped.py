from torch import nn


class CNNGroupedEncoder(nn.Module):

    def __init__(self, num_channels, num_filters):
        super().__init__()
        self.num_channels = num_channels
        self.num_filters = num_filters
        self.conv1 = nn.Conv1d(in_channels=num_channels, out_channels=num_filters * num_channels,
                               kernel_size=3, groups=num_channels, bias=False)
        self.conv2 = nn.Conv1d(in_channels=num_filters * num_channels, out_channels=num_filters * num_channels,
                               kernel_size=3, groups=num_channels, bias=False)
        self.conv3 = nn.Conv1d(in_channels=num_filters * num_channels, out_channels=num_filters * num_channels,
                               kernel_size=3, groups=num_channels, bias=False)
        self.bn1 = nn.BatchNorm1d(self.num_filters*self.num_channels)
        self.bn2 = nn.BatchNorm1d(self.num_filters*self.num_channels)
        self.bn3 = nn.BatchNorm1d(self.num_filters*self.num_channels)
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        # x: (B, C, L)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        # x: (B, F*C, L-6)
        x = x.mean(dim=2)
        # x: (B, F*C)
        x = x.reshape(x.size(0), self.num_channels, self.num_filters)
        # x: (B, C, F)
        return x
