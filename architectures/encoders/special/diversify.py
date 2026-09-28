from torch import nn


class DiversifyEncoder(nn.Module):

    def __init__(self, num_channels, num_filters):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=num_channels, out_channels=num_filters, kernel_size=3, bias=False)
        self.conv2 = nn.Conv1d(in_channels=num_filters, out_channels=num_filters, kernel_size=3, bias=False)
        self.conv3 = nn.Conv1d(in_channels=num_filters, out_channels=num_filters, kernel_size=3, bias=False)
        self.bn1 = nn.BatchNorm1d(num_filters)
        self.bn2 = nn.BatchNorm1d(num_filters)
        self.bn3 = nn.BatchNorm1d(num_filters)
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        # x: (B, C, L)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        # x: (B, F, L-6)
        return x.flatten(1)  # (B, F*(L-6))
