import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.signal import stft


class TimeSeriesDataset(Dataset):

    def __init__(self, data: np.ndarray, labels: np.ndarray, domain_labels: np.ndarray, device: torch.device):
        self.labels = torch.from_numpy(labels).long().to(device)
        self.data = torch.from_numpy(data).float().to(device)
        self.domain_labels = torch.from_numpy(domain_labels).long().to(device)
        self.num_classes = len(np.unique(labels))
        self.num_channels = data.shape[1]

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, i):
        return {"sequence": self.data[i], "label": self.labels[i], "domain": self.domain_labels[i], "index": i}


class STFTDataset(Dataset):

    def __init__(self, data, labels, domain_labels, device, nperseg_k=None):
        N, C, L = data.shape
        if nperseg_k is None:
            nperseg = 128 if L > 250 else 32
        else:
            nperseg = int(L * nperseg_k)  # paper: window_size * k
        noverlap = nperseg // 2
        # Single STFT call over all (N*C) signals
        data_2d = data.reshape(N * C, L)
        _, _, Zxx = stft(data_2d, nperseg=nperseg, noverlap=noverlap, window='hann')
        # Zxx: (N*C, freq_bins, time_bins) -> (N, C, freq_bins, time_bins)
        Zxx = Zxx.reshape(N, C, Zxx.shape[-2], Zxx.shape[-1])
        self.data = torch.from_numpy(data).float().to(device)
        self.magnitude = torch.from_numpy(np.abs(Zxx)).float().to(device)
        self.phase = torch.from_numpy(np.angle(Zxx)).float().to(device)
        self.labels = torch.from_numpy(labels).long().to(device)
        self.domain_labels = torch.from_numpy(domain_labels).long().to(device)
        self.num_classes = len(np.unique(labels))
        self.num_channels = C

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return {"sequence": (self.magnitude[i], self.phase[i]), "label": self.labels[i], "domain": self.domain_labels[i]}
