import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
from scipy.signal import hilbert


class Augmenter(nn.Module):

    def __init__(self):
        super(Augmenter, self).__init__()

        self.min_amplitude_scaling = 0.75
        self.max_amplitude_scaling = 1.25
        self.min_dc_shift = -0.25
        self.max_dc_shift = 0.25
        self.max_gaussian_noise_std = 0.25
        self.max_channel_dropout_rate = 0.2
        self.max_temporal_cutout_size = 0.2

    def channel_dropout(self, x):
        channel_dropout_rate = torch.rand(1) * self.max_channel_dropout_rate
        channel_probs = torch.rand(x.shape[1])
        channels_to_drop = channel_probs < channel_dropout_rate
        x[:, channels_to_drop, :] = 0.
        return x

    def gaussian_noise(self, x):
        gaussian_noise_std = torch.rand(1, device=x.device) * 0.1
        noise = torch.randn_like(x) * gaussian_noise_std
        return x + noise

    def temporal_cutout(self, x):
        temporal_cutout_size = torch.rand(1) * self.max_temporal_cutout_size
        num_samples_to_mask = int(x.shape[2] * temporal_cutout_size)
        start_index = torch.randint(0, x.shape[2] - num_samples_to_mask, (1,))
        x[:, :, start_index:start_index + num_samples_to_mask] = 0.
        return x

    def amplitude_scaling(self, x):
        amplitude_scaling_factor = torch.rand(1, device=x.device) * \
                                   (self.max_amplitude_scaling - self.min_amplitude_scaling) \
                                   + self.min_amplitude_scaling
        return x * amplitude_scaling_factor

    def amplitude_shift(self, x):
        dc_shift = torch.rand(1, device=x.device) * (self.max_dc_shift - self.min_dc_shift) + self.min_dc_shift
        return x + dc_shift

    def forward(self, x, visualise=False):
        x_orig = x.clone()
        # x_aug = self.channel_dropout(x)
        # x_aug = self.temporal_cutout(x_aug)
        x_aug = self.amplitude_scaling(x)
        x_aug = self.amplitude_shift(x_aug)
        # x_aug = self.gaussian_noise(x_aug)
        return x_aug


def compute_hilbert_transform(x, y, d):
    # hilbert() operates on last axis by default — vectorized over N and C
    x_hilbert = np.imag(hilbert(x))
    x_augmented = np.concatenate([x, x_hilbert], axis=0)
    y_augmented = np.concatenate([y, y], axis=0)
    d_augmented = np.concatenate([d, d], axis=0)
    return x_augmented, y_augmented, d_augmented
