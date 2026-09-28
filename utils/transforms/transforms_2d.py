import numpy as np


#
# x: (N, C, L)
# returns: (N, C, F, T)
#


def apply_stft(
    x: np.ndarray,
    norm: str = "ortho",
):

    if x.shape[-1] > 300:
        n_fft = 128
    # elif x.shape[-1] > 100:
    #     n_fft = 64
    else:
        n_fft = 32
    hop_length = n_fft // 4

    N, C, L = x.shape
    window = np.hanning(n_fft)

    T = 1 + (L - n_fft) // hop_length
    F = n_fft // 2 + 1

    output = np.zeros((N, C, F, T))

    for t in range(T):
        start = t * hop_length
        frame = x[:, :, start:start + n_fft] * window
        fft = np.fft.rfft(frame, axis=-1, norm=norm)
        output[:, :, :, t] = np.abs(fft)

    return output


def apply_log_power_stft(x: np.ndarray, norm: str = "ortho", eps: float = 1e-8):
    S = apply_stft(x, norm=norm)
    return np.log(S**2 + eps)