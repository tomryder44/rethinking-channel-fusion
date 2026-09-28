import numpy as np
from scipy.signal import hilbert


#
# x: (N, C, L)
#


def apply_fft(x: np.ndarray, norm: str = "ortho"):
    window = np.hanning(x.shape[-1])
    x = x * window
    fft_result = np.fft.rfft(x, axis=-1, norm=norm)
    return np.abs(fft_result)


def apply_log_power(x: np.ndarray, norm: str = "ortho", eps: float = 1e-8):
    window = np.hanning(x.shape[-1])
    x = x * window
    fft_result = np.fft.rfft(x, axis=-1, norm=norm)
    power = np.abs(fft_result) ** 2
    return np.log(power + eps)


def apply_diff(x: np.ndarray):
    return np.diff(x, axis=-1)


def apply_hilbert(x: np.ndarray):
    analytic = hilbert(x, axis=-1)
    return np.abs(analytic)


# def apply_instfreq(x: np.ndarray):
#     analytic = hilbert(x, axis=-1)
#     phase = np.unwrap(np.angle(analytic), axis=-1)
#     return np.diff(phase, axis=-1)
#
#
# def apply_cepstrum(x: np.ndarray):
#     spectrum = np.fft.rfft(x, axis=-1)
#     log_mag = np.log(np.abs(spectrum) + 1e-8)
#     return np.fft.irfft(log_mag, n=x.shape[-1], axis=-1)
#
#
# def apply_autocorr(x: np.ndarray, max_lag: int = None):
#     N = x.shape[-1]
#     max_lag = max_lag or N
#     fft = np.fft.rfft(x, n=2*N, axis=-1)
#     acf = np.fft.irfft(np.abs(fft)**2, axis=-1)[..., :max_lag]
#     return acf / (acf[..., :1] + 1e-8)


# def apply_phase(x: np.ndarray, norm: str = "ortho"):
#     fft_result = np.fft.rfft(x, axis=-1, norm=norm)
#     return np.angle(fft_result)
#
#
# def apply_power(x: np.ndarray, norm: str = "ortho"):
#     fft_result = np.fft.rfft(x, axis=-1, norm=norm)
#     return np.abs(fft_result) ** 2
#
#
# def apply_log(x: np.ndarray):
#     return np.log(np.abs(x) + 1e-8)