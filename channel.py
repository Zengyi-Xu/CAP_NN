"""Visible-light communication channel model.

Ports MATLAB function:
  - vlc_channel.m
"""
from typing import Optional

import numpy as np


def vlc_channel(
    data_in: np.ndarray,
    snr_db: float,
    fs_hz: float,
    factor: float,
    nonlinear: bool = False,
    vpp: float = 1.2,
) -> np.ndarray:
    """Apply VLC channel model with optional LED nonlinearity and frequency fading.

    Parameters
    ----------
    data_in : np.ndarray
        Input waveform (real).
    snr_db : float
        Target SNR in dB after channel.
    fs_hz : float
        Parameter controlling the exponential decay bandwidth in MATLAB code
        (named Fs in vlc_channel.m).
    factor : float
        Decay factor; larger -> wider bandwidth / less fading.
    nonlinear : bool
        Whether to apply the weak LED nonlinearity model.
    vpp : float
        Peak-to-peak voltage used for nonlinearity scaling.

    Returns
    -------
    data_rx : np.ndarray
        Real received waveform with AWGN added.
    """
    data_tx = np.asarray(data_in, dtype=float).flatten()

    if nonlinear:
        x = data_tx / (np.max(data_tx) - np.min(data_tx)) * 2 * vpp
        # Weak NL model from MATLAB
        data_tx = 4.412 / (1.0 + np.exp(-1.07 * x)) - 2.206
        data_tx = data_tx / np.sqrt(np.mean(data_tx ** 2))

    N = len(data_tx)
    n = np.arange(1, N // 2 + 1)
    df = 2 * fs_hz / N
    fsn = df * n
    ch1 = np.exp(-fsn / factor)
    ch2 = ch1[::-1]
    ch = np.concatenate([ch1, ch2])

    data_ch_fft = np.fft.fft(data_tx)
    data_ch_after = data_ch_fft * ch
    data_ch_ifft = np.real(np.fft.ifft(data_ch_after))
    data_tx = data_ch_ifft - np.mean(data_ch_ifft)

    # AWGN with measured power
    sig_power = np.mean(data_tx ** 2)
    noise_power = sig_power / (10 ** (snr_db / 10))
    noise = np.sqrt(noise_power) * np.random.randn(len(data_tx))
    data_rx = data_tx + noise
    return data_rx
