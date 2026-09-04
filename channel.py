"""可见光通信信道模型。

移植自 MATLAB 函数：
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
    """应用 VLC 信道模型，可选 LED 非线性和频率衰落。

    Parameters
    ----------
    data_in : np.ndarray
        输入波形（实数）。
    snr_db : float
        信道之后的目标 SNR（dB）。
    fs_hz : float
        控制指数衰减带宽的参数（MATLAB 代码中名为 Fs，
        即 vlc_channel.m 中的 Fs）。
    factor : float
        衰减因子；越大 -> 带宽越宽 / 衰落越小。
    nonlinear : bool
        是否应用弱 LED 非线性模型。
    vpp : float
        用于非线性缩放的峰峰值电压。

    Returns
    -------
    data_rx : np.ndarray
        加入 AWGN 后的实数接收波形。
    """
    data_tx = np.asarray(data_in, dtype=float).flatten()

    if nonlinear:
        x = data_tx / (np.max(data_tx) - np.min(data_tx)) * 2 * vpp
        # 来自 MATLAB 的弱非线性模型
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

    # 按实测功率添加 AWGN
    sig_power = np.mean(data_tx ** 2)
    noise_power = sig_power / (10 ** (snr_db / 10))
    noise = np.sqrt(noise_power) * np.random.randn(len(data_tx))
    data_rx = data_tx + noise
    return data_rx
