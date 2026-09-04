"""无载波幅相（CAP）解调核心。

移植自 MATLAB 函数：
  - CAPmatch_filter.m
  - mdb_match_filter.m
"""
from typing import Optional, Tuple

import numpy as np
from scipy.signal import convolve, upfirdn

from constellation import average_power, demodulate, load_constellation


def capmatch_filter(
    rx_signal: np.ndarray,
    gt: np.ndarray,
    t: np.ndarray,
    fc: float,
    taps: int,
    upsampleno: int,
    offsetsample: int = 0,
) -> np.ndarray:
    """单带 CAP 匹配滤波 / 下变频器（移植自 CAPmatch_filter.m）。

    参数
    ----------
    rx_signal : np.ndarray
        实数接收到的多带或单带 CAP 波形。
    gt : np.ndarray
        基带成形滤波器，长度 == taps。
    t : np.ndarray
        与滤波器抽头对应的时间向量。
    fc : float
        目标 CAP 子带的中心频率。
    taps : int
        滤波器长度（奇数）。
    upsampleno : int
        上采样因子。
    offsetsample : int
        下采样相位偏移。

    返回
    -------
    complex_data : np.ndarray
        匹配滤波后、符号速率下的复数符号。
    """
    rx_signal = np.asarray(rx_signal).flatten()
    gtI = gt * np.cos(2 * np.pi * fc * t)
    gtQ = gt * np.sin(2 * np.pi * fc * t)

    half = (taps - 1) // 2
    data_ext = np.concatenate([rx_signal[-half:], rx_signal, rx_signal[:half]])

    I = convolve(data_ext, gtI, mode="full")
    Q = convolve(data_ext, gtQ, mode="full")

    DataCap = I[taps - 1 : -(taps - 1)] + 1j * Q[taps - 1 : -(taps - 1)]

    # 下采样到符号速率
    received = DataCap[offsetsample::upsampleno]
    received = received - np.mean(received)
    received = received / np.sqrt(np.mean(np.abs(received) ** 2))
    return received


def mdb_match_filter(
    rx_signal: np.ndarray,
    gt: np.ndarray,
    t: np.ndarray,
    fc: float,
    upsampleno: int,
    offsetsample: int = 0,
) -> np.ndarray:
    """双二进制 CAP 匹配滤波器（移植自 mdb_match_filter.m）。

    先下混频，再施加基带滤波。
    """
    rx_signal = np.asarray(rx_signal).flatten()
    down_data = rx_signal * np.cos(2 * np.pi * fc * t) - 1j * rx_signal * np.sin(2 * np.pi * fc * t)
    band_data = convolve(down_data, gt, mode="same")
    band_data = band_data[offsetsample::upsampleno]
    band_data = band_data - np.mean(band_data)
    band_data = band_data / np.sqrt(np.mean(np.abs(band_data) ** 2))
    return band_data


def demodulate_with_ber(
    rx_symbols: np.ndarray,
    order: int,
    constellation: str,
    origin_data: np.ndarray,
    skip_head: int = 0,
    skip_tail: int = 0,
) -> Tuple[np.ndarray, float, float]:
    """对接收符号进行解调并计算误符号率/误码率。

    参数
    ----------
    rx_symbols : np.ndarray
        接收到的复数符号。
    order : int
        星座阶数。
    constellation : str
        "APSK" 或 "QAM"。
    origin_data : np.ndarray
        原始发送的十进制符号。
    skip_head, skip_tail : int
        在序列头部/尾部丢弃的符号数（滤除滤波器瞬态）。

    返回
    -------
    decisions : np.ndarray
        解调得到的十进制符号。
    ser : float
        误符号率。
    ber : float
        误码率（假设每符号 log2(M) 比特）。
    """
    rx_symbols = np.asarray(rx_symbols).flatten()
    origin_data = np.asarray(origin_data).flatten()
    cons = load_constellation(order, constellation)
    avp = float(np.sqrt(np.mean(np.abs(cons) ** 2)))
    rx_symbols = rx_symbols / np.sqrt(np.mean(np.abs(rx_symbols) ** 2)) * avp

    decisions = demodulate(rx_symbols, order, constellation)
    valid = slice(skip_head, len(decisions) - skip_tail if skip_tail else None)
    dec_valid = decisions[valid]
    tx_valid = origin_data[valid]

    ser = float(np.mean(dec_valid != tx_valid))
    bits_per_sym = int(np.log2(order))
    ber = float(np.sum(dec_valid != tx_valid) * bits_per_sym / (len(tx_valid) * bits_per_sym))
    return decisions, ser, ber
