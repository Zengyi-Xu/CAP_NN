"""LMS 与 LMS+Volterra 均衡器。

移植自 MATLAB 函数：
  - LMS_1DownS_Testnan.m
  - LMS_volterra_1DownS_Testnan.m
"""
from typing import Optional, Tuple

import numpy as np


def lms_equalizer(
    rxdata: np.ndarray,
    txdata: np.ndarray,
    taps_lms: int,
    mu_lms: float,
    numof_ts: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """符号速率 LMS 线性均衡器。

    Parameters
    ----------
    rxdata : np.ndarray
        接收复数符号（一维）。
    txdata : np.ndarray
        发射复数符号（一维），与 rxdata 对齐。
    taps_lms : int
        LMS 抽头数（奇数）。
    mu_lms : float
        LMS 步长。
    numof_ts : int
        训练符号数。

    Returns
    -------
    k : np.ndarray
        均衡输出，与 rxdata 等长（首尾补零填充）。
    y : np.ndarray
        训练阶段的输出。
    E : np.ndarray
        训练阶段的误差。
    W : np.ndarray
        收敛后的 LMS 抽头权重。
    """
    rxdata = np.asarray(rxdata).flatten()
    txdata = np.asarray(txdata).flatten()

    rxdata = rxdata / np.sqrt(np.mean(np.abs(rxdata) ** 2))
    txdata = txdata / np.sqrt(np.mean(np.abs(txdata) ** 2))

    x = rxdata[:numof_ts]
    d = txdata[:numof_ts]

    compensation = taps_lms
    half = (taps_lms - 1) // 2
    W = np.zeros(taps_lms, dtype=complex)

    ntr = len(x)
    y = np.zeros(ntr, dtype=complex)
    E = np.zeros(ntr, dtype=complex)

    nn = 0
    n = compensation - 1
    while n < ntr:
        # MATLAB 索引从 1 开始；中心抽头对齐到 n-compensation/2+1/2
        idx = n - half
        X_lms = x[idx - half : idx + half + 1][::-1]
        y[nn] = np.dot(W, X_lms)
        e = d[idx] - y[nn]
        W = W + mu_lms * e * np.conj(X_lms)
        E[nn] = e
        n += 1
        nn += 1

    # 应用到整个序列
    k = np.zeros(len(rxdata), dtype=complex)
    n = compensation - 1
    mm = 0
    while n < len(rxdata):
        idx = n - half
        X_lms = rxdata[idx - half : idx + half + 1][::-1]
        k[mm] = np.dot(W, X_lms)
        n += 1
        mm += 1

    # 首尾补零以保持长度
    head = (compensation - 1) // 2
    tail = len(rxdata) - mm
    k = np.concatenate([rxdata[:head], k[:mm], rxdata[-tail:]]) if tail > 0 else np.concatenate([rxdata[:head], k[:mm]])
    k = k / np.sqrt(np.mean(np.abs(k) ** 2))
    return k, y, E, W


def lms_volterra_equalizer(
    rxdata: np.ndarray,
    txdata: np.ndarray,
    taps_lms: int,
    mu_lms: float,
    taps_volterra: int,
    mu_volterra: float,
    numof_ts: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """符号速率 LMS 线性 + 二阶 Volterra 均衡器。

    Parameters
    ----------
    rxdata, txdata : np.ndarray
        接收与发射复数符号。
    taps_lms : int
        线性 LMS 抽头数。
    mu_lms : float
        线性步长。
    taps_volterra : int
        Volterra 记忆长度。
    mu_volterra : float
        Volterra 步长。
    numof_ts : int
        训练长度。

    Returns
    -------
    k : np.ndarray
        均衡输出。
    y, E, W : 同 lms_equalizer。
    V : np.ndarray
        收敛后的 Volterra 核（上三角矩阵）。
    X_v : np.ndarray
        最后一个 Volterra 输入矩阵（诊断用）。
    """
    rxdata = np.asarray(rxdata).flatten()
    txdata = np.asarray(txdata).flatten()

    rxdata = rxdata / np.sqrt(np.mean(np.abs(rxdata) ** 2))
    txdata = txdata / np.sqrt(np.mean(np.abs(txdata) ** 2))

    x = rxdata[:numof_ts]
    d = txdata[:numof_ts]

    compensation = max(taps_lms, taps_volterra)
    half_lms = (taps_lms - 1) // 2
    half_vol = (taps_volterra - 1) // 2

    W = np.zeros(taps_lms, dtype=complex)
    delta = 0.0000001
    V = delta * np.eye(taps_volterra, dtype=complex)

    ntr = len(x)
    y = np.zeros(ntr, dtype=complex)
    E = np.zeros(ntr, dtype=complex)

    nn = 0
    n = compensation - 1
    X_v = np.zeros((taps_volterra, taps_volterra), dtype=complex)
    while n < ntr:
        idx_lms = n - half_lms
        X_lms = x[idx_lms - half_lms : idx_lms + half_lms + 1][::-1]

        idx_vol = n - half_vol
        X_vol = x[idx_vol - half_vol : idx_vol + half_vol + 1][::-1]
        X_v = np.triu(np.outer(X_vol, np.conj(X_vol)))
        VV = np.sum(V * X_v)

        y[nn] = np.dot(W, X_lms) + VV
        e = d[idx_lms] - y[nn]
        W = W + mu_lms * e * np.conj(X_lms)
        V = V + mu_volterra * e * np.conj(X_v)
        E[nn] = e
        n += 1
        nn += 1

    # 应用到整个序列
    k = np.zeros(len(rxdata), dtype=complex)
    n = compensation - 1
    mm = 0
    while n < len(rxdata):
        idx_lms = n - half_lms
        X_lms = rxdata[idx_lms - half_lms : idx_lms + half_lms + 1][::-1]

        idx_vol = n - half_vol
        X_vol = rxdata[idx_vol - half_vol : idx_vol + half_vol + 1][::-1]
        R_v = np.triu(np.outer(X_vol, np.conj(X_vol)))
        R_vv = np.sum(V * R_v)

        k[mm] = np.dot(W, X_lms) + R_vv
        n += 1
        mm += 1

    head = (compensation - 1) // 2
    tail = len(rxdata) - mm
    if tail > 0:
        k = np.concatenate([rxdata[:head], k[:mm], rxdata[-tail:]])
    else:
        k = np.concatenate([rxdata[:head], k[:mm]])
    k = k / np.sqrt(np.mean(np.abs(k) ** 2))
    return k, y, E, W, V, X_v
