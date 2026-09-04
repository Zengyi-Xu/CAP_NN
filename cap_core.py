"""无载波幅相（CAP）调制核心。

移植自 MATLAB 函数：
  - CAPmod.m
  - cap_gen.m
  - shaping_fildes.m
  - Pulse_shaping_ZY.m / Gen_CAP_filters_ZY.m

实现单带与多带 CAP 发射机。
"""
from typing import List, Optional, Tuple, Union

import numpy as np
from scipy.signal import convolve, resample_poly, upfirdn


def srrc_filter(rolloff: float, span: int, sps: int) -> np.ndarray:
    """平方根升余弦（SRRC）滤波器。

    等价于 MATLAB ``rcosdesign(rolloff, span, sps, 'sqrt')``。
    """
    n_taps = span * sps + 1
    t = np.arange(n_taps) - n_taps // 2
    t = t.astype(float)
    h = np.zeros(n_taps, dtype=float)
    for i, ti in enumerate(t):
        ti_norm = ti / sps
        if np.isclose(ti_norm, 0.0):
            h[i] = 1.0 - rolloff + 4 * rolloff / np.pi
        elif np.isclose(np.abs(4 * rolloff * ti_norm), 1.0):
            h[i] = (rolloff / np.sqrt(2)) * (
                (1 + 2 / np.pi) * np.sin(np.pi / (4 * rolloff))
                + (1 - 2 / np.pi) * np.cos(np.pi / (4 * rolloff))
            )
        else:
            num = np.sin(np.pi * ti_norm * (1 - rolloff)) + 4 * rolloff * ti_norm * np.cos(np.pi * ti_norm * (1 + rolloff))
            den = np.pi * ti_norm * (1 - (4 * rolloff * ti_norm) ** 2)
            h[i] = num / den
    # 将能量归一化为 1
    h = h / np.sqrt(np.sum(h ** 2))
    return h


def srrc_filter_full(
    rolloff: float,
    upsamplesymbol: int,
    upsampleno: int,
) -> np.ndarray:
    """按采样率生成全长度 SRRC 滤波器，必要时截断。

    与 MATLAB cap_gen.m / Pulse_shaping_ZY.m 中的 gtr 计算一致。
    """
    t_norm = np.arange(upsamplesymbol, dtype=float) - upsamplesymbol / 2.0
    r = 1.0 / upsampleno
    gtr = np.zeros(upsamplesymbol, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        x = 4 * rolloff * t_norm * r
        gtr1 = np.cos(np.pi * t_norm * (1 + rolloff) * r) + np.sin(
            np.pi * t_norm * (1 - rolloff) * r
        ) / (4 * rolloff * t_norm * r)
        gtr2 = gtr1 / (1 - x ** 2)
        gtr = gtr2 * 4 * rolloff / np.pi

    # 特殊情况：中心抽头
    centre_idx = upsamplesymbol // 2
    gtr[centre_idx] = 1 + rolloff * (4 / np.pi - 1)

    # 特殊情况：x = ±1（|t_norm * r| = 1 / (4*rolloff)）
    special_mask = np.isclose(np.abs(x), 1.0) & (np.arange(upsamplesymbol) != centre_idx)
    gtr[special_mask] = (rolloff / np.sqrt(2)) * (
        (1 + 2 / np.pi) * np.sin(np.pi / (4 * rolloff))
        + (1 - 2 / np.pi) * np.cos(np.pi / (4 * rolloff))
    )
    return gtr


def shaping_filter(rolloff: float, span: int, sps: int, shape: str = "srrc") -> np.ndarray:
    """生成脉冲成形滤波器。

    参数
    ----------
    rolloff : float
        滚降系数。
    span : int
        滤波器跨度（以符号为单位）。
    sps : int
        每符号采样点数。
    shape : str
        "rc"、"srrc" 或 "btn"。

    返回
    -------
    h : np.ndarray
        一维实数滤波器系数。
    """
    shape = shape.lower()
    if shape.startswith("srrc"):
        return srrc_filter(rolloff, span, sps)
    if shape.startswith("rc"):
        # MATLAB rcosdesign(..., 'normal')
        h = srrc_filter(rolloff, span, sps)
        # RC 滤波器可由两个 SRRC 滤波器卷积得到；用 scipy 近似
        h_rc = np.convolve(h, h)
        return h_rc / np.sqrt(np.sum(h_rc ** 2))
    if shape.startswith("btn"):
        # 来自 Paul Haigh 的 CAP 论文的 BTN 滤波器
        delay = span * sps // 2
        t = (np.arange(-delay, delay + 1)) / sps
        h = (
            np.sinc(t)
            * (2 * np.pi * rolloff * t / np.log(2) * np.sin(np.pi * rolloff * t) + 2 * np.cos(np.pi * rolloff * t) - 1)
            / ((np.pi * rolloff * t / np.log(2)) ** 2 + 1)
        )
        return h / np.sqrt(np.sum(h ** 2))
    raise ValueError(f"未知的脉冲形状: {shape}")


def capmod(
    complex_sym: np.ndarray,
    gt: np.ndarray,
    t: np.ndarray,
    fc: float,
    taps: int,
    upsampleno: int,
) -> np.ndarray:
    """单带 CAP 调制器（移植自 CAPmod.m）。

    参数
    ----------
    complex_sym : np.ndarray
        一维复数符号序列。
    gt : np.ndarray
        实数基带成形滤波器（SRRC），长度 == taps。
    t : np.ndarray
        与滤波器抽头对应的时间向量，以 0 为中心。
    fc : float
        该 CAP 子带的载波频率（Hz）。
    taps : int
        滤波器长度（奇数）。
    upsampleno : int
        上采样因子。

    返回
    -------
    cap_signal : np.ndarray
        实数通带 CAP 波形，功率归一化为 1。
    """
    complex_sym = np.asarray(complex_sym).flatten()
    gtI = gt * np.cos(2 * np.pi * fc * t)
    gtQ = gt * np.sin(2 * np.pi * fc * t)

    Idata = np.zeros(len(complex_sym) * upsampleno, dtype=float)
    Qdata = np.zeros_like(Idata)
    Idata[::upsampleno] = complex_sym.real
    Qdata[::upsampleno] = complex_sym.imag

    # 为吸收滤波器瞬态做循环扩展
    half = (taps - 1) // 2
    Idata_ext = np.concatenate([Idata[-half:], Idata, Idata[:half]])
    Qdata_ext = np.concatenate([Qdata[-half:], Qdata, Qdata[:half]])

    I = convolve(Idata_ext, gtI, mode="full")
    Q = convolve(Qdata_ext, gtQ, mode="full")

    DataCapI = I[taps - 1 : -(taps - 1)]
    DataCapQ = Q[taps - 1 : -(taps - 1)]

    DataCap = DataCapI - DataCapQ
    DataCap = DataCap / np.sqrt(np.mean(DataCap ** 2))
    return DataCap


def capmod_db(
    complex_sym: np.ndarray,
    gt: np.ndarray,
    t: np.ndarray,
    fc: float,
    upsampleno: int,
) -> np.ndarray:
    """双二进制 CAP 使用的调制器变体（移植自 mdb_mod.m）。

    先做脉冲成形，再分别对实部/虚部进行混频搬移。
    """
    complex_sym = np.asarray(complex_sym).flatten()
    up_data = np.zeros(len(complex_sym) * upsampleno, dtype=complex)
    up_data[::upsampleno] = complex_sym
    shaped = convolve(up_data, gt, mode="same")
    data_cap = shaped.real * np.cos(2 * np.pi * fc * t) - shaped.imag * np.sin(2 * np.pi * fc * t)
    data_cap = data_cap / np.sqrt(np.mean(data_cap ** 2))
    return data_cap


def multiband_cap_parameters(
    Rs: float,
    m: int,
    rolloff: float,
    cf: float,
    fs: float,
) -> Tuple[np.ndarray, int, int]:
    """计算多带 CAP 参数（移植自 main_CAP_3band_totalB.m 的设置部分）。

    参数
    ----------
    Rs : float
        总符号速率（symbols/s）。
    m : int
        子带个数。
    rolloff : float
        滚降系数。
    cf : float
        压缩因子（0 < cf < 1）。越小 -> 频谱重叠越多。
    fs : float
        采样率（Hz）。

    返回
    -------
    fc : np.ndarray
        各子带中心频率（Hz），按从高到低排序。
    upsampleno : int
        总上采样因子 = round(m * fs / Rs)。
    taps : int
        滤波器长度 = span * upsampleno + 1。
    """
    Bcap = Rs * (1 + rolloff)
    fc = np.zeros(m)
    for n in range(1, m + 1):
        fc[n - 1] = Bcap / (2 * m) - (n - 1) * (Bcap / m - Bcap * (1 - cf)) / (m - 1)
    upsampleno = int(round(m * fs / Rs))
    span = 8
    taps = span * upsampleno + 1
    return fc, upsampleno, taps


def generate_multiband_cap(
    symbols_per_band: List[np.ndarray],
    Rs: float,
    fs: float,
    rolloff: float = 0.2,
    cf: float = 0.11,
    span: int = 8,
    shape: str = "srrc",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """生成多带 CAP 发射波形。

    参数
    ----------
    symbols_per_band : list of np.ndarray
        每个子带的复数符号序列，长度必须一致。
    Rs : float
        总符号速率（symbols/s）。
    fs : float
        采样率（Hz）。
    rolloff : float
        滚降系数。
    cf : float
        压缩因子。
    span : int
        滤波器跨度（以符号为单位）。
    shape : str
        脉冲形状："srrc"、"rc"、"btn"。

    返回
    -------
    tx_signal : np.ndarray
        功率归一化的多带 CAP 波形。
    fc : np.ndarray
        各子带使用的中心频率。
    gt : np.ndarray
        基带成形滤波器。
    t : np.ndarray
        滤波器时间向量。
    band_signals : np.ndarray
        叠加前的各子带信号，二维数组（num_bands, N）。
    """
    m = len(symbols_per_band)
    if m == 0:
        raise ValueError("至少需要 1 个子带")
    numofsymbols = len(symbols_per_band[0])
    if any(len(s) != numofsymbols for s in symbols_per_band):
        raise ValueError("所有子带的符号数必须相同")

    Bcap = Rs * (1 + rolloff)
    fc = np.zeros(m)
    for n in range(1, m + 1):
        fc[n - 1] = Bcap / (2 * m) - (n - 1) * (Bcap / m - Bcap * (1 - cf)) / (m - 1)

    upsampleno = int(round(m * fs / Rs))
    taps = span * upsampleno + 1
    delay = span * upsampleno // 2
    t = (np.arange(-delay, delay + 1)) / fs

    upsamplesymbol = numofsymbols * upsampleno
    gt_full = srrc_filter_full(rolloff, upsamplesymbol, upsampleno)
    # 截断到 taps 长度，保持居中
    centre = upsamplesymbol // 2
    half = (taps - 1) // 2
    gt = gt_full[centre - half : centre + half + 1]

    band_signals = []
    for sym in symbols_per_band:
        band = capmod(sym, gt, t, fc[len(band_signals)], taps, upsampleno)
        band_signals.append(band)
    band_signals = np.asarray(band_signals)

    tx_signal = band_signals.sum(axis=0)
    tx_signal = tx_signal / np.sqrt(np.mean(tx_signal ** 2))
    return tx_signal, fc, gt, t, band_signals


def generate_singleband_cap(
    symbols: np.ndarray,
    fs: float,
    Rs: float,
    rolloff: float = 0.205,
    subcar: float = 0.5,
    start_freq: float = 0.01,
    taps: int = 35,
    shape: str = "srrc",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """生成单带 CAP 发射波形（移植自 oldcapAPSKTxRx20220406.m）。

    参数
    ----------
    symbols : np.ndarray
        一维复数符号序列。
    fs : float
        采样率（Hz）。
    Rs : float
        符号速率（symbols/s）。
    rolloff : float
        滚降系数。
    subcar : float
        归一化载波位置参数（0.5 表示子带居中）。
    start_freq : float
        额外的低频偏移（归一化单位）。
    taps : int
        滤波器长度（奇数）。
    shape : str
        脉冲形状。

    返回
    -------
    tx_signal : np.ndarray
        功率归一化的 CAP 波形。
    filter_I : np.ndarray
        I 路成形滤波器。
    filter_Q : np.ndarray
        Q 路成形滤波器。
    t : np.ndarray
        全长度滤波器时间向量。
    """
    symbols = np.asarray(symbols).flatten()
    upsampleno = int(round(fs / Rs))
    upsamplesymbol = len(symbols) * upsampleno
    # MATLAB: linspace(1,upsamplesymbol,upsamplesymbol)-1-upsamplesymbol/2
    t_norm = np.arange(upsamplesymbol, dtype=float) - upsamplesymbol / 2.0
    t = t_norm / fs

    # 归一化符号速率 = 1/upsampleno
    r = 1.0 / upsampleno
    gtr = srrc_filter_full(rolloff, upsamplesymbol, upsampleno)

    BW = 1 + rolloff
    subcar1 = BW * subcar + start_freq

    filter_I_full = gtr * np.cos(2 * np.pi * subcar1 * t_norm * r)
    filter_Q_full = gtr * np.sin(2 * np.pi * subcar1 * t_norm * r)

    # 截断到指定 taps 长度，保持居中
    centre = upsamplesymbol // 2
    half = taps // 2
    filter_I = filter_I_full[centre - half : centre + half + 1]
    filter_Q = filter_Q_full[centre - half : centre + half + 1]

    up_data = np.zeros(len(symbols) * upsampleno, dtype=complex)
    up_data[::upsampleno] = symbols

    DataCapI = convolve(up_data.real, filter_I, mode="same")
    DataCapQ = convolve(up_data.imag, filter_Q, mode="same")
    tx_signal = DataCapI - DataCapQ
    tx_signal = tx_signal / np.sqrt(np.mean(tx_signal ** 2))
    return tx_signal, filter_I, filter_Q, t
