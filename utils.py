"""通用工具函数：文件 I/O、同步、重采样、绘图等."""
import os
import numpy as np
import scipy.signal as sg
import scipy.io as sio
from scipy.interpolate import CubicSpline
from pathlib import Path
import matplotlib
import config

# 在 IPython/Spyder 环境中，提前用 magic 设置 inline 后端，确保图像显示在 Plots 窗口
if config.PLOT_SHOW:
    try:
        from IPython import get_ipython
        ip = get_ipython()
        if ip is not None:
            ip.run_line_magic("matplotlib", "inline")
    except Exception:
        pass

import matplotlib.pyplot as plt


# -----------------------------------------------------------------------------
# 文件 I/O
# -----------------------------------------------------------------------------
def load_txt(path: Path, dtype=float) -> np.ndarray:
    """读取文本文件为 numpy 数组."""
    if isinstance(path, str):
        path = Path(path)
    return np.loadtxt(path, dtype=dtype)


def save_txt(path: Path, data: np.ndarray, fmt="%.6f") -> None:
    """保存 numpy 数组到文本文件."""
    if isinstance(path, str):
        path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, data, fmt=fmt)


def load_mat(path: Path, squeeze=True) -> dict:
    """读取 .mat 文件，返回 dict."""
    if isinstance(path, str):
        path = Path(path)
    return sio.loadmat(path, squeeze_me=squeeze)


def save_mat(path: Path, **kwargs) -> None:
    """保存变量到 .mat 文件."""
    if isinstance(path, str):
        path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sio.savemat(path, kwargs)


# -----------------------------------------------------------------------------
# 波形处理
# -----------------------------------------------------------------------------
def add_dummy(x: np.ndarray, base: int = 64) -> tuple:
    """在波形末尾补零使其长度为 base 的整数倍.

    Returns:
        with_dummy: 补零后的数组
        dummies: 补的零数组
    """
    x = np.asarray(x).ravel()
    if x.ndim != 1:
        raise ValueError("Input x must be a 1-D array")
    n_dummy = (-len(x)) % base
    if n_dummy == 0:
        return x.copy(), np.array([])
    dummies = np.zeros(n_dummy, dtype=x.dtype)
    return np.concatenate([x, dummies]), dummies


def resample_signal(x: np.ndarray, fs_target: float, fs_source: float) -> np.ndarray:
    """使用有理数重采样把 x 从 fs_source 重采样到 fs_target."""
    x = np.asarray(x).ravel()
    if fs_source == fs_target:
        return x
    # 寻找近似整数比
    from fractions import Fraction
    frac = fs_target / fs_source
    f = Fraction(frac).limit_denominator(1000)
    up, down = f.numerator, f.denominator
    return sg.resample_poly(x, up, down)


def sync_waveform(rx: np.ndarray, tx: np.ndarray, h: int = 1) -> np.ndarray:
    """用 FFT 加速互相关对接收波形做符号同步，取与 tx 等长片段.

    Args:
        rx: 接收波形 (1-D)
        tx: 发送波形 (1-D)
        h: 同步偏移修正量

    Returns:
        同步后的接收波形，长度与 tx 相同
    """
    rx = np.asarray(rx).ravel()
    tx = np.asarray(tx).ravel()
    # 使用 FFT 加速（避免 np.correlate 在大长度时 O(N^2) 直接计算）
    corr = sg.correlate(np.real(rx), np.real(tx), mode="full", method="fft")
    lags = np.arange(-(len(tx) - 1), len(rx))
    offset = np.argmax(np.abs(corr))
    start = lags[offset] + h
    end = start + len(tx)
    if end > len(rx):
        # 如果超出范围，找次大值
        corr[offset] = 0
        offset = np.argmax(np.abs(corr))
        start = lags[offset] + h
        end = start + len(tx)
    if start < 0 or end > len(rx):
        raise ValueError(
            f"Synchronization failed: start={start}, end={end}, rx_len={len(rx)}"
        )
    return rx[start:end]


def center_normalize(x: np.ndarray) -> np.ndarray:
    """最大-最小值归一化并居中 (与 MATLAB 代码一致)."""
    x = np.asarray(x).ravel()
    x = x / (np.max(x) - np.min(x))
    x = x - (np.abs(np.max(x)) - np.abs(np.min(x))) / 2
    return x


# -----------------------------------------------------------------------------
# 预均衡辅助
# -----------------------------------------------------------------------------
def apply_hardware_preEQ(dataifft1: np.ndarray,
                         fsamp: float,
                         upsampleno: int,
                         th7_file: Path) -> np.ndarray:
    """应用硬件预均衡 (pre_equ_flag=3)."""
    f_hardware = load_txt(th7_file).ravel()
    f_hardware_use = np.concatenate([np.flip(f_hardware), f_hardware])

    dataout = np.asarray(dataifft1).ravel()
    diff = 53 if upsampleno != 1 else 0

    idx_src = np.arange(len(f_hardware_use))
    idx_dst = np.linspace(0, len(f_hardware_use) - 1,
                          int(len(dataout) / upsampleno) + diff * 2)
    cs = CubicSpline(idx_src, f_hardware_use)
    f_hardware_use = cs(idx_dst)

    pad_len = int(len(dataout) / 2 - len(dataout) / 2 / upsampleno - diff)
    f_pad = np.ones(pad_len) * np.min(f_hardware)
    f_hardware_use = np.concatenate([f_pad, f_hardware_use, f_pad])

    data1_hardware = np.real(np.fft.ifft(
        np.fft.fftshift(np.fft.fftshift(np.fft.fft(dataout)) * f_hardware_use)
    ))
    dataout = data1_hardware / np.max(np.abs(data1_hardware))
    return dataout


# -----------------------------------------------------------------------------
# 绘图辅助
# -----------------------------------------------------------------------------
def _should_show(show: bool = None) -> bool:
    """根据传入参数或 config 决定是否显示图像."""
    if show is None:
        return config.PLOT_SHOW
    return show


def _finalize_figure(fig, out: Path, show: bool = None) -> None:
    """保存并/或显示图像.

    在 Spyder 中显示时先不关闭 figure，否则 Plots 窗口会清空。
    """
    show = _should_show(show)
    if out is not None and config.PLOT_SAVE:
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=config.PLOT_DPI, bbox_inches="tight")
    if show:
        # 在 Spyder 的 Inline 后端中会显示到 Plots 窗口
        plt.show()
    else:
        plt.close(fig)


# -----------------------------------------------------------------------------
# 基础绘图
# -----------------------------------------------------------------------------
def plot_time_waveform(t: np.ndarray, sig: np.ndarray, title: str, out: Path = None,
                       show: bool = None) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t, sig, "b.-")
    ax.set_title(title)
    ax.set_xlabel("Sample")
    ax.set_ylabel("Amplitude")
    ax.grid(True)
    _finalize_figure(fig, out, show)


def plot_spectrum(sig: np.ndarray, fs: float, title: str, out: Path = None,
                  show: bool = None) -> None:
    n = len(sig)
    freqs = np.fft.fftshift(np.fft.fftfreq(n, d=1.0 / fs))
    spec = 20 * np.log10(np.abs(np.fft.fftshift(np.fft.fft(sig))) + 1e-12)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(freqs, spec, "b-")
    ax.set_title(title)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude (dB)")
    ax.grid(True)
    _finalize_figure(fig, out, show)


def plot_constellation(iq: np.ndarray, title: str, out: Path = None,
                       show: bool = None) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(iq.real, iq.imag, "b.", alpha=0.3)
    ax.set_title(title)
    ax.set_xlabel("I")
    ax.set_ylabel("Q")
    ax.grid(True)
    ax.axis("equal")
    _finalize_figure(fig, out, show)


def plot_snrs(est: np.ndarray, real: np.ndarray, out: Path = None,
              show: bool = None) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(est, "b", label="Est-SNR", marker="o")
    ax.plot(real, "r", label="TestReal-SNR", marker="x")
    ax.set_title("Estimated vs Recovered SNR")
    ax.set_xlabel("Subcarrier")
    ax.set_ylabel("SNR")
    ax.legend()
    ax.grid(True)
    _finalize_figure(fig, out, show)


# -----------------------------------------------------------------------------
# 新增：DMT 专用绘图
# -----------------------------------------------------------------------------
def plot_dmt_spectrogram(sig: np.ndarray, fs: float, title: str,
                         out: Path = None, show: bool = None) -> None:
    """绘制 DMT 信号时频谱（spectrogram）."""
    sig = np.asarray(sig).ravel()
    nperseg = min(1024, len(sig) // 8)
    noverlap = nperseg // 2
    f, t, Sxx = sg.spectrogram(sig, fs=fs, nperseg=nperseg, noverlap=noverlap,
                               window="hann", scaling="spectrum")
    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.pcolormesh(t * 1e6, f / 1e9, 10 * np.log10(Sxx + 1e-12),
                       shading="gouraud", cmap="jet")
    ax.set_title(title)
    ax.set_xlabel("Time (us)")
    ax.set_ylabel("Frequency (GHz)")
    fig.colorbar(im, ax=ax, label="Power (dB)")
    _finalize_figure(fig, out, show)


def plot_tx_rx_nonlinearity(tx: np.ndarray, rx: np.ndarray,
                            out: Path = None, show: bool = None) -> None:
    """绘制接收信号非线性：横轴为发射波形幅值，纵轴为接收波形幅值."""
    tx = np.asarray(tx).ravel()
    rx = np.asarray(rx).ravel()
    if len(tx) != len(rx):
        raise ValueError("tx and rx must have the same length")

    fig, ax = plt.subplots(figsize=(7, 7))
    # 点太多时用 hexbin，否则 scatter
    if len(tx) > 5000:
        hb = ax.hexbin(tx, rx, gridsize=80, cmap="GnBu", mincnt=1)
        fig.colorbar(hb, ax=ax, label="Density")
    else:
        ax.plot(tx, rx, "b.", alpha=0.2)

    # 理想线性参考线（斜率为当前 tx->rx 最小二乘增益）
    if np.any(tx):
        gain = np.sum(tx * rx) / np.sum(tx ** 2)
        t = np.linspace(tx.min(), tx.max(), 100)
        ax.plot(t, gain * t, "g--", lw=2, label=f"Linear fit (gain={gain:.3f})")
    ax.set_title("TX-RX Nonlinearity")
    ax.set_xlabel("TX Amplitude")
    ax.set_ylabel("RX Amplitude")
    ax.legend()
    ax.grid(True)
    ax.axis("equal")
    _finalize_figure(fig, out, show)


def plot_bit_power_loading(subcarriers: np.ndarray,
                           snrs_db: np.ndarray,
                           RQ: np.ndarray,
                           S: np.ndarray,
                           ratio: int,
                           rate_gbps: float,
                           out: Path = None,
                           show: bool = None) -> None:
    """绘制每个子载波的 SNR、比特加载与功率分配，并标注 ratio."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # 上半：SNR + bit allocation 双 y 轴
    ax1_bits = ax1.twinx()
    l1 = ax1.plot(subcarriers, snrs_db, "b-", lw=1.5, label="SNR (dB)")
    l2 = ax1_bits.plot(subcarriers, RQ, "r-", lw=1.5, marker="x",
                       markersize=3, label="Bit allocation")
    ax1.set_ylabel("SNR (dB)", color="b")
    ax1_bits.set_ylabel("Bits / symbol", color="r")
    ax1.set_title(f"Bit-Power Loading (ratio={ratio}, rate={rate_gbps:.2f} Gbps)")
    ax1.grid(True)
    lines = l1 + l2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper right")

    # 下半：功率分配
    ax2.plot(subcarriers, S, "g-", lw=1.5, marker="o", markersize=2,
             label="Power allocation")
    ax2.set_xlabel("Subcarrier")
    ax2.set_ylabel("Power scaling")
    ax2.legend()
    ax2.grid(True)

    _finalize_figure(fig, out, show)


def plot_ser_ber_per_carrier(ser: np.ndarray, ber: np.ndarray,
                             RQ: np.ndarray = None,
                             out: Path = None, show: bool = None) -> None:
    """绘制每个子载波的 SER 与 BER."""
    carrier_idx = np.arange(len(ser))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    ax1.plot(carrier_idx, ser, "r-", marker="o", markersize=3)
    ax1.set_ylabel("SER")
    ax1.set_title("Symbol Error Rate per Subcarrier")
    ax1.grid(True)

    ax2.semilogy(carrier_idx, np.where(ber > 0, ber, 1e-12), "b-", marker="x", markersize=3)
    ax2.set_xlabel("Subcarrier Index")
    ax2.set_ylabel("BER")
    ax2.set_title("Bit Error Rate per Subcarrier")
    ax2.grid(True, which="both", ls="--")

    # 可选叠加 bit allocation，便于对比
    if RQ is not None:
        ax2_twin = ax2.twinx()
        ax2_twin.plot(carrier_idx, RQ, "g--", alpha=0.5, label="Bit allocation")
        ax2_twin.set_ylabel("Bits / symbol", color="g")
        ax2_twin.legend(loc="upper right")

    _finalize_figure(fig, out, show)


def plot_constellation_density(out2: np.ndarray,
                               in_ref: np.ndarray,
                               RQ: np.ndarray,
                               pilot_mask: np.ndarray = None,
                               out: Path = None,
                               show: bool = None) -> None:
    """按调制阶数分类绘制接收星座点分布密度.

    Args:
        out2: 均衡后频域符号 (carrierno1, datano)
        in_ref: 发送参考符号 (carrierno1, datano)
        RQ: 每个子载波的比特数
        pilot_mask: 导频位置掩码
    """
    out2 = np.asarray(out2)
    in_ref = np.asarray(in_ref)
    RQ = np.asarray(RQ).ravel()
    if pilot_mask is None:
        pilot_mask = np.zeros(out2.shape, dtype=bool)
    else:
        pilot_mask = np.asarray(pilot_mask, dtype=bool)

    orders = sorted({int(b) for b in RQ if b > 0})
    if not orders:
        print("No data carriers to plot constellation density.")
        return

    ncols = min(3, len(orders))
    nrows = int(np.ceil(len(orders) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4.5 * nrows),
                             squeeze=False)

    for idx, bits in enumerate(orders):
        ax = axes[idx // ncols, idx % ncols]
        carriers = np.where(RQ == bits)[0]
        pts = []
        ideal = []
        for n in carriers:
            valid = ~pilot_mask[n, :]
            if not np.any(valid):
                continue
            pts.append(out2[n, valid])
            ideal.append(in_ref[n, valid])
        if not pts:
            ax.set_visible(False)
            continue
        pts = np.concatenate(pts)
        ideal = np.concatenate(ideal)

        # 星座点密度：hexbin
        hb = ax.hexbin(pts.real, pts.imag, gridsize=max(30, 2 * int(2 ** (bits / 2))),
                       cmap="GnBu", mincnt=1)
        fig.colorbar(hb, ax=ax, label="Density")

        # 叠加理想星座点（取发送符号的 unique 值）
        ideal_unique = np.unique(np.round(ideal, decimals=6))
        ax.plot(ideal_unique.real, ideal_unique.imag, "c+", markersize=6,
                label="Ideal constellation")

        ax.set_title(f"{2**bits}-QAM (bits={bits}, carriers={len(carriers)})")
        ax.set_xlabel("I")
        ax.set_ylabel("Q")
        ax.axis("equal")
        ax.grid(True)
        ax.legend(loc="upper right", fontsize=7)

    # 隐藏未使用的子图
    for idx in range(len(orders), nrows * ncols):
        axes[idx // ncols, idx % ncols].set_visible(False)

    fig.suptitle("Constellation Density by Modulation Order", y=1.02)
    _finalize_figure(fig, out, show)



def plot_constellation_by_order(out2: np.ndarray,
                                in_ref: np.ndarray,
                                RQ: np.ndarray,
                                pilot_mask: np.ndarray = None,
                                out: Path = None,
                                show: bool = None) -> None:
    """按调制阶数分类绘制接收星座点散点图（每个阶数一张子图）.

    Args:
        out2: 均衡后频域符号 (carrierno1, datano)
        in_ref: 发送参考符号 (carrierno1, datano)
        RQ: 每个子载波的比特数
        pilot_mask: 导频位置掩码
    """
    out2 = np.asarray(out2)
    in_ref = np.asarray(in_ref)
    RQ = np.asarray(RQ).ravel()
    if pilot_mask is None:
        pilot_mask = np.zeros(out2.shape, dtype=bool)
    else:
        pilot_mask = np.asarray(pilot_mask, dtype=bool)

    orders = sorted({int(b) for b in RQ if b > 0})
    if not orders:
        print("No data carriers to plot constellation by order.")
        return

    ncols = min(3, len(orders))
    nrows = int(np.ceil(len(orders) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4.5 * nrows),
                             squeeze=False)

    for idx, bits in enumerate(orders):
        ax = axes[idx // ncols, idx % ncols]
        carriers = np.where(RQ == bits)[0]
        pts = []
        ideal = []
        for n in carriers:
            valid = ~pilot_mask[n, :]
            if not np.any(valid):
                continue
            pts.append(out2[n, valid])
            ideal.append(in_ref[n, valid])
        if not pts:
            ax.set_visible(False)
            continue
        pts = np.concatenate(pts)
        ideal = np.concatenate(ideal)

        ax.plot(pts.real, pts.imag, "b.", alpha=0.3, markersize=3)
        ideal_unique = np.unique(np.round(ideal, decimals=6))
        ax.plot(ideal_unique.real, ideal_unique.imag, "r+", markersize=6,
                label="Ideal constellation")

        ax.set_title(f"{2**bits}-QAM (bits={bits}, carriers={len(carriers)})")
        ax.set_xlabel("I")
        ax.set_ylabel("Q")
        ax.axis("equal")
        ax.grid(True)
        ax.legend(loc="upper right", fontsize=7)

    # 隐藏未使用的子图
    for idx in range(len(orders), nrows * ncols):
        axes[idx // ncols, idx % ncols].set_visible(False)

    fig.suptitle("RX Constellation by Modulation Order", y=1.02)
    _finalize_figure(fig, out, show)
