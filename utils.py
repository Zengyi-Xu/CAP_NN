"""通用工具函数：文件读写、同步、重采样、绘图等。"""
import json
import os
import numpy as np
import scipy.signal as sg
import scipy.io as sio
from scipy.interpolate import CubicSpline
from pathlib import Path
import matplotlib
import config

# 在 IPython/Spyder 中提前设置 inline 后端，使图像显示在 Plots 面板中
if config.PLOT_SHOW:
    try:
        from IPython import get_ipython
        ip = get_ipython()
        if ip is not None:
            ip.run_line_magic("matplotlib", "inline")
    except Exception:
        pass

import matplotlib.pyplot as plt

# 统一默认分辨率
matplotlib.rcParams["figure.dpi"] = config.PLOT_DPI


# -----------------------------------------------------------------------------
# 文件读写
# -----------------------------------------------------------------------------
def load_txt(path: Path, dtype=float) -> np.ndarray:
    """将文本文件读取为 numpy 数组。"""
    if isinstance(path, str):
        path = Path(path)
    return np.loadtxt(path, dtype=dtype)


def save_txt(path: Path, data: np.ndarray, fmt="%.6f") -> None:
    """将 numpy 数组保存到文本文件。"""
    if isinstance(path, str):
        path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, data, fmt=fmt)


def load_mat(path: Path, squeeze=True) -> dict:
    """读取 .mat 文件并返回字典。"""
    if isinstance(path, str):
        path = Path(path)
    return sio.loadmat(path, squeeze_me=squeeze)


def save_mat(path: Path, **kwargs) -> None:
    """将变量保存到 .mat 文件。"""
    if isinstance(path, str):
        path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sio.savemat(path, kwargs)


def save_rx_config(rx_file: Path, **kwargs) -> Path:
    """保存与接收波形对应的信号配置 JSON。

    Args:
        rx_file: 接收波形文件路径，如 rawOSC_QPSK_SNRest_0.txt
        **kwargs: 要记录的配置项

    Returns:
        保存的 JSON 文件路径，如 rawOSC_QPSK_SNRest_0_config.json
    """
    if isinstance(rx_file, str):
        rx_file = Path(rx_file)
    cfg_path = rx_file.parent / f"{rx_file.stem}_config.json"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    # 确保可序列化
    record = {}
    for k, v in kwargs.items():
        if isinstance(v, np.ndarray):
            record[k] = v.tolist()
        elif isinstance(v, Path):
            record[k] = str(v)
        else:
            record[k] = v
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)
    return cfg_path


# -----------------------------------------------------------------------------
# 波形处理
# -----------------------------------------------------------------------------
def add_dummy(x: np.ndarray, base: int = 64) -> tuple:
    """在波形末尾补零，使其长度为 base 的整数倍。

    Returns:
        with_dummy: 补零后的数组
        dummies: 追加的零数组
    """
    x = np.asarray(x).ravel()
    if x.ndim != 1:
        raise ValueError("输入 x 必须是一维数组")
    n_dummy = (-len(x)) % base
    if n_dummy == 0:
        return x.copy(), np.array([])
    dummies = np.zeros(n_dummy, dtype=x.dtype)
    return np.concatenate([x, dummies]), dummies


def resample_signal(x: np.ndarray, fs_target: float, fs_source: float) -> np.ndarray:
    """使用有理数重采样将 x 从 fs_source 重采样到 fs_target。"""
    x = np.asarray(x).ravel()
    if fs_source == fs_target:
        return x
    # 寻找一个近似的整数比
    from fractions import Fraction
    frac = fs_target / fs_source
    f = Fraction(frac).limit_denominator(1000)
    up, down = f.numerator, f.denominator
    return sg.resample_poly(x, up, down)


def sync_waveform(rx: np.ndarray, tx: np.ndarray, h: int = 1) -> np.ndarray:
    """使用 FFT 加速的互相关对接收波形进行符号同步，
    返回与 tx 长度相同的一段。

    Args:
        rx: 接收波形（一维）
        tx: 发送波形（一维）
        h: 同步偏移修正量

    Returns:
        与 tx 等长的同步后接收波形
    """
    rx = np.asarray(rx).ravel()
    tx = np.asarray(tx).ravel()
    # 使用 FFT 加速（避免大长度下 np.correlate 的 O(N^2) 直接计算）
    corr = sg.correlate(np.real(rx), np.real(tx), mode="full", method="fft")
    lags = np.arange(-(len(tx) - 1), len(rx))
    offset = np.argmax(np.abs(corr))
    start = lags[offset] + h
    end = start + len(tx)
    if end > len(rx):
        # 若超出范围，则找第二大的值
        corr[offset] = 0
        offset = np.argmax(np.abs(corr))
        start = lags[offset] + h
        end = start + len(tx)
    if start < 0 or end > len(rx):
        raise ValueError(
            f"同步失败：start={start}, end={end}, rx_len={len(rx)}"
        )
    return rx[start:end]


def center_normalize(x: np.ndarray) -> np.ndarray:
    """最大-最小归一化并居中（与 MATLAB 代码一致）。"""
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
    """应用硬件预均衡（pre_equ_flag=3）。"""
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
    """根据参数或配置决定是否显示图像。"""
    if show is None:
        return config.PLOT_SHOW
    return show


def _finalize_figure(fig, out: Path, show: bool = None) -> None:
    """保存和/或显示图像。

    在 Spyder 中显示图像时不要关闭它，否则 Plots 面板会被清空。
    """
    show = _should_show(show)
    if out is not None and config.PLOT_SAVE:
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=config.PLOT_DPI, bbox_inches="tight")
    if show:
        # 在 Spyder 的 Inline 后端下显示于 Plots 面板
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
    ax.set_xlabel("采样点")
    ax.set_ylabel("幅度")
    ax.grid(True)
    _finalize_figure(fig, out, show)


def plot_spectrum(sig: np.ndarray, fs: float, title: str, out: Path = None,
                  show: bool = None) -> None:
    n = len(sig)
    freqs = np.fft.fftshift(np.fft.fftfreq(n, d=1.0 / fs))
    # 与 MATLAB 一致：10*log10(abs(fft(sig)))
    spec = 10 * np.log10(np.abs(np.fft.fftshift(np.fft.fft(sig))) + 1e-12)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(freqs, spec, "b-")
    ax.set_title(title)
    ax.set_xlabel("频率 (Hz)")
    ax.set_ylabel("幅度 (dB)")
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


def _snr_to_db(snr: np.ndarray) -> np.ndarray:
    """将线性 SNR 转换为 dB，避免 log(0)。"""
    return 10 * np.log10(np.maximum(np.asarray(snr, dtype=float), 1e-12))


def plot_snrs(est: np.ndarray, real: np.ndarray, out: Path = None,
              show: bool = None) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(_snr_to_db(est), "b", label="估计 SNR", marker="o")
    ax.plot(_snr_to_db(real), "r", label="实测 SNR", marker="x")
    ax.set_title("估计 SNR 与恢复 SNR 对比")
    ax.set_xlabel("子载波")
    ax.set_ylabel("SNR (dB)")
    ax.legend()
    ax.grid(True)
    _finalize_figure(fig, out, show)


# -----------------------------------------------------------------------------
# 新增：DMT 专用绘图
# -----------------------------------------------------------------------------
def plot_dmt_spectrogram(sig: np.ndarray, fs: float, title: str,
                         out: Path = None, show: bool = None) -> None:
    """绘制 DMT 信号频谱图。"""
    sig = np.asarray(sig).ravel()
    nperseg = min(1024, len(sig) // 8)
    noverlap = nperseg // 2
    f, t, Sxx = sg.spectrogram(sig, fs=fs, nperseg=nperseg, noverlap=noverlap,
                               window="hann", scaling="spectrum")
    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.pcolormesh(t * 1e6, f / 1e9, 10 * np.log10(Sxx + 1e-12),
                       shading="gouraud", cmap="jet")
    ax.set_title(title)
    ax.set_xlabel("时间 (us)")
    ax.set_ylabel("频率 (GHz)")
    fig.colorbar(im, ax=ax, label="功率 (dB)")
    _finalize_figure(fig, out, show)


def plot_tx_rx_nonlinearity(tx: np.ndarray, rx: np.ndarray,
                            out: Path = None, show: bool = None) -> None:
    """绘制接收信号非线性：x 轴为发送波形幅度，y 轴为接收波形幅度。"""
    tx = np.asarray(tx).ravel()
    rx = np.asarray(rx).ravel()
    if len(tx) != len(rx):
        raise ValueError("tx 和 rx 的长度必须相同")

    fig, ax = plt.subplots(figsize=(7, 7))
    # 点数过多时使用 hexbin，否则使用散点图
    if len(tx) > 5000:
        hb = ax.hexbin(tx, rx, gridsize=80, cmap="GnBu", mincnt=1)
        fig.colorbar(hb, ax=ax, label="密度")
    else:
        ax.plot(tx, rx, "b.", alpha=0.2)

    # 理想线性参考线（斜率为 tx->rx 增益的最小二乘估计）
    if np.any(tx):
        gain = np.sum(tx * rx) / np.sum(tx ** 2)
        t = np.linspace(tx.min(), tx.max(), 100)
        ax.plot(t, gain * t, "g--", lw=2, label=f"线性拟合 (gain={gain:.3f})")
    ax.set_title("发送-接收非线性")
    ax.set_xlabel("发送幅度")
    ax.set_ylabel("接收幅度")
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
    """绘制各子载波的 SNR、比特加载和功率分配，并标注比率。"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # 上图：SNR + 比特分配，双 y 轴
    ax1_bits = ax1.twinx()
    l1 = ax1.plot(subcarriers, snrs_db, "b-", lw=1.5, label="SNR (dB)")
    l2 = ax1_bits.plot(subcarriers, RQ, "r-", lw=1.5, marker="x",
                       markersize=3, label="比特分配")
    ax1.set_ylabel("SNR (dB)", color="b")
    ax1_bits.set_ylabel("比特/符号", color="r")
    ax1.set_title(f"比特-功率加载 (ratio={ratio}, rate={rate_gbps:.2f} Gbps)")
    ax1.grid(True)
    lines = l1 + l2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper right")

    # 下图：功率分配
    ax2.plot(subcarriers, S, "g-", lw=1.5, marker="o", markersize=2,
             label="功率分配")
    ax2.set_xlabel("子载波")
    ax2.set_ylabel("功率缩放")
    ax2.legend()
    ax2.grid(True)

    _finalize_figure(fig, out, show)


def plot_ser_ber_per_carrier(ser: np.ndarray, ber: np.ndarray,
                             RQ: np.ndarray = None,
                             out: Path = None, show: bool = None) -> None:
    """绘制各子载波的 SER 和 BER。"""
    carrier_idx = np.arange(len(ser))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    ax1.plot(carrier_idx, ser, "r-", marker="o", markersize=3)
    ax1.set_ylabel("SER")
    ax1.set_title("各子载波符号误码率")
    ax1.grid(True)

    ax2.semilogy(carrier_idx, np.where(ber > 0, ber, 1e-12), "b-", marker="x", markersize=3)
    ax2.set_xlabel("子载波序号")
    ax2.set_ylabel("BER")
    ax2.set_title("各子载波比特误码率")
    ax2.grid(True, which="both", ls="--")

    # 可选：叠加比特分配以便对比
    if RQ is not None:
        ax2_twin = ax2.twinx()
        ax2_twin.plot(carrier_idx, RQ, "g--", alpha=0.5, label="比特分配")
        ax2_twin.set_ylabel("比特/符号", color="g")
        ax2_twin.legend(loc="upper right")

    _finalize_figure(fig, out, show)


def plot_constellation_density(out2: np.ndarray,
                               in_ref: np.ndarray,
                               RQ: np.ndarray,
                               pilot_mask: np.ndarray = None,
                               out: Path = None,
                               show: bool = None) -> None:
    """按调制阶数分类绘制接收星座密度图。

    Args:
        out2: 均衡后的频域符号 (carrierno1, datano)
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
        print("没有可绘制星座密度的数据子载波。")
        return

    ncols = min(3, len(orders))
    nrows = int(np.ceil(len(orders) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4.5 * nrows),
                             squeeze=False)

    for idx, bits in enumerate(orders):
        ax = axes[idx // ncols, idx % ncols]
        carriers = np.where(RQ == bits)[0]
        pts = []
        for n in carriers:
            valid = ~pilot_mask[n, :]
            if not np.any(valid):
                continue
            pts.append(out2[n, valid])
        if not pts:
            ax.set_visible(False)
            continue
        pts = np.concatenate(pts)

        # 星座点密度：hexbin
        hb = ax.hexbin(pts.real, pts.imag, gridsize=max(30, 2 * int(2 ** (bits / 2))),
                       cmap="GnBu", mincnt=1)
        fig.colorbar(hb, ax=ax, label="密度")

        ax.set_title(f"{2**bits}-QAM (bits={bits}, carriers={len(carriers)})")
        ax.set_xlabel("I")
        ax.set_ylabel("Q")
        ax.axis("equal")
        ax.grid(True)

    # 隐藏未使用的子图
    for idx in range(len(orders), nrows * ncols):
        axes[idx // ncols, idx % ncols].set_visible(False)

    fig.suptitle("按调制阶数分类的星座密度", y=1.02)
    _finalize_figure(fig, out, show)



def plot_constellation_by_order(out2: np.ndarray,
                                in_ref: np.ndarray,
                                RQ: np.ndarray,
                                pilot_mask: np.ndarray = None,
                                out: Path = None,
                                show: bool = None) -> None:
    """按调制阶数分类绘制接收星座散点图（每种阶数一个子图）。

    Args:
        out2: 均衡后的频域符号 (carrierno1, datano)
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
        print("没有可绘制星座图的数据子载波。")
        return

    ncols = min(3, len(orders))
    nrows = int(np.ceil(len(orders) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4.5 * nrows),
                             squeeze=False)

    for idx, bits in enumerate(orders):
        ax = axes[idx // ncols, idx % ncols]
        carriers = np.where(RQ == bits)[0]
        pts = []
        for n in carriers:
            valid = ~pilot_mask[n, :]
            if not np.any(valid):
                continue
            pts.append(out2[n, valid])
        if not pts:
            ax.set_visible(False)
            continue
        pts = np.concatenate(pts)

        ax.plot(pts.real, pts.imag, "b.", alpha=0.3, markersize=3)

        ax.set_title(f"{2**bits}-QAM (bits={bits}, carriers={len(carriers)})")
        ax.set_xlabel("I")
        ax.set_ylabel("Q")
        ax.axis("equal")
        ax.grid(True)

    # 隐藏未使用的子图
    for idx in range(len(orders), nrows * ncols):
        axes[idx // ncols, idx % ncols].set_visible(False)

    fig.suptitle("按调制阶数分类的接收星座", y=1.02)
    _finalize_figure(fig, out, show)
