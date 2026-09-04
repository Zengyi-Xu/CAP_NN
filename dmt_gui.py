# -*- coding: utf-8 -*-
"""DMT 通信系统实验平台 GUI。

共六个标签页：
    1. 波形与频谱       —— 发送/接收时域波形与频谱
    2. DMT 调制         —— 比特/功率加载、星座图、星座图密度
    3. 传输结果         —— 实验记录表、信噪比对比、
                          各子载波 SER/误码率、非线性、实验历史趋势
    4. 运行测试         —— 从 GUI 调用 main.py 并实时显示日志输出
    5. Keithley 2400    —— 通过 RS-232/USB 控制 Keithley 2400 源表
    6. 网格扫描         —— 偏置 vs Vpp 参数自动扫描，输出 CSV/CodePlot

数据来源（与 main.py 的自动保存位置相同）：
    data/records/record_<run_id>.json          每次运行的参数和结果
                                               （下拉框的主要来源）
    data/codeplot_assets/<run_id>/data/*.npz   每次运行的绘图数据（存在时显示）

运行方式（在项目目录下）：
    python dmt_gui.py

仅依赖 numpy / matplotlib / tkinter（Python 自带）；无需额外安装。
"""
import ast
import json
import os
import queue
import re
import subprocess
import sys
import threading
import tkinter as tk
import traceback
from tkinter import ttk, messagebox, filedialog
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import config
from record import generate_run_id
from keithley2400_controller import (
    Keithley2400,
    refresh_port_list,
    parse_port_entry,
    K2400ConnectionError,
    K2400CommandError,
    K2400ConfigError,
)
from grid_scan import GridScanner, GridScanConfig, list_grid_scans, load_summary, GRID_SCAN_DIR

try:
    import openpyxl
    _HAS_OPENPYXL = True
except Exception:
    _HAS_OPENPYXL = False

try:
    from PIL import Image, ImageDraw, ImageFont
    _HAS_PIL = True
except Exception:
    _HAS_PIL = False

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

# 首选 UI / 绘图无衬线字体（树莓派自带）
UI_FONT = "Liberation Sans"
UI_FONT_FALLBACKS = ["DejaVu Sans", "Liberation Sans"]

# 代码/日志区域首选等宽字体（树莓派自带）
MONO_FONT = "Liberation Mono Bold"
MONO_FONT_FALLBACKS = ["Liberation Mono", "DejaVu Sans Mono", "Courier"]

import matplotlib.font_manager as fm


def _pick_available_font(candidates):
    """返回 candidates 中系统上存在的第一个字体。"""
    for name in candidates:
        try:
            fm.findfont(name, fallback_to_default=False)
            return name
        except Exception:
            continue
    return candidates[-1] if candidates else "DejaVu Sans"


FONT_FAMILY = _pick_available_font([UI_FONT] + UI_FONT_FALLBACKS)
FONT_MONO = _pick_available_font([MONO_FONT] + MONO_FONT_FALLBACKS)

plt.rcParams["font.sans-serif"] = [FONT_FAMILY] + [f for f in UI_FONT_FALLBACKS if f != FONT_FAMILY]
plt.rcParams["axes.unicode_minus"] = False

PROJECT_ROOT = Path(__file__).resolve().parent
ASSETS_DIR = PROJECT_ROOT / "data" / "codeplot_assets"
RECORDS_DIR = PROJECT_ROOT / "data" / "records"
MAIN_PY = PROJECT_ROOT / "main.py"
CONFIG_PY = PROJECT_ROOT / "config.py"

APP_EMOJI = "📶"  # 用于窗口图标和标题的 emoji；可按喜好更改


# ═══════════════════════════════════════════════════════════════════════════════
# 高 DPI 适配（必须在创建 Tk 之前调用）
# ═══════════════════════════════════════════════════════════════════════════════

def enable_dpi_awareness():
    """按实际 DPI 渲染窗口，避免高 DPI 屏幕上界面过小/模糊。"""
    try:
        import ctypes
        # PROCESS_PER_MONITOR_DPI_AWARE = 2
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        try:
            import ctypes
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════════════════════
# 配色方案（现代浅色字典风格）
# ═══════════════════════════════════════════════════════════════════════════════

COLOR_BG = "#F3F5F7"          # 窗口背景（浅灰）
COLOR_CARD = "#FFFFFF"        # 卡片白色
COLOR_BORDER = "#E2E8F0"      # 卡片边框
COLOR_PRIMARY = "#164E63"     # 主色（深青）
COLOR_PRIMARY_HOVER = "#0E7490"
COLOR_TEXT = "#1F2937"        # 主文字
COLOR_TEXT_DIM = "#64748B"    # 次要文字
COLOR_DANGER = "#B91C1C"
COLOR_SELECT = "#164E63"      # 选中颜色


# ═══════════════════════════════════════════════════════════════════════════════
# Data discovery
# ═══════════════════════════════════════════════════════════════════════════════

def run_data_dir(run_id):
    return ASSETS_DIR / run_id / "data"


def available_plots(run_id, names):
    """返回该运行实际存在的图像名称列表（保持 names 的顺序）。"""
    d = run_data_dir(run_id)
    return [n for n in names if (d / f"{n}.npz").is_file()]


def _resolve_source_run_id(step: str, run_suffix: str) -> str:
    """根据 step 和 run_suffix 解析源 RX 文件的完整 run_id。"""
    try:
        if step in ("all", "step2"):
            pattern = "rawOSC_QPSK_SNRest_*.txt"
        elif step == "step4":
            pattern = "rawOSC_DMT_*.txt"
        else:
            return None
        suffix_pattern = f"*{run_suffix}.txt"
        matches = sorted(config.RXDATA_DIR.glob(suffix_pattern),
                         key=lambda p: p.stat().st_mtime, reverse=True)
        if not matches:
            return None
        stage_prefix = pattern.replace("_*.txt", "")
        stage_matches = [m for m in matches if m.name.startswith(stage_prefix)]
        target = stage_matches[0] if stage_matches else matches[0]
        parts = target.stem.split("_")
        if len(parts) >= 3:
            return "_".join(parts[-3:])
    except Exception:
        pass
    return None


def _write_array_to_sheet(ws, arr, start_row=1, start_col=1):
    """将 numpy 数组写入 openpyxl 工作表；复数拆分为实部/虚部两列。"""
    # 处理 0 维标量（避免 arr.shape[0] 出现元组索引越界）
    if arr.ndim == 0:
        val = arr.item()
        if np.iscomplexobj(arr):
            ws.cell(row=start_row, column=start_col, value=float(val.real))
            ws.cell(row=start_row, column=start_col + 1, value=float(val.imag))
        else:
            ws.cell(row=start_row, column=start_col,
                    value=float(val) if isinstance(val, (int, float, np.number)) else val)
        return
    # 写入表头
    if start_row > 1:
        if arr.dtype.kind == "c":
            if arr.ndim == 1:
                ws.cell(row=start_row - 1, column=start_col, value="实部")
                ws.cell(row=start_row - 1, column=start_col + 1, value="虚部")
            else:
                for c in range(arr.shape[1]):
                    ws.cell(row=start_row - 1, column=start_col + c * 2,
                            value=f"第{c}列_实部")
                    ws.cell(row=start_row - 1, column=start_col + c * 2 + 1,
                            value=f"第{c}列_虚部")
        else:
            if arr.ndim == 1:
                ws.cell(row=start_row - 1, column=start_col, value="数值")
            else:
                for c in range(arr.shape[1]):
                    ws.cell(row=start_row - 1, column=start_col + c,
                            value=f"第{c}列")
    # 写入数据
    if arr.dtype.kind == "c":
        for r in range(arr.shape[0]):
            if arr.ndim == 1:
                ws.cell(row=start_row + r, column=start_col,
                        value=float(arr[r].real))
                ws.cell(row=start_row + r, column=start_col + 1,
                        value=float(arr[r].imag))
            else:
                for c in range(arr.shape[1]):
                    ws.cell(row=start_row + r, column=start_col + c * 2,
                            value=float(arr[r, c].real))
                    ws.cell(row=start_row + r, column=start_col + c * 2 + 1,
                            value=float(arr[r, c].imag))
    else:
        for r in range(arr.shape[0]):
            if arr.ndim == 1:
                val = arr[r]
                ws.cell(row=start_row + r, column=start_col,
                        value=float(val) if isinstance(val, (int, float, np.number)) else val)
            else:
                for c in range(arr.shape[1]):
                    val = arr[r, c]
                    ws.cell(row=start_row + r, column=start_col + c,
                            value=float(val) if isinstance(val, (int, float, np.number)) else val)


def _safe_sheet_name(name: str) -> str:
    """Excel 工作表名必须 <= 31 个字符且不含特殊字符。"""
    invalid = ["\\", "/", "?", "*", "[", "]", ":"]
    for ch in invalid:
        name = name.replace(ch, "_")
    return name[:31]


def _cleanup_empty_plot_dirs():
    """删除 data/plots 下所有空文件夹（保留包含文件的目录）。"""
    try:
        if not config.PLOT_DIR.is_dir():
            return
        for d in config.PLOT_DIR.iterdir():
            if d.is_dir() and not any(d.iterdir()):
                d.rmdir()
    except Exception:
        pass


def _npz_to_xlsx(npz_path: Path, xlsx_path: Path):
    """将 NPZ 文件中的每个数组导出到 Excel 的单独工作表。"""
    if not _HAS_OPENPYXL:
        raise RuntimeError("缺少 openpyxl；请运行: pip install openpyxl")
    data = np.load(npz_path)
    wb = openpyxl.Workbook()
    wb.remove(wb.active)
    errors = []
    for key in data.files:
        arr = data[key]
        if arr.ndim > 2:
            arr = arr.reshape(arr.shape[0], -1)
        sheet_name = _safe_sheet_name(key)
        ws = wb.create_sheet(title=sheet_name)
        ws.cell(row=1, column=1, value=f"数组: {key}")
        ws.cell(row=1, column=2, value=f"形状: {arr.shape}")
        ws.cell(row=1, column=3, value=f"数据类型: {arr.dtype}")
        try:
            _write_array_to_sheet(ws, arr, start_row=3, start_col=1)
        except Exception as exc:
            errors.append(f"{key}: {exc}")
            ws.cell(row=3, column=1, value=f"写入失败: {exc}")
    wb.save(xlsx_path)
    if errors:
        raise RuntimeError("部分数组写入失败: " + "; ".join(errors))


def list_records():
    """读取 data/records 下所有 record_*.json，按时间戳升序返回。"""
    records = []
    if RECORDS_DIR.is_dir():
        for p in sorted(RECORDS_DIR.glob("record_*.json")):
            try:
                rec = json.loads(p.read_text(encoding="utf-8"))
                records.append(rec)
            except (json.JSONDecodeError, OSError):
                continue
    records.sort(key=lambda r: r.get("timestamp", ""))
    return records


# ═══════════════════════════════════════════════════════════════════════════════
# 绘图函数（与 plot_adapter.py 中的 CodePlot 模板保持同步）
# ═══════════════════════════════════════════════════════════════════════════════

def build_time(fig, npz_path, title):
    data = np.load(npz_path)
    t, sig = data["t"], data["sig"]
    ax = fig.add_subplot(111)
    ax.plot(t, sig, "b.-", linewidth=1, markersize=2)
    ax.set_title(title)
    ax.set_xlabel("采样点")
    ax.set_ylabel("幅度")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()


def build_spectrum(fig, npz_path, title):
    data = np.load(npz_path)
    sig, fs = data["sig"], float(data["fs"])
    n = len(sig)
    freqs = np.fft.fftshift(np.fft.fftfreq(n, d=1.0 / fs))
    spec = 10 * np.log10(np.abs(np.fft.fftshift(np.fft.fft(sig))) + 1e-12)
    ax = fig.add_subplot(111)
    ax.plot(freqs / 1e9, spec, "b-", linewidth=1)
    ax.set_title(title)
    ax.set_xlabel("频率 (GHz)")
    ax.set_ylabel("幅度 (dB)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()


def build_constellation(fig, npz_path, title):
    data = np.load(npz_path)
    iq = data["iq"]
    ax = fig.add_subplot(111)
    ax.plot(iq.real, iq.imag, "b.", alpha=0.3, markersize=3)
    ax.set_title(title)
    ax.set_xlabel("I")
    ax.set_ylabel("Q")
    ax.grid(True, alpha=0.3)
    ax.axis("equal")
    fig.tight_layout()


def build_snr(fig, npz_path, title):
    data = np.load(npz_path)
    est = 10 * np.log10(np.maximum(data["est"].astype(float), 1e-12))
    real = 10 * np.log10(np.maximum(data["real"].astype(float), 1e-12))
    ax = fig.add_subplot(111)
    ax.plot(est, "b", label="估计SNR", marker="o", markersize=3, linewidth=1)
    ax.plot(real, "r", label="实测SNR", marker="x", markersize=3,
            linewidth=1)
    ax.set_title(title)
    ax.set_xlabel("子载波")
    ax.set_ylabel("信噪比 (dB)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()


def build_nonlinearity(fig, npz_path, title):
    data = np.load(npz_path)
    tx, rx = data["tx"], data["rx"]
    ax = fig.add_subplot(111)
    if len(tx) > 5000:
        hb = ax.hexbin(tx, rx, gridsize=80, cmap="GnBu", mincnt=1)
        fig.colorbar(hb, ax=ax, label="密度")
    else:
        ax.plot(tx, rx, "b.", alpha=0.2, markersize=3)
    if np.any(tx):
        gain = np.sum(tx * rx) / np.sum(tx ** 2)
        t = np.linspace(tx.min(), tx.max(), 100)
        ax.plot(t, gain * t, "g--", linewidth=2,
                label=f"线性拟合 (增益={gain:.3f})")
    ax.set_title(title)
    ax.set_xlabel("发送端幅度")
    ax.set_ylabel("接收端幅度")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()


def build_bit_power_loading(fig, npz_path, title):
    data = np.load(npz_path)
    subcarriers = data["subcarriers"]
    snrs_db = data["snrs_db"]
    RQ, S = data["RQ"], data["S"]
    ratio = int(data["ratio"])
    rate_gbps = float(data["rate_gbps"])

    ax1 = fig.add_subplot(211)
    ax1_bits = ax1.twinx()
    ax1.plot(subcarriers, snrs_db, "b-", linewidth=1.5, label="信噪比 (dB)")
    ax1_bits.plot(subcarriers, RQ, "r-", linewidth=1.5, marker="x",
                  markersize=3, label="比特分配")
    ax1.set_ylabel("信噪比 (dB)", color="b")
    ax1_bits.set_ylabel("比特/符号", color="r")
    ax1.set_title(f"{title} (ratio={ratio}, 速率={rate_gbps:.2f} Gbps)")
    ax1.grid(True, alpha=0.3)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_bits.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    ax2 = fig.add_subplot(212, sharex=ax1)
    ax2.plot(subcarriers, S, "g-", linewidth=1.5, marker="o", markersize=2,
             label="功率分配")
    ax2.set_xlabel("子载波")
    ax2.set_ylabel("功率缩放")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    fig.tight_layout()


def build_ser_ber(fig, npz_path, title):
    data = np.load(npz_path)
    ser, ber = data["ser"], data["ber"]
    RQ = data["RQ"] if "RQ" in data else None
    carrier_idx = np.arange(len(ser))

    ax1 = fig.add_subplot(211)
    ax1.plot(carrier_idx, ser, "r-", marker="o", markersize=3, linewidth=1)
    ax1.set_ylabel("SER")
    ax1.set_title(f"{title} — 各子载波 SER")
    ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(212)
    ax2.semilogy(carrier_idx, np.where(ber > 0, ber, 1e-12), "b-",
                 marker="x", markersize=3, linewidth=1)
    ax2.set_xlabel("子载波序号")
    ax2.set_ylabel("误码率")
    ax2.set_title("各子载波误码率")
    ax2.grid(True, which="both", ls="--", alpha=0.3)

    if RQ is not None:
        ax2_twin = ax2.twinx()
        ax2_twin.plot(carrier_idx, RQ, "g--", alpha=0.5,
                      label="比特分配")
        ax2_twin.set_ylabel("比特/符号", color="g")
        ax2_twin.legend(loc="upper right")
    fig.tight_layout()


def _orders_and_grid(RQ):
    orders = sorted({int(b) for b in RQ if b > 0})
    if not orders:
        return [], 0, 0
    ncols = min(3, len(orders))
    nrows = int(np.ceil(len(orders) / ncols))
    return orders, nrows, ncols


def build_const_density(fig, npz_path, title):
    data = np.load(npz_path)
    out2, RQ, pilot_mask = data["out2"], data["RQ"], data["pilot_mask"]
    orders, nrows, ncols = _orders_and_grid(RQ)
    if not orders:
        return
    for idx, bits in enumerate(orders):
        ax = fig.add_subplot(nrows, ncols, idx + 1)
        pts = [out2[n, ~pilot_mask[n, :]] for n in np.where(RQ == bits)[0]
               if np.any(~pilot_mask[n, :])]
        if not pts:
            ax.set_visible(False)
            continue
        pts = np.concatenate(pts)
        gridsize = max(30, 2 * int(2 ** (bits / 2)))
        hb = ax.hexbin(pts.real, pts.imag, gridsize=gridsize, cmap="GnBu",
                       mincnt=1)
        fig.colorbar(hb, ax=ax, label="密度")
        ax.set_title(f"{2 ** bits}-QAM (比特={bits})")
        ax.set_xlabel("I")
        ax.set_ylabel("Q")
        ax.axis("equal")
        ax.grid(True, alpha=0.3)
    fig.suptitle(title, y=1.02)
    fig.tight_layout()


def build_const_by_order(fig, npz_path, title):
    data = np.load(npz_path)
    out2, RQ, pilot_mask = data["out2"], data["RQ"], data["pilot_mask"]
    orders, nrows, ncols = _orders_and_grid(RQ)
    if not orders:
        return
    for idx, bits in enumerate(orders):
        ax = fig.add_subplot(nrows, ncols, idx + 1)
        pts = [out2[n, ~pilot_mask[n, :]] for n in np.where(RQ == bits)[0]
               if np.any(~pilot_mask[n, :])]
        if not pts:
            ax.set_visible(False)
            continue
        pts = np.concatenate(pts)
        ax.plot(pts.real, pts.imag, "b.", alpha=0.3, markersize=3)
        ax.set_title(f"{2 ** bits}-QAM (比特={bits})")
        ax.set_xlabel("I")
        ax.set_ylabel("Q")
        ax.axis("equal")
        ax.grid(True, alpha=0.3)
    fig.suptitle(title, y=1.02)
    fig.tight_layout()


def build_records_trend(fig, records, title):
    """跨运行趋势：速率与误码率。"""
    recs = [r for r in records if r.get("final_rate_gbps") is not None]
    if not recs:
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, "暂无实验记录", ha="center", va="center",
                transform=ax.transAxes)
        return
    xs = list(range(len(recs)))
    labels = [r.get("run_id", "")[-6:] for r in recs]
    rates = [r.get("final_rate_gbps", 0) for r in recs]
    bers = [max(r.get("final_ber") or 1e-12, 1e-12) for r in recs]
    snrs = [r.get("mean_recovered_snr_db") for r in recs]

    ax1 = fig.add_subplot(211)
    ax1.plot(xs, rates, "b-o", markersize=4, linewidth=1.5,
             label="最终速率")
    if any(s is not None for s in snrs):
        ax1b = ax1.twinx()
        ax1b.plot(xs, [s if s is not None else np.nan for s in snrs],
                  "g--s", markersize=4, linewidth=1, label="平均信噪比 (dB)")
        ax1b.set_ylabel("平均恢复信噪比 (dB)", color="g")
        ax1b.legend(loc="lower right")
    ax1.set_ylabel("速率 (Gbps)", color="b")
    ax1.set_title(title)
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(xs)
    ax1.set_xticklabels(labels, rotation=45, fontsize=8)

    ax2 = fig.add_subplot(212, sharex=ax1)
    ax2.semilogy(xs, bers, "r-x", markersize=4, linewidth=1.5)
    ax2.set_xlabel("实验（按时间顺序）")
    ax2.set_ylabel("误码率")
    ax2.grid(True, which="both", ls="--", alpha=0.3)
    ax2.set_xticks(xs)
    ax2.set_xticklabels(labels, rotation=45, fontsize=8)
    fig.tight_layout()


# ═══════════════════════════════════════════════════════════════════════════════
# 图像目录：名称 -> (绘图函数, 标题)
# ═══════════════════════════════════════════════════════════════════════════════

FIGURES = {
    # 标签页 1：波形与频谱
    "SNRest_QPSK_time":          (build_time,          "QPSK 探测信号发送端时域波形"),
    "SNRest_QPSK_spec":          (build_spectrum,      "QPSK 探测信号发送端频谱"),
    "SNRest_QPSK_rx_spec":       (build_spectrum,      "QPSK 探测信号接收端频谱"),
    "DMT_bitloading_Tx_time":    (build_time,          "DMT 比特加载发送端时域波形"),
    "DMT_bitloading_Tx_spec":    (build_spectrum,      "DMT 比特加载发送端频谱"),
    "DMT_bitloading_Rx_spec":    (build_spectrum,      "DMT 比特加载接收端频谱"),
    # 标签页 2：DMT 调制
    "bit_power_loading":         (build_bit_power_loading, "比特 / 功率加载"),
    "SNRest_QPSK_constellation": (build_constellation, "QPSK 探测信号星座图"),
    "bitloading_constellation":  (build_constellation, "DMT 比特加载星座图"),
    "constellation_by_order":    (build_const_by_order,    "接收端星座图（按调制阶数）"),
    "constellation_density":     (build_const_density,     "星座图密度（按调制阶数）"),
    # 标签页 3：传输结果
    "SNR_QPSK":                  (build_snr,           "QPSK 估计信噪比与实测信噪比"),
    "SNR_compare":               (build_snr,           "信噪比对比（估计 vs 恢复）"),
    "ser_ber_per_carrier":       (build_ser_ber,       "各子载波 SER / 误码率"),
    "SNRest_QPSK_nonlinearity":  (build_nonlinearity,  "QPSK 发送-接收幅度非线性"),
    "DMT_bitloading_nonlinearity": (build_nonlinearity, "DMT 发送-接收幅度非线性"),
}

TAB_WAVEFORM = [
    "SNRest_QPSK_time", "SNRest_QPSK_spec", "SNRest_QPSK_rx_spec",
    "DMT_bitloading_Tx_time", "DMT_bitloading_Tx_spec",
    "DMT_bitloading_Rx_spec",
]
TAB_MODULATION = [
    "bit_power_loading", "SNRest_QPSK_constellation",
    "bitloading_constellation", "constellation_by_order",
    "constellation_density",
]
TAB_RESULTS = [
    "SNR_QPSK", "SNR_compare", "ser_ber_per_carrier",
    "SNRest_QPSK_nonlinearity", "DMT_bitloading_nonlinearity",
]

TREND_KEY = "__records_trend__"   # 标签页 3 内实验历史趋势的虚拟图像名称


# ═══════════════════════════════════════════════════════════════════════════════
# Styles
# ═══════════════════════════════════════════════════════════════════════════════

def apply_styles(root, scale):
    """现代浅色主题（在 clam 基础上定制）。"""
    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except tk.TclError:
        pass

    f = max(scale, 1.0)
    # 字体使用磅值字号（tk scaling 已处理 DPI）；像素尺寸仅乘以 f
    font_base = (FONT_FAMILY, 10)
    font_bold = (FONT_FAMILY, 10, "bold")
    font_tab = (FONT_FAMILY, 11)
    pad_x = int(round(14 * f))
    pad_y = int(round(8 * f))

    style.configure(".", font=font_base, background=COLOR_BG,
                    foreground=COLOR_TEXT)

    # 框架 / 标签
    style.configure("TFrame", background=COLOR_BG)
    style.configure("Card.TFrame", background=COLOR_CARD)
    style.configure("TLabel", background=COLOR_BG, foreground=COLOR_TEXT)
    style.configure("Card.TLabel", background=COLOR_CARD,
                    foreground=COLOR_TEXT)
    style.configure("Dim.TLabel", background=COLOR_BG,
                    foreground=COLOR_TEXT_DIM)
    style.configure("DimCard.TLabel", background=COLOR_CARD,
                    foreground=COLOR_TEXT_DIM)
    style.configure("Title.TLabel", background=COLOR_BG,
                    foreground=COLOR_PRIMARY,
                    font=(FONT_FAMILY, 17, "bold"))
    style.configure("Subtitle.TLabel", background=COLOR_BG,
                    foreground=COLOR_TEXT_DIM,
                    font=(FONT_FAMILY, 10))
    style.configure("Section.TLabel", background=COLOR_CARD,
                    foreground=COLOR_PRIMARY, font=font_bold)
    # 胶囊徽标（统计信息）
    style.configure("Pill.TLabel", background=COLOR_PRIMARY,
                    foreground="#FFFFFF", font=font_bold,
                    padding=(pad_x, int(round(4 * f))))
    style.configure("Metrics.TLabel", background=COLOR_BG,
                    foreground=COLOR_PRIMARY_HOVER, font=font_bold)

    # Notebook 标签页：大内边距，选中时为白色
    style.configure("TNotebook", background=COLOR_BG, borderwidth=0)
    style.configure("TNotebook.Tab", font=font_tab,
                    padding=(pad_x + 6, pad_y),
                    background="#E5EAEF", foreground=COLOR_TEXT)
    style.map("TNotebook.Tab",
              background=[("selected", COLOR_CARD)],
              foreground=[("selected", COLOR_PRIMARY)])

    # 主按钮（深青背景，白色文字）
    style.configure("Accent.TButton", font=font_bold,
                    padding=(pad_x, pad_y),
                    background=COLOR_PRIMARY, foreground="#FFFFFF",
                    borderwidth=0, focusthickness=0)
    style.map("Accent.TButton",
              background=[("active", COLOR_PRIMARY_HOVER),
                          ("disabled", "#9FB3BC")],
              foreground=[("disabled", "#E5EAEF")])
    # 普通按钮
    style.configure("TButton", font=font_base, padding=(pad_x, pad_y),
                    background="#E5EAEF", foreground=COLOR_TEXT,
                    borderwidth=0)
    style.map("TButton", background=[("active", "#D5DDE4")])
    # 危险按钮
    style.configure("Danger.TButton", font=font_bold,
                    padding=(pad_x, pad_y),
                    background=COLOR_DANGER, foreground="#FFFFFF",
                    borderwidth=0)
    style.map("Danger.TButton",
              background=[("active", "#DC2626"), ("disabled", "#D1A5A5")])

    # 单选/复选（指示器随 DPI 缩放，高 DPI 屏幕上不会过小）
    indicator = int(round(13 * f))
    style.configure("TRadiobutton", background=COLOR_CARD,
                    foreground=COLOR_TEXT, font=font_base,
                    indicatorsize=indicator)
    style.configure("TCheckbutton", background=COLOR_CARD,
                    foreground=COLOR_TEXT, font=font_base,
                    indicatorsize=indicator)
    style.configure("TLabelframe", background=COLOR_CARD,
                    bordercolor=COLOR_BORDER)
    style.configure("TLabelframe.Label", background=COLOR_CARD,
                    foreground=COLOR_PRIMARY, font=font_bold)

    # 下拉框
    style.configure("TCombobox", padding=(int(round(8 * f)),
                                          int(round(4 * f))))

    # 表格
    style.configure("Treeview", background=COLOR_CARD,
                    fieldbackground=COLOR_CARD, foreground=COLOR_TEXT,
                    rowheight=int(round(28 * f)), font=font_base,
                    borderwidth=0)
    style.configure("Treeview.Heading", background="#EEF2F5",
                    foreground=COLOR_PRIMARY, font=font_bold,
                    padding=(int(round(6 * f)), int(round(6 * f))))
    style.map("Treeview",
              background=[("selected", COLOR_SELECT)],
              foreground=[("selected", "#FFFFFF")])

    # 分隔线 / 滚动条
    style.configure("TSeparator", background=COLOR_BORDER)
    style.configure("Vertical.TScrollbar", background="#D5DDE4",
                    troughcolor=COLOR_BG, borderwidth=0, arrowsize=12)


def make_card(parent, **pack_kwargs):
    """白色卡片容器（细边框 + 内边距）。"""
    card = tk.Frame(parent, bg=COLOR_CARD,
                    highlightbackground=COLOR_BORDER, highlightthickness=1,
                    bd=0)
    if pack_kwargs:
        card.pack(**pack_kwargs)
    return card


def _set_windows_taskbar_icon():
    """设置 Windows 任务栏图标：需要显式指定 AppUserModelID 才能摆脱默认的羽毛图标。"""
    if sys.platform != "win32":
        return
    try:
        import ctypes
        # 任意唯一 ID 都可以；与 .ico 文件无关
        app_id = "DMT.PY.NN.GUI.v1"
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(app_id)
    except Exception:
        pass


def _create_emoji_icon(emoji: str, size: int = 64):
    """将 emoji 渲染为窗口图标，返回 (PhotoImage, ico_path)。"""
    if not _HAS_PIL:
        return None, None
    try:
        img = Image.new("RGBA", (size, size), (255, 255, 255, 0))
        draw = ImageDraw.Draw(img)
        # 优先使用 Linux/树莓派字体，其次 Windows emoji 字体，最后回退
        font = None
        for font_name, font_size in [("LiberationSans-Regular.ttf", size - 8),
                                      ("DejaVuSans.ttf", size - 8),
                                      ("seguiemj.ttf", size - 8),
                                      ("segoe ui emoji.ttf", size - 8),
                                      ("arial.ttf", size - 8)]:
            try:
                font = ImageFont.truetype(font_name, font_size)
                break
            except Exception:
                continue
        if font is None:
            font = ImageFont.load_default()
        bbox = draw.textbbox((0, 0), emoji, font=font)
        x = (size - (bbox[2] - bbox[0])) / 2 - bbox[0]
        y = (size - (bbox[3] - bbox[1])) / 2 - bbox[1]
        try:
            draw.text((x, y), emoji, font=font, embedded_color=True)
        except Exception:
            draw.text((x, y), emoji, font=font)
        png_path = config.DATA_DIR / ".gui_icon.png"
        ico_path = config.DATA_DIR / ".gui_icon.ico"
        img.save(png_path)
        # 生成可用于 Windows 任务栏的多尺寸 ICO
        sizes = [16, 24, 32, 48, 64, 128, 256]
        img.save(ico_path, format="ICO", sizes=[(s, s) for s in sizes])
        photo = tk.PhotoImage(file=str(png_path))
        return photo, ico_path
    except Exception:
        return None, None


# ═══════════════════════════════════════════════════════════════════════════════
# GUI components
# ═══════════════════════════════════════════════════════════════════════════════

class PlotPanel(ttk.Frame):
    """左侧图像列表 + 右侧 matplotlib 画布。"""

    def __init__(self, parent, app, plot_names, include_trend=False):
        super().__init__(parent)
        self.app = app
        self.plot_names = list(plot_names)
        self.include_trend = include_trend
        f = app.font_scale

        left = make_card(self)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 8))
        ttk.Label(left, text="图像列表", style="Section.TLabel"
                  ).pack(anchor=tk.W, padx=12, pady=(10, 6))
        lb_frame = tk.Frame(left, bg=COLOR_CARD)
        lb_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))
        self.listbox = tk.Listbox(
            lb_frame, activestyle="none", exportselection=False,
            font=(FONT_FAMILY, 10),
            bg=COLOR_CARD, fg=COLOR_TEXT, bd=0, highlightthickness=0,
            selectbackground=COLOR_SELECT, selectforeground="#FFFFFF",
            selectborderwidth=0, width=24)
        self.listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb = ttk.Scrollbar(lb_frame, orient=tk.VERTICAL,
                           command=self.listbox.yview)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.listbox.configure(yscrollcommand=sb.set)
        self.listbox.bind("<<ListboxSelect>>", self._on_select)

        ttk.Button(left, text="导出当前图像",
                   command=self._export_image
                   ).pack(side=tk.BOTTOM, fill=tk.X, padx=8, pady=(0, 4))
        ttk.Button(left, text="保存全部图像",
                   command=self._save_all_images
                   ).pack(side=tk.BOTTOM, fill=tk.X, padx=8, pady=(0, 4))
        ttk.Button(left, text="导出当前图像数据",
                   command=self._export_data
                   ).pack(side=tk.BOTTOM, fill=tk.X, padx=8, pady=(0, 4))
        ttk.Button(left, text="导出全部图像数据",
                   command=self._export_all_data
                   ).pack(side=tk.BOTTOM, fill=tk.X, padx=8, pady=(0, 8))

        right = make_card(self)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.fig = Figure(figsize=(7, 5), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.toolbar = NavigationToolbar2Tk(self.canvas, right)
        self.toolbar.update()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True,
                                         padx=4, pady=4)

        self._items = []

    def refresh(self):
        """根据当前运行记录刷新列表。"""
        run_id = self.app.current_run
        self.listbox.delete(0, tk.END)
        self._items = []
        if run_id:
            for name in available_plots(run_id, self.plot_names):
                self.listbox.insert(tk.END, f"  {FIGURES[name][1]}")
                self._items.append(name)
        if self.include_trend and self.app.records:
            self.listbox.insert(tk.END, "  实验趋势（速率 / 误码率 / 信噪比）")
            self._items.append(TREND_KEY)
        if self._items:
            self.listbox.selection_set(0)
            self._show(self._items[0])
        else:
            self.fig.clear()
            ax = self.fig.add_subplot(111)
            ax.text(0.5, 0.5, "当前实验没有此类数据",
                    ha="center", va="center", transform=ax.transAxes)
            self.canvas.draw_idle()

    def _on_select(self, _event):
        sel = self.listbox.curselection()
        if sel:
            self._show(self._items[sel[0]])

    def _show(self, name):
        self.fig.clear()
        try:
            if name == TREND_KEY:
                build_records_trend(self.fig, self.app.records, "实验趋势")
            else:
                builder, title = FIGURES[name]
                npz = run_data_dir(self.app.current_run) / f"{name}.npz"
                builder(self.fig, npz, title)
        except Exception as exc:  # 数据缺失/损坏时显示提示而不是崩溃
            self.fig.clear()
            ax = self.fig.add_subplot(111)
            ax.text(0.5, 0.5, f"绘图失败:\n{exc}", ha="center", va="center",
                    transform=ax.transAxes, color="red")
        self.canvas.draw_idle()

    def _safe_savefig(self, path: Path):
        """保存图像；若 bbox_inches='tight' 抛出 IndexError（例如某些
        matplotlib 版本上的比特/功率加载双 y 轴图），则回退为普通保存。"""
        try:
            self.fig.savefig(path, dpi=config.PLOT_DPI, bbox_inches="tight")
        except (IndexError, ValueError):
            self.fig.savefig(path, dpi=config.PLOT_DPI)

    def _export_image(self):
        """将选中的图像导出为 PNG；默认保存位置为 data/plots/<run_id>/。"""
        sel = self.listbox.curselection()
        if not sel:
            messagebox.showinfo("提示", "请先选择一个图像")
            return
        name = self._items[sel[0]]
        run_id = self.app.current_run
        if not run_id:
            messagebox.showinfo("提示", "当前未选择实验")
            return

        plot_dir = config.PLOT_DIR / run_id
        default_name = f"{name}.png"
        path = filedialog.asksaveasfilename(
            title="导出图像",
            initialdir=str(plot_dir),
            initialfile=default_name,
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("PDF", "*.pdf"),
                       ("SVG", "*.svg"), ("全部文件", "*.*")])
        if not path:
            return
        path = Path(path)
        if path.is_file():
            if not messagebox.askyesno("确认覆盖",
                                       f"文件已存在:\n{path}\n\n是否覆盖?"):
                return
        try:
            plot_dir.mkdir(parents=True, exist_ok=True)
            self._safe_savefig(path)
            # 保存绘图参数元数据
            self._save_plot_metadata(name, path)
            messagebox.showinfo("导出成功", f"已保存到:\n{path}")
        except Exception as exc:
            messagebox.showerror("导出失败", str(exc))

    def _save_all_images(self):
        """将当前运行的所有图像保存为 PNG 到 data/plots/<run_id>/。"""
        run_id = self.app.current_run
        if not run_id:
            messagebox.showinfo("提示", "当前未选择实验")
            return
        names = available_plots(run_id, self.plot_names)
        if not names:
            messagebox.showinfo("提示", "当前实验没有可保存的图像")
            return

        plot_dir = config.PLOT_DIR / run_id
        existing = sorted([p.name for p in plot_dir.glob("*.png")]) if plot_dir.is_dir() else []
        if existing:
            if not messagebox.askyesno(
                    "确认覆盖",
                    f"{plot_dir} 中已存在 {len(existing)} 张图像。\n\n"
                    f"是否覆盖?"):
                return

        saved = 0
        errors = []
        current_name = self._items[self.listbox.curselection()[0]] if self.listbox.curselection() else None
        try:
            plot_dir.mkdir(parents=True, exist_ok=True)
            for name in names:
                try:
                    self._show(name)
                    png_path = plot_dir / f"{name}.png"
                    self._safe_savefig(png_path)
                    self._save_plot_metadata(name, png_path)
                    saved += 1
                except Exception as exc:
                    errors.append(f"{name}: {exc}")
            # 恢复之前显示的图像
            if current_name in self._items:
                self._show(current_name)
            else:
                self._show(self._items[0])
        except Exception as exc:
            messagebox.showerror("保存失败", str(exc))
            return
        if errors:
            messagebox.showerror("部分保存失败", "\n".join(errors))
        else:
            messagebox.showinfo("保存成功",
                                f"已保存 {saved} 张图像到:\n{plot_dir}")

    def _save_plot_metadata(self, name, png_path: Path):
        """将当前图像的参数（数组名、形状、数据路径）写入 JSON。"""
        run_id = self.app.current_run
        npz = run_data_dir(run_id) / f"{name}.npz"
        params = {}
        if npz.is_file():
            try:
                data = np.load(npz)
                params["arrays"] = {
                    k: {"shape": list(data[k].shape),
                        "dtype": str(data[k].dtype)}
                    for k in data.files
                }
                params["data_path"] = str(npz)
            except Exception:
                pass
        metadata = {
            "run_id": run_id,
            "name": name,
            "title": FIGURES.get(name, (None, name))[1],
            "png_path": str(png_path),
            "parameters": params,
        }
        json_path = png_path.with_suffix(".json")
        try:
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
        except Exception:
            pass

    def _export_data(self):
        """Export raw data for the selected plot to Excel (each array in its own sheet)."""
        sel = self.listbox.curselection()
        if not sel:
            messagebox.showinfo("提示", "请先选择一个图形")
            return
        name = self._items[sel[0]]
        run_id = self.app.current_run
        if not run_id:
            messagebox.showinfo("提示", "当前未选择实验")
            return

        if name == TREND_KEY:
            # Export run-history trend as JSON
            default_name = f"trend_{run_id}.json"
            path = filedialog.asksaveasfilename(
                title="导出趋势数据",
                initialfile=default_name,
                defaultextension=".json",
                filetypes=[("JSON", "*.json"), ("所有文件", "*.*")])
            if not path:
                return
            try:
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(self.app.records, f, indent=2,
                              ensure_ascii=False, default=str)
                messagebox.showinfo("导出成功", f"已保存到:\n{path}")
            except Exception as exc:
                messagebox.showerror("导出失败", str(exc))
            return

        npz = run_data_dir(run_id) / f"{name}.npz"
        if not npz.is_file():
            messagebox.showerror("导出失败", f"未找到数据文件:\n{npz}")
            return
        default_name = f"{name}_{run_id}.xlsx"
        path = filedialog.asksaveasfilename(
            title="导出图表数据",
            initialfile=default_name,
            defaultextension=".xlsx",
            filetypes=[("Excel 文件", "*.xlsx"), ("所有文件", "*.*")])
        if not path:
            return
        try:
            _npz_to_xlsx(npz, Path(path))
            messagebox.showinfo("导出成功", f"已保存到:\n{path}")
        except Exception:
            tb = traceback.format_exc()
            print(tb)
            messagebox.showerror("导出失败", f"导出当前图表数据失败:\n{tb}")

    def _export_all_data(self):
        """将当前运行的全部图表数据导出到 Excel(每个数组一个工作表)。"""
        run_id = self.app.current_run
        if not run_id:
            messagebox.showinfo("提示", "当前未选择实验")
            return
        names = available_plots(run_id, self.plot_names)
        if not names:
            messagebox.showinfo("提示", "当前实验没有可导出的图表数据")
            return

        default_name = f"all_plots_{run_id}.xlsx"
        path = filedialog.asksaveasfilename(
            title="导出全部图表数据",
            initialfile=default_name,
            defaultextension=".xlsx",
            filetypes=[("Excel 文件", "*.xlsx"), ("所有文件", "*.*")])
        if not path:
            return
        try:
            if not _HAS_OPENPYXL:
                raise RuntimeError("缺少 openpyxl;请运行: pip install openpyxl")
            wb = openpyxl.Workbook()
            wb.remove(wb.active)
            for name in names:
                npz = run_data_dir(run_id) / f"{name}.npz"
                if not npz.is_file():
                    continue
                data = np.load(npz)
                for key in data.files:
                    arr = data[key]
                    if arr.ndim > 2:
                        arr = arr.reshape(arr.shape[0], -1)
                    sheet_name = _safe_sheet_name(f"{name}_{key}")
                    ws = wb.create_sheet(title=sheet_name)
                    ws.cell(row=1, column=1, value=f"图表: {name}")
                    ws.cell(row=1, column=2, value=f"数组: {key}")
                    ws.cell(row=1, column=3, value=f"形状: {arr.shape}")
                    _write_array_to_sheet(ws, arr, start_row=3, start_col=1)
            wb.save(path)
            messagebox.showinfo("导出成功", f"已保存到:\n{path}\n"
                                           f"已导出 {len(names)} 个图表的数据")
        except Exception:
            tb = traceback.format_exc()
            print(tb)
            messagebox.showerror("导出失败", f"导出全部图表数据失败:\n{tb}")


class ResultsPanel(ttk.Frame):
    """标签页 3: 上方为实验记录表,下方为所选运行的结果图表。"""

    COLUMNS = ("run_id", "time", "pilot", "vc", "nn", "est_rate",
               "final_rate", "ber", "ser", "snr")
    HEADINGS = {
        "run_id": ("实验 ID", 170),
        "time": ("时间", 150),
        "pilot": ("导频", 90),
        "vc": ("虚拟信道", 70),
        "nn": ("NN", 50),
        "est_rate": ("估计速率 (Gbps)", 110),
        "final_rate": ("最终速率 (Gbps)", 110),
        "ber": ("BER", 100),
        "ser": ("SER", 100),
        "snr": ("平均信噪比 (dB)", 100),
    }

    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app

        top = ttk.LabelFrame(self, text=" 传输实验记录(点击行切换实验) ")
        top.pack(fill=tk.X, padx=2, pady=(2, 8))
        tree_frame = tk.Frame(top, bg=COLOR_CARD)
        tree_frame.pack(fill=tk.X, padx=6, pady=6)
        self.tree = ttk.Treeview(tree_frame, columns=self.COLUMNS,
                                 show="headings", height=7)
        for col in self.COLUMNS:
            text, width = self.HEADINGS[col]
            self.tree.heading(col, text=text)
            self.tree.column(col, width=int(width * app.font_scale),
                             anchor=tk.CENTER)
        vsb = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL,
                            command=self.tree.yview)
        self.tree.configure(yscrollcommand=vsb.set)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree.pack(fill=tk.X, expand=True)
        self.tree.bind("<<TreeviewSelect>>", self._on_row_select)

        bottom = ttk.LabelFrame(self, text=" 当前实验结果 ")
        bottom.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
        self.plot_panel = PlotPanel(bottom, app, TAB_RESULTS,
                                    include_trend=True)
        self.plot_panel.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        self._iid_to_run = {}

    def refresh(self):
        self.tree.delete(*self.tree.get_children())
        self._iid_to_run = {}
        for rec in reversed(self.app.records):   # 最新在前
            run_id = rec.get("run_id", "")
            ts = rec.get("timestamp", "")[:19].replace("T", " ")
            row = (
                run_id, ts,
                rec.get("pilot_pattern", ""),
                "是" if rec.get("use_virtual_channel") else "否",
                "是" if rec.get("use_nn") else "否",
                f"{rec.get('estimated_rate_gbps', 0):.2f}",
                f"{rec.get('final_rate_gbps', 0):.2f}",
                f"{rec.get('final_ber', 0):.3e}",
                f"{rec.get('final_ser', 0):.3e}",
                f"{rec.get('mean_recovered_snr_db', 0):.2f}",
            )
            iid = self.tree.insert("", tk.END, values=row)
            self._iid_to_run[iid] = run_id
        for iid, rid in self._iid_to_run.items():
            if rid == self.app.current_run:
                self.tree.selection_set(iid)
                self.tree.see(iid)
                break
        self.plot_panel.refresh()

    def _on_row_select(self, _event):
        sel = self.tree.selection()
        if not sel:
            return
        run_id = self._iid_to_run.get(sel[0])
        if run_id and run_id != self.app.current_run:
            self.app.select_run(run_id, source="results")


# ═══════════════════════════════════════════════════════════════════════════════
# config.py 参数面板
# 类型: bool=0/1 或 True/False 开关; int=整数; num=数值(科学计数法/表达式);
#        str=字符串(必须加引号); ichoice=整数下拉框; schoice=字符串下拉框
# ═══════════════════════════════════════════════════════════════════════════════

PARAM_GROUPS = [
    ("运行模式", [
        ("OFFLINE_FLAG", "bool", None),
        ("USE_VIRTUAL_CHANNEL", "bool", None),
        ("POSTEQ_FLAG", "ichoice", ["0", "1", "2", "3"]),
        ("RANDOM_SEED", "int", None),
    ]),
    ("DMT 信号参数", [
        ("CARRIERNO", "int", None),
        ("ZEROPAD1", "int", None),
        ("UPSAMPLENO", "int", None),
        ("DATANO_QPSK", "int", None),
        ("DATANO_BPL", "int", None),
        ("TRAININGNO", "int", None),
        ("CP_RATIO", "num", None),
        ("RATIO", "int", None),
        ("NORMALIZE_FLAG", "ichoice", ["0", "1"]),
    ]),
    ("导频图样", [
        ("PILOT_PATTERN", "schoice", ["training_only", "comb", "mesh"]),
        ("PILOT_COMB_START", "int", None),
        ("PILOT_COMB_SPACING", "int", None),
        ("PILOT_MESH_START_FREQ", "int", None),
        ("PILOT_MESH_FREQ_SPACING", "int", None),
        ("PILOT_MESH_START_TIME", "int", None),
        ("PILOT_MESH_TIME_SPACING", "int", None),
    ]),
    ("预均衡", [
        ("PRE_EQU_FLAG", "ichoice", ["0", "1", "2", "3", "4"]),
        ("PRE_METHOD", "ichoice", ["0", "1", "2", "3", "4", "5"]),
        ("PRE_EQUAL_DB", "num", None),
        ("PRE_EQUAL_DB2", "num", None),
        ("HW_PRE_FBEGIN", "num", None),
        ("HW_PRE_ADB", "num", None),
        ("HW_PRE_FCEN_MHZ", "num", None),
        ("HW_PRE_FHALF_MHZ", "num", None),
        ("HW_PRE_FEND", "num", None),
        ("HW_PRE_R0", "num", None),
    ]),
    ("硬件 (AWG / 示波器)", [
        ("AWG_SAMPLE_RATE", "num", None),
        ("OSC_SAMPLE_RATE", "num", None),
        ("AWG_VPP", "num", None),
        ("AWG_OUTPUT_ROUTE", "schoice", ["DC", "AC", "DAC"]),
        ("M8190A_VISA_ADDR", "str", None),
        ("M8190A_PORT", "int", None),
        ("OSC_VISA_ADDR", "str", None),
        ("OSC_CHANNEL", "schoice", ["CHAN1", "CHAN2", "CHAN3", "CHAN4"]),
        ("OSC_TIMEBASE_SCALE", "num", None),
    ]),
    ("虚拟信道", [
        ("VIRTUAL_CHANNEL_FC", "num", None),
        ("VIRTUAL_CHANNEL_SNR_DB", "num", None),
        ("VIRTUAL_CHANNEL_NONLINEARITY", "num", None),
        ("VIRTUAL_CHANNEL_DELAY", "int", None),
        ("VIRTUAL_CHANNEL_ATTENUATION", "num", None),
    ]),
    ("绘图", [
        ("PLOT_SHOW", "bool", None),
        ("PLOT_SAVE", "bool", None),
        ("PLOT_DPI", "int", None),
    ]),
]


def _safe_eval(text):
    """安全求值:先尝试 literal_eval,再尝试不带内置函数的简单算术运算。"""
    try:
        return ast.literal_eval(text)
    except Exception:
        return eval(text, {"__builtins__": {}}, {})


def _find_assignment(text, name):
    """在 config.py 源码中查找 `NAME = rhs  # comment`,返回 (rhs, gap, comment)。

    gap 是 rhs 与行内注释之间原有的空白,用于写回时保持注释对齐。
    """
    m = re.search(rf"^{re.escape(name)}\s*=\s*(?P<mid>[^#\n]*)"
                  rf"(?P<comment>\#.*)?$", text, flags=re.M)
    if not m:
        return None, None, None
    mid = m.group("mid")
    rhs = mid.rstrip()
    gap = mid[len(rhs):] or "    "
    return rhs, gap, (m.group("comment") or "").strip()


class ConfigPanel(ttk.Frame):
    """可编辑的 config.py 参数编辑器:分组显示,仅将修改过的参数写回文件。"""

    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        f = app.font_scale
        self.params = []   # 每条记录: dict(name, type, choices, var, orig, comment, widget)

        # ── 头部按钮 ────────────────────────────────────────────────
        head = tk.Frame(self, bg=COLOR_CARD)
        head.pack(fill=tk.X, padx=8, pady=(8, 4))
        ttk.Button(head, text="💾 保存到 config.py", style="Accent.TButton",
                   command=self.save).pack(side=tk.LEFT)
        ttk.Button(head, text="⟳ 重新加载", command=self.reload
                   ).pack(side=tk.LEFT, padx=8)
        self.info_var = tk.StringVar(
            value="运行前请保存参数;仅将修改过的参数写回文件")
        ttk.Label(head, textvariable=self.info_var, style="DimCard.TLabel"
                  ).pack(side=tk.LEFT, padx=12)

        # ── 可滚动区域 ──────────────────────────────────────────────
        body = tk.Frame(self, bg=COLOR_CARD)
        body.pack(fill=tk.BOTH, expand=True, padx=4, pady=(0, 4))
        self._canvas = tk.Canvas(body, bg=COLOR_CARD, highlightthickness=0)
        vsb = ttk.Scrollbar(body, orient=tk.VERTICAL,
                            command=self._canvas.yview)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        self._canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self._canvas.configure(yscrollcommand=vsb.set)
        self._inner = tk.Frame(self._canvas, bg=COLOR_CARD)
        win = self._canvas.create_window((0, 0), window=self._inner,
                                         anchor="nw")
        self._inner.bind("<Configure>", lambda _e: self._canvas.configure(
            scrollregion=self._canvas.bbox("all")))
        self._canvas.bind("<Configure>", lambda e: self._canvas.itemconfigure(
            win, width=e.width))
        # 鼠标滚轮(仅在光标位于参数区域内时启用)
        self._canvas.bind("<Enter>", lambda _e: self._canvas.bind_all(
            "<MouseWheel>", self._on_wheel))
        self._canvas.bind("<Leave>", lambda _e: self._canvas.unbind_all(
            "<MouseWheel>"))

        self.reload()

    def _on_wheel(self, event):
        self._canvas.yview_scroll(int(-event.delta / 120), "units")

    # ── 读取 / 构建 ───────────────────────────────────────────────

    def reload(self):
        for child in self._inner.winfo_children():
            child.destroy()
        self.params = []
        if CONFIG_PY.is_file():
            with CONFIG_PY.open("r", encoding="utf-8", newline="") as fp:
                text = fp.read()
        else:
            text = ""
        f = self.app.font_scale
        mono = (FONT_MONO, 10)
        mono_bold = (FONT_MONO, 10, "bold")
        hint_font = (FONT_FAMILY, 9)

        for group, items in PARAM_GROUPS:
            gf = ttk.LabelFrame(self._inner, text=f" {group} ")
            gf.pack(fill=tk.X, padx=6, pady=4)
            for name, ptype, choices in items:
                rhs, gap, comment = _find_assignment(text, name)
                row = tk.Frame(gf, bg=COLOR_CARD)
                row.pack(fill=tk.X, padx=8, pady=2)
                tk.Label(row, text=name, font=mono_bold, bg=COLOR_CARD,
                         fg=COLOR_TEXT, anchor=tk.W,
                         width=24).pack(side=tk.LEFT)
                if rhs is None:
                    tk.Label(row, text="(在 config.py 中未找到)",
                             font=hint_font, bg=COLOR_CARD,
                             fg=COLOR_DANGER).pack(side=tk.LEFT)
                    continue
                p = {"name": name, "type": ptype, "choices": choices,
                     "orig": rhs, "gap": gap, "comment": comment}
                self._make_widget(row, p, mono)
                if comment:
                    tk.Label(row, text=comment.lstrip("# ").strip(),
                             font=hint_font, bg=COLOR_CARD,
                             fg=COLOR_TEXT_DIM, anchor=tk.W,
                             justify=tk.LEFT
                             ).pack(side=tk.LEFT, padx=10, fill=tk.X,
                                    expand=True)
                self.params.append(p)
        self.info_var.set(f"已读取 {len(self.params)} 个参数 ({CONFIG_PY.name})")

    def _make_widget(self, row, p, mono):
        f = self.app.font_scale
        ptype, rhs = p["type"], p["orig"]
        entry_kw = dict(font=mono, bg="#F8FAFC", fg=COLOR_TEXT, bd=0,
                        highlightthickness=1,
                        highlightbackground=COLOR_BORDER,
                        highlightcolor=COLOR_PRIMARY_HOVER)
        if ptype == "bool":
            var = tk.BooleanVar(value=bool(_safe_eval(rhs)))
            p["var"] = var
            p["widget"] = ttk.Checkbutton(row, variable=var)
        elif ptype in ("ichoice", "schoice"):
            var = tk.StringVar(value=str(_safe_eval(rhs)))
            p["var"] = var
            p["widget"] = ttk.Combobox(
                row, textvariable=var, values=p["choices"],
                state="readonly", width=16, font=mono)
        else:
            var = tk.StringVar(value=rhs)
            p["var"] = var
            width = 36 if ptype == "str" else 16
            p["widget"] = tk.Entry(row, textvariable=var,
                                   width=width, **entry_kw)
        p["widget"].pack(side=tk.LEFT)

    # ── 获取取值 / 写回 ────────────────────────────────────────────

    def _literal_for(self, p, errors):
        """将控件当前值转换为可写入 config.py 的字面量字符串。"""
        ptype, name = p["type"], p["name"]
        if ptype == "bool":
            one, zero = ("True", "False") if p["orig"] in ("True", "False") \
                else ("1", "0")
            return one if p["var"].get() else zero
        if ptype == "ichoice":
            return str(int(p["var"].get()))
        if ptype == "schoice":
            return f'"{p["var"].get()}"'
        raw = p["var"].get().strip()
        if not raw:
            errors.append(f"{name}: 不能为空")
            return None
        try:
            val = _safe_eval(raw)
        except Exception:
            errors.append(f"{name}: 无法解析 «{raw}»")
            return None
        if ptype == "int":
            if not isinstance(val, (int, float)) or int(val) != val:
                errors.append(f"{name}: 需要整数,得到 «{raw}»")
                return None
            return str(int(val))
        if ptype == "num":
            if not isinstance(val, (int, float)):
                errors.append(f"{name}: 需要数值,得到 «{raw}»")
                return None
            return raw
        if ptype == "str":
            if not isinstance(val, str):
                errors.append(f"{name}: 字符串必须加引号,例如 \"TCPIP0::…\"")
                return None
            return raw
        return raw

    def get_value(self, name):
        """返回某参数控件的当前值(以 Python 值表示)。"""
        for p in self.params:
            if p["name"] == name:
                lit = self._literal_for(p, [])
                return _safe_eval(lit) if lit is not None else None
        return None

    def set_value(self, name, value):
        for p in self.params:
            if p["name"] == name:
                if p["type"] == "bool":
                    p["var"].set(bool(int(value)))
                else:
                    p["var"].set(str(value))
                return

    def save(self, silent=False):
        if not CONFIG_PY.is_file():
            messagebox.showerror("未找到 config.py", str(CONFIG_PY))
            return False
        errors = []
        pending = []
        for p in self.params:
            lit = self._literal_for(p, errors)
            if lit is not None and lit != p["orig"]:
                pending.append((p, lit))
        if errors:
            messagebox.showerror("参数校验失败", "\n".join(errors))
            return False
        with CONFIG_PY.open("r", encoding="utf-8", newline="") as fp:
            text = fp.read()
        for p, lit in pending:
            pattern = rf"^{re.escape(p['name'])}\s*=.*$"
            repl = f"{p['name']} = {lit}"
            if p["comment"]:
                repl += f"{p.get('gap') or '    '}{p['comment']}"
            text, n = re.subn(pattern, repl, text, count=1, flags=re.M)
            if n == 0:
                errors.append(p["name"])
            p["orig"] = lit
        if errors:
            messagebox.showerror("写回失败",
                                 "以下参数未能在 config.py 中找到: "
                                 + ", ".join(errors))
            return False
        with CONFIG_PY.open("w", encoding="utf-8", newline="") as fp:
            fp.write(text)
        msg = f"已将 {len(pending)} 个参数保存到 config.py" if pending \
            else "无改动,config.py 未变化"
        self.info_var.set(msg)
        if not silent and pending:
            messagebox.showinfo("保存成功", msg)
        return True


class RunPanel(ttk.Frame):
    """标签页 4: 通过 GUI 调用 main.py 运行实验。"""

    MODES = {
        "online":  ("在线 (AWG + 示波器)", {"--offline": "0", "--use-awg": "1",
                                                     "--use-virtual-channel": "0"}),
        "offline": ("离线 (读取已有的 RX 采集)", {"--offline": "1", "--use-awg": "0",
                                                           "--use-virtual-channel": "0"}),
        "virtual": ("离线 + 虚拟信道 (无硬件)", {"--offline": "1", "--use-awg": "0",
                                                                "--use-virtual-channel": "1"}),
    }
    STEPS = {
        "all": "完整流程 (探测 + Bitloading)",
        "step1": "step1 生成 QPSK 探测 TX",
        "step2": "step2 接收 QPSK 并估计信噪比",
        "step3": "step3 生成 Bitloading TX",
        "step4": "step4 接收 Bitloading 并解调",
    }

    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        self.proc = None
        self._queue = queue.Queue()
        f = app.font_scale

        # ── 参数卡片 ───────────────────────────────────────────────
        opt = ttk.LabelFrame(self, text=" 实验参数 ")
        opt.pack(fill=tk.X, padx=2, pady=(2, 8))

        ttk.Label(opt, text="运行模式:", style="Card.TLabel"
                  ).grid(row=0, column=0, sticky=tk.W, padx=12, pady=(10, 4))
        self.mode_var = tk.StringVar(value="virtual")
        col = 1
        for key, (label, _args) in self.MODES.items():
            ttk.Radiobutton(opt, text=label, value=key,
                            variable=self.mode_var
                            ).grid(row=0, column=col, sticky=tk.W,
                                   padx=(4, 16), pady=(10, 4))
            col += 1

        ttk.Label(opt, text="流程步骤:", style="Card.TLabel"
                  ).grid(row=1, column=0, sticky=tk.W, padx=12, pady=4)
        self.step_var = tk.StringVar(value="all")
        self.step_combo = ttk.Combobox(
            opt, state="readonly", width=34,
            values=[f"{k}  {v}" for k, v in self.STEPS.items()],
            font=(FONT_FAMILY, 10))
        self.step_combo.current(0)
        self.step_combo.grid(row=1, column=1, columnspan=2, sticky=tk.W,
                             padx=(4, 16), pady=4)

        self.use_nn_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(opt, text="启用 NN 后均衡器 (--use-nn, 首次运行需要训练)",
                        variable=self.use_nn_var
                        ).grid(row=2, column=1, columnspan=3, sticky=tk.W,
                               padx=(4, 16), pady=(4, 10))

        # ── 离线定位选项 ─────────────────────────────────────────────
        offline_frame = ttk.LabelFrame(opt, text=" 离线定位选项(可选) ")
        offline_frame.grid(row=3, column=0, columnspan=5, sticky=tk.EW,
                           padx=12, pady=(0, 8))

        ttk.Label(offline_frame, text="STEP2 波形 ID 的后 6 位:",
                  style="Card.TLabel"
                  ).grid(row=0, column=0, sticky=tk.W, padx=(8, 4), pady=(8, 4))
        self.qpsk_suffix_var = tk.StringVar(value="")
        ttk.Entry(offline_frame, textvariable=self.qpsk_suffix_var, width=14
                  ).grid(row=0, column=1, sticky=tk.W, padx=4, pady=(8, 4))
        ttk.Label(offline_frame, text="留空则使用最新的 QPSK RX 文件",
                  style="DimCard.TLabel"
                  ).grid(row=0, column=2, sticky=tk.W, padx=4, pady=(8, 4))

        ttk.Label(offline_frame, text="STEP4 波形 ID 的后 6 位:",
                  style="Card.TLabel"
                  ).grid(row=1, column=0, sticky=tk.W, padx=(8, 4), pady=4)
        self.bpl_suffix_var = tk.StringVar(value="")
        ttk.Entry(offline_frame, textvariable=self.bpl_suffix_var, width=14
                  ).grid(row=1, column=1, sticky=tk.W, padx=4, pady=4)
        ttk.Label(offline_frame, text="留空则使用最新的 Bitloading RX 文件",
                  style="DimCard.TLabel"
                  ).grid(row=1, column=2, sticky=tk.W, padx=4, pady=4)

        ttk.Label(offline_frame, text="完整 run-id / 文件名:",
                  style="Card.TLabel"
                  ).grid(row=2, column=0, sticky=tk.W, padx=(8, 4), pady=(0, 8))
        self.full_run_id_var = tk.StringVar(value="")
        ttk.Entry(offline_frame, textvariable=self.full_run_id_var, width=36
                  ).grid(row=2, column=1, sticky=tk.W, padx=4, pady=(0, 8))
        ttk.Label(offline_frame, text="粘贴完整 id/文件名以自动识别阶段并去除扩展名",
                  style="DimCard.TLabel"
                  ).grid(row=2, column=2, sticky=tk.W, padx=4, pady=(0, 8))

        self.full_run_id_var.trace_add("write", self._on_full_run_id_change)

        btn_bar = tk.Frame(opt, bg=COLOR_CARD)
        btn_bar.grid(row=4, column=0, columnspan=5, sticky=tk.W,
                     padx=12, pady=(0, 8))
        self.run_btn = ttk.Button(btn_bar, text="▶  开始测试",
                                  style="Accent.TButton",
                                  command=self.start_run)
        self.run_btn.pack(side=tk.LEFT, padx=(0, 8))
        self.stop_btn = ttk.Button(btn_bar, text="■  停止",
                                   style="Danger.TButton",
                                   command=self.stop_run, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT)
        self.run_status = ttk.Label(btn_bar, text="就绪", style="DimCard.TLabel")
        self.run_status.pack(side=tk.LEFT, padx=16)

        # ── NN 进度条(训练/预测期间显示,避免刷屏日志) ──
        self.progress_frame = tk.Frame(opt, bg=COLOR_CARD)
        self.progress_frame.grid(row=5, column=0, columnspan=5,
                                 sticky=tk.W, padx=12, pady=(0, 10))
        self.progress_frame.grid_remove()
        self.progress_var = tk.DoubleVar(value=0.0)
        self.progress_bar = ttk.Progressbar(
            self.progress_frame, variable=self.progress_var,
            maximum=100.0, length=220)
        self.progress_bar.pack(side=tk.LEFT)
        self.nn_status = ttk.Label(self.progress_frame, text="",
                                   style="Card.TLabel")
        self.nn_status.pack(side=tk.LEFT, padx=(10, 0))

        # ── 底部:左侧 config 参数面板 + 右侧运行日志 ──
        paned = ttk.Panedwindow(self, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
        paned.after_idle(lambda: paned.sashpos(
            0, int(paned.winfo_width() * 0.56)))

        cfg_card = ttk.LabelFrame(paned, text=" config.py 实验参数 ")
        paned.add(cfg_card, weight=3)
        self.config_panel = ConfigPanel(cfg_card, app)
        self.config_panel.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        # ── 日志卡片 ────────────────────────────────────────────────
        log_card = ttk.LabelFrame(paned, text=" 运行日志 ")
        paned.add(log_card, weight=2)
        log_frame = tk.Frame(log_card, bg=COLOR_CARD)
        log_frame.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)
        self.log_text = tk.Text(
            log_frame, wrap=tk.NONE, state=tk.DISABLED, width=40,
            font=(FONT_MONO, 9),
            bg="#0F172A", fg="#E2E8F0", bd=0, highlightthickness=0,
            insertbackground="#E2E8F0")
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        lsb = ttk.Scrollbar(log_frame, orient=tk.VERTICAL,
                            command=self.log_text.yview)
        lsb.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.configure(yscrollcommand=lsb.set)
        self.log_text.tag_configure("head", foreground="#22D3EE")
        self.log_text.tag_configure("err", foreground="#F87171")

        self._append_log(
            "提示:选择运行模式和步骤,然后点击“开始测试”。\n"
            "在线模式需要 M8190A 和示波器;未连接硬件时请使用虚拟信道模式。\n"
            "在离线/虚拟模式下,可以填写 STEP2 / STEP4 波形 ID 的后 6 位,\n"
            "或粘贴完整 run-id / 文件名以自动识别阶段并去除扩展名。\n"
            "测试完成后数据会自动刷新并切换到最新实验。\n", "head")

    # ── 运行控制 ────────────────────────────────────────────────

    def build_command(self):
        mode = self.mode_var.get()
        args = self.MODES[mode][1]
        step = self.step_combo.get().split()[0] or "all"
        cmd = [sys.executable, str(MAIN_PY)]
        for k, v in args.items():
            cmd += [k, v]
        cmd += ["--use-nn", "1" if self.use_nn_var.get() else "0",
                "--step", step]
        # 在离线/虚拟模式下追加定位参数
        if mode in ("offline", "virtual"):
            qpsk_suffix = self.qpsk_suffix_var.get().strip()
            bpl_suffix = self.bpl_suffix_var.get().strip()
            if qpsk_suffix and step in ("all", "step2"):
                cmd += ["--run-suffix-qpsk", qpsk_suffix]
            if bpl_suffix and step in ("all", "step4"):
                cmd += ["--run-suffix-bpl", bpl_suffix]
        return cmd

    def _on_full_run_id_change(self, *args):
        """从完整 run-id 或文件名自动识别阶段、去除扩展名,并填入对应的输入框。"""
        full = self.full_run_id_var.get().strip()
        if not full:
            return
        # 去除常见文件扩展名
        for ext in (".txt", ".json", ".npz", ".png", ".xlsx", ".mat"):
            if full.lower().endswith(ext):
                full = full[:-len(ext)]
                break
        # 取最后一段作为后缀
        suffix = full.split("_")[-1] if "_" in full else full
        if not suffix or len(suffix) < 4:
            return
        # 识别阶段前缀
        upper = full.upper()
        is_qpsk = "QPSK" in upper or "SNREST" in upper
        is_bpl = "DMT" in upper and not is_qpsk
        if is_qpsk:
            if self.qpsk_suffix_var.get() != suffix:
                self.qpsk_suffix_var.set(suffix)
        elif is_bpl:
            if self.bpl_suffix_var.get() != suffix:
                self.bpl_suffix_var.set(suffix)
        else:
            # When stage cannot be detected, fill both boxes (handy for one-click full pipeline)
            if self.qpsk_suffix_var.get() != suffix:
                self.qpsk_suffix_var.set(suffix)
            if self.bpl_suffix_var.get() != suffix:
                self.bpl_suffix_var.set(suffix)

    def start_run(self):
        if self.proc is not None:
            return
        if not MAIN_PY.is_file():
            messagebox.showerror("未找到 main.py", f"Could not find {MAIN_PY}")
            return
        # Write config panel changes back to config.py first (abort if validation fails)
        if not self.config_panel.save(silent=True):
            return
        # In main.py, AWG download in step1/step3 is controlled by config.OFFLINE_FLAG,
        # which may differ from the GUI mode; check early to avoid accidental hardware access
        mode = self.mode_var.get()
        off_flag = self.config_panel.get_value("OFFLINE_FLAG")
        if mode in ("offline", "virtual") and off_flag == 0:
            if messagebox.askyesno(
                    "检测到 OFFLINE_FLAG=0",
                    "config.py 中 OFFLINE_FLAG=0：运行期间 step1/step3 仍会向 AWG 下载波形。"
                    "\n\n运行前是否将 OFFLINE_FLAG 设为 1？\n"
                    "（选择“否”取消本次运行）"):
                self.config_panel.set_value("OFFLINE_FLAG", "1")
                self.config_panel.save(silent=True)
            else:
                return
        elif mode == "online" and off_flag not in (0, None):
            if not messagebox.askyesno(
                    "检测到 OFFLINE_FLAG=1",
                    "在线模式但 config.py 中 OFFLINE_FLAG=1：step1/step3 不会向 "
                    "AWG 下载波形，示波器可能采不到信号。\n\n仍要继续吗？"):
                return
        cmd = self.build_command()

        # Generate run_id for this run; if a suffix was specified in offline mode,
        # prefer the source RX file's own run_id to keep numbering consistent
        run_id = generate_run_id()
        step = self.step_combo.get().split()[0] or "all"
        qpsk_suffix = self.qpsk_suffix_var.get().strip()
        bpl_suffix = self.bpl_suffix_var.get().strip()
        if self.mode_var.get() in ("offline", "virtual"):
            if step == "step2" and qpsk_suffix:
                src_run_id = _resolve_source_run_id("step2", qpsk_suffix)
                if src_run_id:
                    run_id = src_run_id
            elif step == "step4" and bpl_suffix:
                src_run_id = _resolve_source_run_id("step4", bpl_suffix)
                if src_run_id:
                    run_id = src_run_id
            elif step == "all" and bpl_suffix:
                # Full pipeline uses the bitloading file's run_id
                src_run_id = _resolve_source_run_id("step4", bpl_suffix)
                if src_run_id:
                    run_id = src_run_id
        plot_save = self.config_panel.get_value("PLOT_SAVE")
        if plot_save:
            plot_dir = config.PLOT_DIR / run_id
            if plot_dir.is_dir() and any(plot_dir.glob("*.png")):
                if not messagebox.askyesno(
                        "确认覆盖",
                        f"Images for this run-id already exist in {plot_dir}.\n\n"
                        f"Overwrite?"):
                    return
        cmd += ["--run-id", run_id]

        mode_label = self.MODES[self.mode_var.get()][0]
        if self.mode_var.get() == "online":
            if not messagebox.askyesno(
                    "确认在线模式",
                    "在线模式将控制 M8190A 和示波器，确认硬件已连接？"):
                return
        env = dict(os.environ)
        env["MPLBACKEND"] = "Agg"        # suppress matplotlib windows in subprocess
        env["PYTHONUNBUFFERED"] = "1"
        try:
            self.proc = subprocess.Popen(
                cmd, cwd=str(PROJECT_ROOT), env=env,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace", bufsize=1)
        except Exception as exc:
            messagebox.showerror("启动失败", str(exc))
            self.proc = None
            return
        self._append_log(f"\n$ {' '.join(cmd)}\n", "head")
        self._append_log(f"Mode: {mode_label}\n", "head")
        self.run_btn.configure(state=tk.DISABLED)
        self.stop_btn.configure(state=tk.NORMAL)
        self.run_status.configure(text="运行中…")
        self.progress_var.set(0.0)
        self.progress_frame.grid_remove()
        self.nn_status.configure(text="")
        self.app.set_running(True)
        threading.Thread(target=self._reader, daemon=True).start()
        self.after(100, self._poll)

    def _reader(self):
        assert self.proc and self.proc.stdout
        for line in self.proc.stdout:
            self._queue.put(("line", line))
        code = self.proc.wait()
        self._queue.put(("done", code))

    def _poll(self):
        try:
            while True:
                kind, payload = self._queue.get_nowait()
                if kind == "line":
                    line = payload.rstrip("\n")
                    if line.startswith("[NN_PROGRESS]"):
                        self._update_nn_progress(line)
                        continue
                    tag = "err" if ("Error" in payload or "Traceback"
                                    in payload) else None
                    self._append_log(payload, tag)
                else:
                    self._on_done(payload)
                    return
        except queue.Empty:
            pass
        if self.proc is not None:
            self.after(100, self._poll)

    def _update_nn_progress(self, line):
        """Parse the [NN_PROGRESS] marker from NN output and update the progress bar (not the log)."""
        try:
            payload = json.loads(line.split("[NN_PROGRESS]", 1)[1].strip())
        except Exception:
            return
        phase = payload.get("phase", "train")
        if phase == "train":
            epoch = payload.get("epoch", 0)
            total = payload.get("total", 1) or 1
            train_loss = payload.get("train_loss")
            val_loss = payload.get("val_loss")
            self.progress_var.set(min(100.0, epoch / total * 100.0))
            parts = [f"NN training… Epoch {epoch}/{total}"]
            if train_loss is not None:
                parts.append(f"train_loss={train_loss:.4f}")
            if val_loss is not None:
                parts.append(f"val_loss={val_loss:.4f}")
            self.nn_status.configure(text="  ｜  ".join(parts))
            self.progress_frame.grid()
        elif phase == "predict":
            self.progress_var.set(0.0)
            self.nn_status.configure(text="NN 预测中…")
            self.progress_frame.grid()
        elif phase == "done":
            self.progress_var.set(100.0)
            self.nn_status.configure(text="NN 均衡完成")
            self.progress_frame.grid()

    def _on_done(self, code):
        self.proc = None
        self.run_btn.configure(state=tk.NORMAL)
        self.stop_btn.configure(state=tk.DISABLED)
        self.app.set_running(False)
        if code == 0:
            self.run_status.configure(text="测试完成 ✔")
            self._append_log("\n========== Test Completed ==========\n", "head")
            self.app.reload_data(select_latest=True)
        else:
            self.run_status.configure(text=f"Abnormal exit (code={code})")
            self._append_log(f"\nProcess exited with code {code}\n", "err")

    def stop_run(self):
        if self.proc is not None:
            self.proc.terminate()
            self._append_log("\n(Process stop requested…)\n", "err")

    def _append_log(self, text, tag=None):
        self.log_text.configure(state=tk.NORMAL)
        if tag:
            self.log_text.insert(tk.END, text, tag)
        else:
            self.log_text.insert(tk.END, text)
        self.log_text.see(tk.END)
        self.log_text.configure(state=tk.DISABLED)


# ═══════════════════════════════════════════════════════════════════════════════
# Tab 5: Keithley 2400 source meter control
# ═══════════════════════════════════════════════════════════════════════════════

class Keithley2400Panel(ttk.Frame):
    """GUI panel for controlling a Keithley 2400 over RS-232 or USB-to-RS232."""

    SOURCE_MODES = ["voltage", "current"]

    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        self.instrument: Optional[Keithley2400] = None

        # Persistent settings mirror the config module defaults
        self.port_var = tk.StringVar(value=config.K2400_PORT)
        self.source_mode_var = tk.StringVar(value=config.K2400_SOURCE_MODE)
        self.level_var = tk.StringVar(value=str(config.K2400_LEVEL))
        self.compliance_var = tk.StringVar(value=str(config.K2400_COMPLIANCE))
        self.nplc_var = tk.StringVar(value=str(config.K2400_NPLC))
        self.auto_range_var = tk.BooleanVar(value=True)
        self.output_var = tk.BooleanVar(value=False)

        self._build_ui()
        self._refresh_ports()

    # --- UI construction ----------------------------------------------------
    def _build_ui(self):
        self.configure(style="TFrame")
        canvas = tk.Canvas(self, bg=COLOR_BG, highlightthickness=0)
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)
        scrollable = ttk.Frame(canvas)
        scrollable.bind("<Configure>", lambda _e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Header
        hdr = tk.Frame(scrollable, bg=COLOR_BG)
        hdr.pack(fill=tk.X, padx=16, pady=(14, 6))
        ttk.Label(hdr, text="Keithley 2400 源表控制",
                  style="Title.TLabel").pack(side=tk.LEFT)
        self.status_lbl = ttk.Label(hdr, text="未连接", style="Pill.TLabel")
        self.status_lbl.pack(side=tk.RIGHT)

        # Connection card
        card = make_card(scrollable)
        card.pack(fill=tk.X, padx=16, pady=6)
        inner = tk.Frame(card, bg=COLOR_CARD)
        inner.pack(fill=tk.X, padx=12, pady=12)

        ttk.Label(inner, text="COM 端口", style="Section.TLabel").grid(
            row=0, column=0, sticky=tk.W, padx=(0, 8), pady=4)
        self.port_combo = ttk.Combobox(inner, textvariable=self.port_var,
                                       values=[], width=40, state="readonly")
        self.port_combo.grid(row=0, column=1, sticky=tk.W, padx=(0, 8), pady=4)
        ttk.Button(inner, text="⟳ 刷新", command=self._refresh_ports).grid(
            row=0, column=2, padx=(0, 8), pady=4)

        ttk.Label(inner, text="波特率", style="Section.TLabel").grid(
            row=1, column=0, sticky=tk.W, padx=(0, 8), pady=4)
        self.baud_combo = ttk.Combobox(inner, values=[9600, 19200, 38400, 57600, 115200],
                                       width=12, state="readonly")
        self.baud_combo.set(str(config.K2400_BAUDRATE))
        self.baud_combo.grid(row=1, column=1, sticky=tk.W, padx=(0, 8), pady=4)

        self.conn_btn = ttk.Button(inner, text="连接", command=self._toggle_connect)
        self.conn_btn.grid(row=1, column=2, padx=(0, 8), pady=4)

        # Source settings card
        card2 = make_card(scrollable)
        card2.pack(fill=tk.X, padx=16, pady=6)
        inner2 = tk.Frame(card2, bg=COLOR_CARD)
        inner2.pack(fill=tk.X, padx=12, pady=12)

        ttk.Label(inner2, text="源模式", style="Section.TLabel").grid(
            row=0, column=0, sticky=tk.W, padx=(0, 8), pady=4)
        self.mode_combo = ttk.Combobox(inner2, textvariable=self.source_mode_var,
                                       values=self.SOURCE_MODES, state="readonly", width=14)
        self.mode_combo.bind("<<ComboboxSelected>>", self._on_mode_change)
        self.mode_combo.grid(row=0, column=1, sticky=tk.W, padx=(0, 8), pady=4)

        ttk.Label(inner2, text="电平", style="Section.TLabel").grid(
            row=1, column=0, sticky=tk.W, padx=(0, 8), pady=4)
        self.level_entry = ttk.Entry(inner2, textvariable=self.level_var, width=14)
        self.level_entry.grid(row=1, column=1, sticky=tk.W, padx=(0, 8), pady=4)
        self.level_unit_lbl = ttk.Label(inner2, text="V", style="Section.TLabel")
        self.level_unit_lbl.grid(row=1, column=2, sticky=tk.W, pady=4)

        ttk.Label(inner2, text="限值", style="Section.TLabel").grid(
            row=2, column=0, sticky=tk.W, padx=(0, 8), pady=4)
        self.comp_entry = ttk.Entry(inner2, textvariable=self.compliance_var, width=14)
        self.comp_entry.grid(row=2, column=1, sticky=tk.W, padx=(0, 8), pady=4)
        self.comp_unit_lbl = ttk.Label(inner2, text="A", style="Section.TLabel")
        self.comp_unit_lbl.grid(row=2, column=2, sticky=tk.W, pady=4)

        ttk.Label(inner2, text="NPLC", style="Section.TLabel").grid(
            row=3, column=0, sticky=tk.W, padx=(0, 8), pady=4)
        self.nplc_entry = ttk.Entry(inner2, textvariable=self.nplc_var, width=14)
        self.nplc_entry.grid(row=3, column=1, sticky=tk.W, padx=(0, 8), pady=4)

        ttk.Checkbutton(inner2, text="自动量程", variable=self.auto_range_var).grid(
            row=4, column=1, sticky=tk.W, padx=(0, 8), pady=4)

        ttk.Button(inner2, text="应用设置", command=self._apply_settings).grid(
            row=5, column=1, sticky=tk.W, padx=(0, 8), pady=(12, 0))

        # Output / measure card
        card3 = make_card(scrollable)
        card3.pack(fill=tk.X, padx=16, pady=6)
        inner3 = tk.Frame(card3, bg=COLOR_CARD)
        inner3.pack(fill=tk.X, padx=12, pady=12)

        self.out_btn = ttk.Button(inner3, text="输出开", command=self._toggle_output)
        self.out_btn.grid(row=0, column=0, padx=(0, 8), pady=4)
        ttk.Button(inner3, text="测量", command=self._measure).grid(
            row=0, column=1, padx=(0, 8), pady=4)
        ttk.Button(inner3, text="复位仪器", command=self._reset).grid(
            row=0, column=2, padx=(0, 8), pady=4)

        self.last_measure_lbl = ttk.Label(
            inner3, text="上次测量: --", style="Section.TLabel")
        self.last_measure_lbl.grid(row=1, column=0, columnspan=4, sticky=tk.W, pady=(12, 0))

        # Log card
        card4 = make_card(scrollable)
        card4.pack(fill=tk.BOTH, expand=True, padx=16, pady=6)
        inner4 = tk.Frame(card4, bg=COLOR_CARD)
        inner4.pack(fill=tk.BOTH, expand=True, padx=12, pady=12)
        ttk.Label(inner4, text="通信日志", style="Section.TLabel").pack(anchor=tk.W)
        self.log_text = tk.Text(inner4, height=12, wrap=tk.WORD, font=(FONT_MONO, 9),
                                bg="#FAFAFA", fg=COLOR_TEXT, relief=tk.FLAT,
                                highlightbackground=COLOR_BORDER, highlightthickness=1)
        self.log_text.pack(fill=tk.BOTH, expand=True, pady=(6, 0))
        self.log_text.configure(state=tk.DISABLED)

        self._on_mode_change()

    # --- Helpers ------------------------------------------------------------
    def _log(self, text: str, tag: Optional[str] = None):
        self.log_text.configure(state=tk.NORMAL)
        if tag:
            self.log_text.insert(tk.END, text + "\n", tag)
        else:
            self.log_text.insert(tk.END, text + "\n")
        self.log_text.see(tk.END)
        self.log_text.configure(state=tk.DISABLED)

    def _refresh_ports(self):
        ports = refresh_port_list()
        self.port_combo["values"] = ports
        if ports and not self.port_var.get():
            self.port_var.set(parse_port_entry(ports[0]))
        elif ports:
            current = self.port_var.get()
            matching = [p for p in ports if parse_port_entry(p) == current]
            if not matching:
                self.port_var.set(parse_port_entry(ports[0]))

    def _on_mode_change(self, _event=None):
        mode = self.source_mode_var.get()
        if mode == "voltage":
            self.level_unit_lbl.configure(text="V")
            self.comp_unit_lbl.configure(text="A")
        else:
            self.level_unit_lbl.configure(text="A")
            self.comp_unit_lbl.configure(text="V")

    # --- Instrument control -------------------------------------------------
    def _toggle_connect(self):
        if self.instrument is not None and self.instrument.connected:
            self._disconnect()
        else:
            self._connect()

    def _connect(self):
        port = self.port_var.get()
        if not port:
            messagebox.showwarning("未选择端口", "请先选择一个 COM 端口。")
            return
        baud = int(self.baud_combo.get() or config.K2400_BAUDRATE)
        try:
            self.instrument = Keithley2400(port=port, baudrate=baud,
                                           timeout=config.K2400_TIMEOUT)
            self.instrument.connect()
            self.conn_btn.configure(text="断开连接")
            self.status_lbl.configure(text=f"已连接 ({port})")
            self._log(f"已连接到 {port}，波特率 {baud}")
            self._apply_settings()
        except K2400ConnectionError as exc:
            messagebox.showerror("连接失败", str(exc))
            self._log(f"连接失败: {exc}", "err")
            self.instrument = None
        except Exception as exc:
            messagebox.showerror("连接错误", str(exc))
            self._log(f"连接错误: {exc}", "err")
            self.instrument = None

    def _disconnect(self):
        if self.instrument is not None:
            try:
                self.instrument.disconnect()
            except Exception as exc:
                self._log(f"断开连接出错: {exc}", "err")
            finally:
                self.instrument = None
        self.conn_btn.configure(text="连接")
        self.status_lbl.configure(text="已断开")
        self.output_var.set(False)
        self.out_btn.configure(text="输出 ON")
        self._log("已断开")

    def _apply_settings(self):
        if self.instrument is None or not self.instrument.connected:
            self._log("未连接，设置未应用。")
            return
        try:
            mode = self.source_mode_var.get()
            level = float(self.level_var.get())
            compliance = float(self.compliance_var.get())
            nplc = float(self.nplc_var.get())
            auto_range = self.auto_range_var.get()

            self.instrument.set_source_mode(mode)
            self.instrument.set_compliance(compliance)
            self.instrument.set_nplc(nplc)
            self.instrument.set_range(auto=auto_range)
            self.instrument.set_output_level(level)

            self._log(f"设置已应用: {mode} 源, level={level}, "
                      f"compliance={compliance}, NPLC={nplc}, auto_range={auto_range}")
        except ValueError:
            messagebox.showerror("数值无效", "电平、限流值和 NPLC 必须为数字。")
        except K2400ConfigError as exc:
            messagebox.showerror("配置错误", str(exc))
        except Exception as exc:
            messagebox.showerror("应用设置失败", str(exc))
            self._log(f"应用设置失败: {exc}", "err")

    def _toggle_output(self):
        if self.instrument is None or not self.instrument.connected:
            self._log("未连接。")
            return
        try:
            if self.output_var.get():
                self.instrument.output_off()
                self.output_var.set(False)
                self.out_btn.configure(text="输出 ON")
                self._log("输出 OFF")
            else:
                self.instrument.output_on()
                self.output_var.set(True)
                self.out_btn.configure(text="输出 OFF")
                self._log("输出 ON")
        except Exception as exc:
            messagebox.showerror("输出控制失败", str(exc))
            self._log(f"输出控制失败: {exc}", "err")

    def _measure(self):
        if self.instrument is None or not self.instrument.connected:
            self._log("未连接。")
            return
        try:
            data = self.instrument.measure()
            text = (f"V={data['voltage']:.6e} V, "
                    f"I={data['current']:.6e} A, "
                    f"R={data['resistance']:.6e} Ω, "
                    f"t={data['timestamp']:.6f} s")
            self.last_measure_lbl.configure(text=f"上次测量: {text}")
            self._log(f"测量: {text}")
        except K2400CommandError as exc:
            messagebox.showerror("测量失败", str(exc))
            self._log(f"测量失败: {exc}", "err")
        except Exception as exc:
            messagebox.showerror("测量错误", str(exc))
            self._log(f"测量错误: {exc}", "err")

    def _reset(self):
        if self.instrument is None or not self.instrument.connected:
            self._log("未连接。")
            return
        try:
            self.instrument.reset()
            self._log("仪器已复位。")
            self._apply_settings()
        except Exception as exc:
            messagebox.showerror("复位失败", str(exc))
            self._log(f"复位失败: {exc}", "err")

    def on_close(self):
        """应用退出时调用，以安全地关闭输出。"""
        self._disconnect()


# ═══════════════════════════════════════════════════════════════════════════════
# 标签页 6: 网格扫描（偏置 vs Vpp）
# ═══════════════════════════════════════════════════════════════════════════════

class GridScanPanel(ttk.Frame):
    """用于自动化二维参数扫描（Keithley 偏置 vs AWG Vpp）的 GUI 面板。"""

    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        self.scanner: Optional[GridScanner] = None
        self.scan_thread: Optional[threading.Thread] = None
        self.stop_requested = False

        # 配置变量
        self.p1_name_var = tk.StringVar(value=config.GRID_SCAN_PARAM1_NAME)
        self.p1_mode_var = tk.StringVar(value=config.GRID_SCAN_PARAM1_MODE)
        self.p1_start_var = tk.StringVar(value=str(config.GRID_SCAN_PARAM1_START))
        self.p1_stop_var = tk.StringVar(value=str(config.GRID_SCAN_PARAM1_STOP))
        self.p1_step_var = tk.StringVar(value=str(config.GRID_SCAN_PARAM1_STEP))
        self.vpp_start_var = tk.StringVar(value=str(config.GRID_SCAN_VPP_START))
        self.vpp_stop_var = tk.StringVar(value=str(config.GRID_SCAN_VPP_STOP))
        self.vpp_step_var = tk.StringVar(value=str(config.GRID_SCAN_VPP_STEP))
        self.run_mode_var = tk.StringVar(value=config.GRID_SCAN_RUN_MODE)
        self.repeats_var = tk.StringVar(value=str(config.GRID_SCAN_REPEATS))
        self.offline_var = tk.BooleanVar(value=bool(config.OFFLINE_FLAG))
        self.use_virtual_channel_var = tk.BooleanVar(value=bool(config.USE_VIRTUAL_CHANNEL))
        self.use_nn_var = tk.BooleanVar(value=bool(config.USE_NN))
        self.port_var = tk.StringVar(value=config.K2400_PORT)
        self.baud_var = tk.StringVar(value=str(config.K2400_BAUDRATE))
        self.compliance_var = tk.StringVar(value=str(config.K2400_COMPLIANCE))
        self.nplc_var = tk.StringVar(value=str(config.K2400_NPLC))

        self._build_ui()
        self._refresh_ports()
        self._refresh_scan_list()

    # --- 界面构建 ----------------------------------------------------
    def _build_ui(self):
        paned = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=12, pady=12)

        # 左侧：配置
        left = tk.Frame(paned, bg=COLOR_BG)
        paned.add(left, weight=1)

        hdr = tk.Frame(left, bg=COLOR_BG)
        hdr.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(hdr, text="网格扫描配置", style="Title.TLabel").pack(anchor=tk.W)

        card = make_card(left)
        card.pack(fill=tk.X, pady=6)
        inner = tk.Frame(card, bg=COLOR_CARD)
        inner.pack(fill=tk.X, padx=12, pady=12)

        # 参数 1 (Keithley)
        ttk.Label(inner, text="参数 1 (Keithley)", style="Section.TLabel").grid(
            row=0, column=0, columnspan=4, sticky=tk.W, pady=(0, 6))

        ttk.Label(inner, text="名称").grid(row=1, column=0, sticky=tk.W, padx=(0, 4))
        ttk.Entry(inner, textvariable=self.p1_name_var, width=14).grid(
            row=1, column=1, sticky=tk.W, padx=(0, 8), pady=2)
        ttk.Label(inner, text="模式").grid(row=1, column=2, sticky=tk.W, padx=(0, 4))
        ttk.Combobox(inner, textvariable=self.p1_mode_var,
                     values=["voltage", "current"], state="readonly", width=10).grid(
            row=1, column=3, sticky=tk.W, pady=2)

        ttk.Label(inner, text="起始值").grid(row=2, column=0, sticky=tk.W, padx=(0, 4))
        ttk.Entry(inner, textvariable=self.p1_start_var, width=10).grid(
            row=2, column=1, sticky=tk.W, padx=(0, 8), pady=2)
        ttk.Label(inner, text="结束值").grid(row=2, column=2, sticky=tk.W, padx=(0, 4))
        ttk.Entry(inner, textvariable=self.p1_stop_var, width=10).grid(
            row=2, column=3, sticky=tk.W, pady=2)

        ttk.Label(inner, text="步进").grid(row=3, column=0, sticky=tk.W, padx=(0, 4))
        ttk.Entry(inner, textvariable=self.p1_step_var, width=10).grid(
            row=3, column=1, sticky=tk.W, padx=(0, 8), pady=2)

        # 参数 2 (AWG Vpp)
        ttk.Label(inner, text="参数 2 (AWG Vpp)", style="Section.TLabel").grid(
            row=4, column=0, columnspan=4, sticky=tk.W, pady=(12, 6))

        ttk.Label(inner, text="起始值").grid(row=5, column=0, sticky=tk.W, padx=(0, 4))
        ttk.Entry(inner, textvariable=self.vpp_start_var, width=10).grid(
            row=5, column=1, sticky=tk.W, padx=(0, 8), pady=2)
        ttk.Label(inner, text="结束值").grid(row=5, column=2, sticky=tk.W, padx=(0, 4))
        ttk.Entry(inner, textvariable=self.vpp_stop_var, width=10).grid(
            row=5, column=3, sticky=tk.W, pady=2)

        ttk.Label(inner, text="步进").grid(row=6, column=0, sticky=tk.W, padx=(0, 4))
        ttk.Entry(inner, textvariable=self.vpp_step_var, width=10).grid(
            row=6, column=1, sticky=tk.W, padx=(0, 8), pady=2)

        # 处理流程选项
        ttk.Label(inner, text="处理流程", style="Section.TLabel").grid(
            row=7, column=0, columnspan=4, sticky=tk.W, pady=(12, 6))

        ttk.Label(inner, text="运行模式").grid(row=8, column=0, sticky=tk.W, padx=(0, 4))
        ttk.Combobox(inner, textvariable=self.run_mode_var,
                     values=["step1-4", "step1-2"], state="readonly", width=12).grid(
            row=8, column=1, sticky=tk.W, padx=(0, 8), pady=2)
        ttk.Label(inner, text="重复次数").grid(row=8, column=2, sticky=tk.W, padx=(0, 4))
        ttk.Entry(inner, textvariable=self.repeats_var, width=10).grid(
            row=8, column=3, sticky=tk.W, pady=2)

        ttk.Checkbutton(inner, text="离线模式", variable=self.offline_var).grid(
            row=9, column=0, columnspan=2, sticky=tk.W, pady=2)
        ttk.Checkbutton(inner, text="虚拟信道", variable=self.use_virtual_channel_var).grid(
            row=9, column=2, columnspan=2, sticky=tk.W, pady=2)
        ttk.Checkbutton(inner, text="使用 NN 后均衡器", variable=self.use_nn_var).grid(
            row=10, column=0, columnspan=2, sticky=tk.W, pady=2)

        # Keithley 连接
        ttk.Label(inner, text="Keithley 连接", style="Section.TLabel").grid(
            row=11, column=0, columnspan=4, sticky=tk.W, pady=(12, 6))

        ttk.Label(inner, text="COM 端口").grid(row=12, column=0, sticky=tk.W, padx=(0, 4))
        self.port_combo = ttk.Combobox(inner, textvariable=self.port_var,
                                       values=[], width=18, state="readonly")
        self.port_combo.grid(row=12, column=1, sticky=tk.W, padx=(0, 8), pady=2)
        ttk.Button(inner, text="⟳ 刷新", command=self._refresh_ports).grid(
            row=12, column=2, columnspan=2, sticky=tk.W, pady=2)

        ttk.Label(inner, text="波特率").grid(row=13, column=0, sticky=tk.W, padx=(0, 4))
        ttk.Entry(inner, textvariable=self.baud_var, width=10).grid(
            row=13, column=1, sticky=tk.W, padx=(0, 8), pady=2)
        ttk.Label(inner, text="限流值").grid(row=13, column=2, sticky=tk.W, padx=(0, 4))
        ttk.Entry(inner, textvariable=self.compliance_var, width=10).grid(
            row=13, column=3, sticky=tk.W, pady=2)

        ttk.Label(inner, text="NPLC").grid(row=14, column=0, sticky=tk.W, padx=(0, 4))
        ttk.Entry(inner, textvariable=self.nplc_var, width=10).grid(
            row=14, column=1, sticky=tk.W, padx=(0, 8), pady=2)

        # 运行控制
        ctrl = tk.Frame(left, bg=COLOR_CARD)
        ctrl.pack(fill=tk.X, pady=(12, 0), padx=2)
        self.run_btn = ttk.Button(ctrl, text="▶ 开始网格扫描", command=self._start_scan)
        self.run_btn.pack(side=tk.LEFT, padx=(8, 8), pady=8)
        self.stop_btn = ttk.Button(ctrl, text="⏹ 停止", command=self._stop_scan, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=(0, 8), pady=8)

        self.progress_var = tk.DoubleVar(value=0.0)
        self.progress_lbl = ttk.Label(left, text="就绪")
        self.progress_lbl.pack(anchor=tk.W, pady=(8, 0))
        self.progress = ttk.Progressbar(left, variable=self.progress_var, maximum=1.0)
        self.progress.pack(fill=tk.X, pady=(4, 0))

        # 右侧：结果浏览
        right = tk.Frame(paned, bg=COLOR_BG)
        paned.add(right, weight=2)

        ttk.Label(right, text="历史网格扫描", style="Title.TLabel").pack(anchor=tk.W, pady=(0, 8))

        # 扫描选择器
        sel = tk.Frame(right, bg=COLOR_BG)
        sel.pack(fill=tk.X, pady=(0, 6))
        self.scan_var = tk.StringVar()
        self.scan_combo = ttk.Combobox(sel, textvariable=self.scan_var,
                                       values=[], state="readonly", width=40)
        self.scan_combo.pack(side=tk.LEFT, padx=(0, 8))
        self.scan_combo.bind("<<ComboboxSelected>>", self._on_scan_selected)
        ttk.Button(sel, text="⟳ 刷新", command=self._refresh_scan_list).pack(side=tk.LEFT)

        # 汇总表
        self.tree = ttk.Treeview(right, show="headings", height=8)
        self.tree.pack(fill=tk.X, pady=(0, 8))
        self.tree["columns"] = ("point", "param1", "vpp", "ber", "ser", "snr_db", "rate_gbps")
        for col in self.tree["columns"]:
            self.tree.heading(col, text=col.replace("_", " ").title())
            self.tree.column(col, width=80, anchor=tk.CENTER)
        self.tree.column("param1", width=100)
        self.tree.column("vpp", width=80)

        # 等高线图
        plot_card = make_card(right)
        plot_card.pack(fill=tk.BOTH, expand=True, pady=(6, 0))
        plot_inner = tk.Frame(plot_card, bg=COLOR_CARD)
        plot_inner.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        self.plot_metric_var = tk.StringVar(value="snr_db")
        ttk.Label(plot_inner, text="等高线指标").pack(anchor=tk.W)
        metric_combo = ttk.Combobox(plot_inner, textvariable=self.plot_metric_var,
                                    values=["snr_db", "ber", "ser"], state="readonly", width=12)
        metric_combo.pack(anchor=tk.W, pady=(0, 6))
        metric_combo.bind("<<ComboboxSelected>>", self._on_metric_changed)

        self.fig = Figure(figsize=(6, 4), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, plot_inner)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.ax.set_title("选择一个网格扫描以查看等高线图")
        self.canvas.draw()

    # --- 辅助函数 ------------------------------------------------------------
    def _refresh_ports(self):
        ports = refresh_port_list()
        self.port_combo["values"] = ports
        if ports and not self.port_var.get():
            self.port_var.set(parse_port_entry(ports[0]))

    def _refresh_scan_list(self):
        scans = list_grid_scans()
        self.scan_combo["values"] = [s["scan_id"] for s in scans]
        if scans and not self.scan_var.get():
            self.scan_var.set(scans[0]["scan_id"])
            self._on_scan_selected()

    def _on_scan_selected(self, _event=None):
        scan_id = self.scan_var.get()
        if not scan_id:
            return
        header, rows = load_summary(scan_id)
        if header is None:
            return

        # Update table
        for item in self.tree.get_children():
            self.tree.delete(item)
        for row in rows:
            self.tree.insert("", tk.END, values=row)

        self._draw_contour(scan_id)

    def _on_metric_changed(self, _event=None):
        scan_id = self.scan_var.get()
        if scan_id:
            self._draw_contour(scan_id)

    def _draw_contour(self, scan_id: str):
        from scipy.interpolate import griddata

        self.ax.clear()
        header, rows = load_summary(scan_id)
        if header is None or not rows:
            self.ax.set_title("无数据")
            self.canvas.draw()
            return

        col_idx = {h: i for i, h in enumerate(header.split(","))}
        metric = self.plot_metric_var.get()
        if metric not in col_idx:
            metric = "snr_db"

        try:
            bias = np.array([float(r[col_idx["param1"]]) for r in rows])
            vpp = np.array([float(r[col_idx["vpp"]]) for r in rows])
            z = np.array([float(r[col_idx[metric]]) if r[col_idx[metric]] else np.nan for r in rows])
        except Exception:
            self.ax.set_title("数据无效")
            self.canvas.draw()
            return

        xi = np.linspace(bias.min(), bias.max(), 100)
        yi = np.linspace(vpp.min(), vpp.max(), 100)
        Xi, Yi = np.meshgrid(xi, yi)
        Zi = griddata((bias, vpp), z, (Xi, Yi), method="cubic")

        if np.any(np.isfinite(Zi)):
            levels = np.linspace(np.nanmin(Zi), np.nanmax(Zi), 20)
            im = self.ax.contourf(Xi, Yi, Zi, levels=levels, cmap="viridis", extend="both")
            self.fig.colorbar(im, ax=self.ax, label=metric)
        self.ax.set_xlabel(header.split(",")[1] if len(header.split(",")) > 1 else "param1")
        self.ax.set_ylabel("AWG 峰峰电压 Vpp (V)")
        self.ax.set_title(f"{metric.upper()} 网格扫描 ({scan_id})")
        self.fig.tight_layout()
        self.canvas.draw()

    # --- 扫描控制 -------------------------------------------------------
    def _build_config(self) -> GridScanConfig:
        offline = self.offline_var.get()
        return GridScanConfig(
            scan_id="",
            param1_name=self.p1_name_var.get().strip() or "bias",
            param1_mode=self.p1_mode_var.get(),
            param1_start=float(self.p1_start_var.get()),
            param1_stop=float(self.p1_stop_var.get()),
            param1_step=float(self.p1_step_var.get()),
            vpp_start=float(self.vpp_start_var.get()),
            vpp_stop=float(self.vpp_stop_var.get()),
            vpp_step=float(self.vpp_step_var.get()),
            run_mode=self.run_mode_var.get(),
            step4_repeats=int(self.repeats_var.get()),
            use_awg=not offline,
            use_nn=self.use_nn_var.get(),
            use_virtual_channel=self.use_virtual_channel_var.get(),
            offline=offline,
            keithley_port=self.port_var.get(),
            keithley_baudrate=int(self.baud_var.get()),
            keithley_timeout=config.K2400_TIMEOUT,
            keithley_compliance=float(self.compliance_var.get()),
            keithley_nplc=float(self.nplc_var.get()),
        )

    def _start_scan(self):
        if self.scan_thread is not None and self.scan_thread.is_alive():
            return
        try:
            cfg = self._build_config()
        except Exception as exc:
            messagebox.showerror("配置无效", str(exc))
            return

        self.stop_requested = False
        self.run_btn.configure(state=tk.DISABLED)
        self.stop_btn.configure(state=tk.NORMAL)
        self.progress_var.set(0.0)
        self.progress_lbl.configure(text=f"扫描 {cfg.scan_id} 启动中...")

        def progress_cb(fraction):
            self.after(0, lambda: self.progress_var.set(fraction))

        def run():
            try:
                scanner = GridScanner(cfg)
                self.scanner = scanner
                csv_path = scanner.run(progress_callback=progress_cb)
                self.after(0, lambda: self._scan_done(csv_path, cfg.scan_id))
            except Exception as exc:
                self.after(0, lambda: self._scan_error(exc))

        self.scan_thread = threading.Thread(target=run, daemon=True)
        self.scan_thread.start()

    def _stop_scan(self):
        self.stop_requested = True
        if self.scanner is not None:
            try:
                self.scanner.request_stop()
            except Exception:
                pass
        self.progress_lbl.configure(text="已请求停止")

    def _scan_done(self, csv_path: Path, scan_id: str):
        self.run_btn.configure(state=tk.NORMAL)
        self.stop_btn.configure(state=tk.DISABLED)
        self.progress_lbl.configure(text=f"完成: {csv_path}")
        self.progress_var.set(1.0)
        self._refresh_scan_list()
        self.scan_var.set(scan_id)
        self._on_scan_selected()
        messagebox.showinfo("网格扫描完成", f"汇总已保存到:\n{csv_path}")

    def _scan_error(self, exc: Exception):
        self.run_btn.configure(state=tk.NORMAL)
        self.stop_btn.configure(state=tk.DISABLED)
        self.progress_lbl.configure(text=f"错误: {exc}")
        messagebox.showerror("网格扫描失败", str(exc))


# ═══════════════════════════════════════════════════════════════════════════════
# 主窗口
# ═══════════════════════════════════════════════════════════════════════════════

class DmtGuiApp(tk.Tk):
    def __init__(self):
        super().__init__()

        # Windows 任务栏图标需要显式的 AppUserModelID
        _set_windows_taskbar_icon()

        # DPI 缩放
        dpi = self.winfo_fpixels("1i")
        self.font_scale = max(dpi / 96.0, 1.0)
        self.tk.call("tk", "scaling", dpi / 72.0)

        self.title(f"{APP_EMOJI} DMT 通信系统实验平台")
        self._icon, self._icon_ico = _create_emoji_icon(APP_EMOJI)
        # Windows 上优先使用 ICO，避免任务栏显示羽毛图标；其他平台使用 photo
        if self._icon_ico is not None and sys.platform == "win32":
            try:
                self.iconbitmap(str(self._icon_ico))
            except Exception:
                if self._icon is not None:
                    self.iconphoto(True, self._icon)
        elif self._icon is not None:
            self.iconphoto(True, self._icon)
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        w = min(int(sw * 0.82), int(1500 * self.font_scale))
        h = min(int(sh * 0.85), int(950 * self.font_scale))
        self.geometry(f"{w}x{h}+{(sw - w) // 2}+{(sh - h) // 2}")
        self.minsize(int(1050 * self.font_scale), int(680 * self.font_scale))
        self.configure(bg=COLOR_BG)

        apply_styles(self, self.font_scale)

        # 清理历史遗留的空绘图目录
        _cleanup_empty_plot_dirs()

        self.runs = []
        self.records = []
        self.current_run = None
        self._record_by_run = {}
        self._running = False

        # ── 顶部标题栏 ───────────────────────────────────────────────
        header = tk.Frame(self, bg=COLOR_BG)
        header.pack(fill=tk.X, padx=16, pady=(14, 6))
        ttk.Label(header, text=f"{APP_EMOJI} DMT 通信系统实验平台",
                  style="Title.TLabel").pack(side=tk.LEFT)
        self.pill_runs = ttk.Label(header, style="Pill.TLabel")
        self.pill_runs.pack(side=tk.RIGHT, padx=(8, 0))
        self.pill_records = ttk.Label(header, style="Pill.TLabel")
        self.pill_records.pack(side=tk.RIGHT)

        # ── 实验选择栏 ──────────────────────────────────────────────
        sel = make_card(self)
        sel.pack(fill=tk.X, padx=16, pady=(4, 8))
        inner = tk.Frame(sel, bg=COLOR_CARD)
        inner.pack(fill=tk.X, padx=12, pady=10)
        f = self.font_scale
        ttk.Label(inner, text="实验 ID (run_id)", style="Section.TLabel"
                  ).pack(side=tk.LEFT)
        self.run_var = tk.StringVar()
        self.run_combo = ttk.Combobox(
            inner, textvariable=self.run_var, state="readonly",
            width=30, font=(FONT_MONO, 10))
        self.run_combo.pack(side=tk.LEFT, padx=(8, 8))
        self.run_combo.bind("<<ComboboxSelected>>",
                            lambda _e: self.select_run(self.run_var.get()))
        ttk.Button(inner, text="⟳ 刷新数据", command=self.reload_data
                   ).pack(side=tk.LEFT)
        self.metrics_var = tk.StringVar(value="")
        ttk.Label(inner, textvariable=self.metrics_var, style="Metrics.TLabel"
                  ).pack(side=tk.LEFT, padx=20)

        # ── 六个标签页 ────────────────────────────────────────────────
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=16, pady=(0, 8))

        tab1 = ttk.Frame(self.notebook)
        self.notebook.add(tab1, text="  📈 波形与频谱  ")
        self.panel_wave = PlotPanel(tab1, self, TAB_WAVEFORM)
        self.panel_wave.pack(fill=tk.BOTH, expand=True, padx=2, pady=6)

        tab2 = ttk.Frame(self.notebook)
        self.notebook.add(tab2, text="  🎛 DMT 调制  ")
        self.panel_mod = PlotPanel(tab2, self, TAB_MODULATION)
        self.panel_mod.pack(fill=tk.BOTH, expand=True, padx=2, pady=6)

        tab3 = ResultsPanel(self.notebook, self)
        self.notebook.add(tab3, text="  📊 传输结果  ")
        self.results_panel = tab3

        tab4 = RunPanel(self.notebook, self)
        self.notebook.add(tab4, text="  ▶ 运行测试  ")
        self.panel_run = tab4

        tab5 = Keithley2400Panel(self.notebook, self)
        self.notebook.add(tab5, text="  ⚡ Keithley 2400  ")
        self.panel_k2400 = tab5

        tab6 = GridScanPanel(self.notebook, self)
        self.notebook.add(tab6, text="  🔲 网格扫描  ")
        self.panel_grid = tab6

        # 默认显示“运行测试”页（第 4 个标签页，索引 3）
        self.notebook.select(3)
        
        # ── 状态栏 ───────────────────────────────────────────────────
        self.status_var = tk.StringVar()
        status = tk.Label(self, textvariable=self.status_var, anchor=tk.W,
                          bg=COLOR_BG, fg=COLOR_TEXT_DIM, bd=0,
                          font=(FONT_FAMILY, 9))
        status.pack(fill=tk.X, side=tk.BOTTOM, padx=16, pady=(0, 8))

        # 确保 GUI 关闭时 Keithley 输出已关闭
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        self.reload_data()

    # ── 数据 ───────────────────────────────────────────────────────

    def reload_data(self, select_latest=False):
        self.records = list_records()
        self._record_by_run = {r.get("run_id"): r for r in self.records}
        # 按记录去重并最新优先显示，以便单步运行也能被看到
        seen = set()
        self.runs = []
        for rec in reversed(self.records):
            run_id = rec.get("run_id", "")
            if run_id and run_id not in seen:
                seen.add(run_id)
                self.runs.append(run_id)
        self.run_combo["values"] = self.runs
        self.pill_runs.configure(text=f"{len(self.runs)} 个实验")
        self.pill_records.configure(text=f"{len(self.records)} 条记录")
        if not self.runs:
            self.status_var.set(
                f"未找到实验数据 ({RECORDS_DIR})。请先在\"运行测试\"页面运行一次测试。")
            return
        latest = self.runs[0]
        if select_latest or self.current_run not in self.runs:
            self.select_run(latest)
        else:
            self.select_run(self.current_run)
        self.status_var.set(f"数据目录: {ASSETS_DIR}; 记录目录: {RECORDS_DIR}")

    def select_run(self, run_id, source=None):
        if not run_id:
            return
        self.current_run = run_id
        if self.run_var.get() != run_id:
            self.run_var.set(run_id)
        rec = self._record_by_run.get(run_id)
        if rec:
            self.metrics_var.set(
                f"速率 {rec.get('final_rate_gbps', 0):.2f} Gbps | "
                f"BER {rec.get('final_ber', 0):.3e} | "
                f"SER {rec.get('final_ser', 0):.3e} | "
                f"平均信噪比 {rec.get('mean_recovered_snr_db', 0):.2f} dB | "
                f"导频 {rec.get('pilot_pattern', '-')}")
        else:
            self.metrics_var.set("（该实验无记录文件）")
        self.panel_wave.refresh()
        self.panel_mod.refresh()
        if source != "results":
            self.results_panel.refresh()
        else:
            self.results_panel.plot_panel.refresh()

    def set_running(self, running):
        self._running = running
        self.run_combo.configure(state="readonly" if not running
                                 else tk.DISABLED)

    def _on_close(self):
        """Clean up the Keithley connection and stop grid scans before exit."""
        try:
            if hasattr(self, "panel_k2400"):
                self.panel_k2400.on_close()
        except Exception:
            pass
        try:
            if hasattr(self, "panel_grid"):
                self.panel_grid._stop_scan()
        except Exception:
            pass
        self.destroy()


def main():
    enable_dpi_awareness()
    app = DmtGuiApp()
    app.mainloop()


if __name__ == "__main__":
    main()
