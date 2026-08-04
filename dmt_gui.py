# -*- coding: utf-8 -*-
"""DMT 通信系统实验平台 GUI.

四个子页面：
    1. 波形时频域   —— TX/RX 时域波形与频谱
    2. DMT 符号调制 —— bit/power loading、星座图、星座密度
    3. 传输实验结果 —— 实验记录表、SNR 对比、每载波 SER/BER、非线性、历次趋势
    4. 运行测试     —— 通过 GUI 调用 main.py 完成实验，日志实时显示

数据来源（与 main.py 自动保存一致）：
    data/codeplot_assets/<run_id>/data/*.npz   每次实验的绘图数据
    data/records/record_<run_id>.json          每次实验的参数与结果记录

运行方式（在项目目录下）：
    .venv\\Scripts\\python dmt_gui.py

仅依赖 numpy / matplotlib / tkinter（Python 自带），不需要额外安装。
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
from tkinter import ttk, messagebox
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

# 中文字体
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

PROJECT_ROOT = Path(__file__).resolve().parent
ASSETS_DIR = PROJECT_ROOT / "data" / "codeplot_assets"
RECORDS_DIR = PROJECT_ROOT / "data" / "records"
MAIN_PY = PROJECT_ROOT / "main.py"
CONFIG_PY = PROJECT_ROOT / "config.py"


# ═══════════════════════════════════════════════════════════════════════════════
# 高 DPI 适配（必须在创建 Tk 之前调用）
# ═══════════════════════════════════════════════════════════════════════════════

def enable_dpi_awareness():
    """让 Windows 按真实 DPI 渲染，避免高分屏下界面过小或模糊."""
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
# 配色（参考现代浅色词典风格）
# ═══════════════════════════════════════════════════════════════════════════════

COLOR_BG = "#F3F5F7"          # 窗口底色（浅灰）
COLOR_CARD = "#FFFFFF"        # 卡片白
COLOR_BORDER = "#E2E8F0"      # 卡片描边
COLOR_PRIMARY = "#164E63"     # 主色（深青）
COLOR_PRIMARY_HOVER = "#0E7490"
COLOR_TEXT = "#1F2937"        # 主文字
COLOR_TEXT_DIM = "#64748B"    # 次要文字
COLOR_DANGER = "#B91C1C"
COLOR_SELECT = "#164E63"      # 列表选中
FONT_FAMILY = "Microsoft YaHei UI"
FONT_MONO = "Consolas"


# ═══════════════════════════════════════════════════════════════════════════════
# 数据发现
# ═══════════════════════════════════════════════════════════════════════════════

def list_runs():
    """扫描 data/codeplot_assets，返回按时间倒序的 run_id 列表."""
    if not ASSETS_DIR.is_dir():
        return []
    runs = [p.name for p in ASSETS_DIR.iterdir()
            if p.is_dir() and (p / "data").is_dir()]
    return sorted(runs, reverse=True)


def run_data_dir(run_id):
    return ASSETS_DIR / run_id / "data"


def available_plots(run_id, names):
    """返回该 run 实际存在的图名列表（保持 names 顺序）."""
    d = run_data_dir(run_id)
    return [n for n in names if (d / f"{n}.npz").is_file()]


def list_records():
    """读取 data/records 下所有 record_*.json，按时间升序返回."""
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
# 绘图函数（与 plot_adapter.py 中的 CodePlot 模板保持一致）
# ═══════════════════════════════════════════════════════════════════════════════

def build_time(fig, npz_path, title):
    data = np.load(npz_path)
    t, sig = data["t"], data["sig"]
    ax = fig.add_subplot(111)
    ax.plot(t, sig, "b.-", linewidth=1, markersize=2)
    ax.set_title(title)
    ax.set_xlabel("Sample")
    ax.set_ylabel("Amplitude")
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
    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel("Magnitude (dB)")
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
    ax.plot(est, "b", label="Est-SNR", marker="o", markersize=3, linewidth=1)
    ax.plot(real, "r", label="TestReal-SNR", marker="x", markersize=3,
            linewidth=1)
    ax.set_title(title)
    ax.set_xlabel("Subcarrier")
    ax.set_ylabel("SNR (dB)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()


def build_nonlinearity(fig, npz_path, title):
    data = np.load(npz_path)
    tx, rx = data["tx"], data["rx"]
    ax = fig.add_subplot(111)
    if len(tx) > 5000:
        hb = ax.hexbin(tx, rx, gridsize=80, cmap="GnBu", mincnt=1)
        fig.colorbar(hb, ax=ax, label="Density")
    else:
        ax.plot(tx, rx, "b.", alpha=0.2, markersize=3)
    if np.any(tx):
        gain = np.sum(tx * rx) / np.sum(tx ** 2)
        t = np.linspace(tx.min(), tx.max(), 100)
        ax.plot(t, gain * t, "g--", linewidth=2,
                label=f"Linear fit (gain={gain:.3f})")
    ax.set_title(title)
    ax.set_xlabel("TX Amplitude")
    ax.set_ylabel("RX Amplitude")
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
    ax1.plot(subcarriers, snrs_db, "b-", linewidth=1.5, label="SNR (dB)")
    ax1_bits.plot(subcarriers, RQ, "r-", linewidth=1.5, marker="x",
                  markersize=3, label="Bit allocation")
    ax1.set_ylabel("SNR (dB)", color="b")
    ax1_bits.set_ylabel("Bits / symbol", color="r")
    ax1.set_title(f"{title} (ratio={ratio}, rate={rate_gbps:.2f} Gbps)")
    ax1.grid(True, alpha=0.3)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_bits.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    ax2 = fig.add_subplot(212, sharex=ax1)
    ax2.plot(subcarriers, S, "g-", linewidth=1.5, marker="o", markersize=2,
             label="Power allocation")
    ax2.set_xlabel("Subcarrier")
    ax2.set_ylabel("Power scaling")
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
    ax1.set_title(f"{title} — SER per Subcarrier")
    ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(212)
    ax2.semilogy(carrier_idx, np.where(ber > 0, ber, 1e-12), "b-",
                 marker="x", markersize=3, linewidth=1)
    ax2.set_xlabel("Subcarrier Index")
    ax2.set_ylabel("BER")
    ax2.set_title("BER per Subcarrier")
    ax2.grid(True, which="both", ls="--", alpha=0.3)

    if RQ is not None:
        ax2_twin = ax2.twinx()
        ax2_twin.plot(carrier_idx, RQ, "g--", alpha=0.5,
                      label="Bit allocation")
        ax2_twin.set_ylabel("Bits / symbol", color="g")
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
        fig.colorbar(hb, ax=ax, label="Density")
        ax.set_title(f"{2 ** bits}-QAM (bits={bits})")
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
        ax.set_title(f"{2 ** bits}-QAM (bits={bits})")
        ax.set_xlabel("I")
        ax.set_ylabel("Q")
        ax.axis("equal")
        ax.grid(True, alpha=0.3)
    fig.suptitle(title, y=1.02)
    fig.tight_layout()


def build_records_trend(fig, records, title):
    """历次实验趋势：速率与 BER."""
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
             label="Final rate")
    if any(s is not None for s in snrs):
        ax1b = ax1.twinx()
        ax1b.plot(xs, [s if s is not None else np.nan for s in snrs],
                  "g--s", markersize=4, linewidth=1, label="Mean SNR (dB)")
        ax1b.set_ylabel("Mean recovered SNR (dB)", color="g")
        ax1b.legend(loc="lower right")
    ax1.set_ylabel("Rate (Gbps)", color="b")
    ax1.set_title(title)
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(xs)
    ax1.set_xticklabels(labels, rotation=45, fontsize=8)

    ax2 = fig.add_subplot(212, sharex=ax1)
    ax2.semilogy(xs, bers, "r-x", markersize=4, linewidth=1.5)
    ax2.set_xlabel("Run (时间顺序)")
    ax2.set_ylabel("BER")
    ax2.grid(True, which="both", ls="--", alpha=0.3)
    ax2.set_xticks(xs)
    ax2.set_xticklabels(labels, rotation=45, fontsize=8)
    fig.tight_layout()


# ═══════════════════════════════════════════════════════════════════════════════
# 图目录：名称 -> (绘图函数, 中文标题)
# ═══════════════════════════════════════════════════════════════════════════════

FIGURES = {
    # 子页面 1：波形时频域
    "SNRest_QPSK_time":          (build_time,          "QPSK 探测 TX 时域波形"),
    "SNRest_QPSK_spec":          (build_spectrum,      "QPSK 探测 TX 频谱"),
    "SNRest_QPSK_rx_spec":       (build_spectrum,      "QPSK 探测 RX 频谱"),
    "DMT_bitloading_Tx_time":    (build_time,          "DMT Bitloading TX 时域波形"),
    "DMT_bitloading_Tx_spec":    (build_spectrum,      "DMT Bitloading TX 频谱"),
    "DMT_bitloading_Rx_spec":    (build_spectrum,      "DMT Bitloading RX 频谱"),
    # 子页面 2：DMT 符号调制
    "bit_power_loading":         (build_bit_power_loading, "Bit / Power Loading"),
    "SNRest_QPSK_constellation": (build_constellation, "QPSK 探测星座图"),
    "bitloading_constellation":  (build_constellation, "DMT Bitloading 星座图"),
    "constellation_by_order":    (build_const_by_order,    "RX 星座图（按调制阶数）"),
    "constellation_density":     (build_const_density,     "星座点密度（按调制阶数）"),
    # 子页面 3：传输实验结果
    "SNR_QPSK":                  (build_snr,           "QPSK 估计 SNR vs 实测 SNR"),
    "SNR_compare":               (build_snr,           "SNR 对比（估计 vs 恢复）"),
    "ser_ber_per_carrier":       (build_ser_ber,       "每子载波 SER / BER"),
    "SNRest_QPSK_nonlinearity":  (build_nonlinearity,  "QPSK TX-RX 幅值非线性"),
    "DMT_bitloading_nonlinearity": (build_nonlinearity, "DMT TX-RX 幅值非线性"),
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

TREND_KEY = "__records_trend__"   # 子页面 3 中的历次趋势虚拟图名


# ═══════════════════════════════════════════════════════════════════════════════
# 样式
# ═══════════════════════════════════════════════════════════════════════════════

def apply_styles(root, scale):
    """现代化浅色主题（clam 基础上定制）."""
    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except tk.TclError:
        pass

    f = max(scale, 1.0)
    # 字体用磅值（tk scaling 已按 DPI 缩放），像素类尺寸才乘 f
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

    # Notebook 选项卡：大留白、选中变白色
    style.configure("TNotebook", background=COLOR_BG, borderwidth=0)
    style.configure("TNotebook.Tab", font=font_tab,
                    padding=(pad_x + 6, pad_y),
                    background="#E5EAEF", foreground=COLOR_TEXT)
    style.map("TNotebook.Tab",
              background=[("selected", COLOR_CARD)],
              foreground=[("selected", COLOR_PRIMARY)])

    # 主按钮（深青底白字）
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

    # 单选 / 复选（indicator 尺寸随 DPI 放大，避免高分屏下过小）
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
    """白色卡片容器（细描边 + 内边距）."""
    card = tk.Frame(parent, bg=COLOR_CARD,
                    highlightbackground=COLOR_BORDER, highlightthickness=1,
                    bd=0)
    if pack_kwargs:
        card.pack(**pack_kwargs)
    return card


# ═══════════════════════════════════════════════════════════════════════════════
# GUI 组件
# ═══════════════════════════════════════════════════════════════════════════════

class PlotPanel(ttk.Frame):
    """左侧图列表 + 右侧 matplotlib 画布."""

    def __init__(self, parent, app, plot_names, include_trend=False):
        super().__init__(parent)
        self.app = app
        self.plot_names = list(plot_names)
        self.include_trend = include_trend
        f = app.font_scale

        left = make_card(self)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 8))
        ttk.Label(left, text="图表列表", style="Section.TLabel"
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
        """根据当前 run 重新填充列表."""
        run_id = self.app.current_run
        self.listbox.delete(0, tk.END)
        self._items = []
        if run_id:
            for name in available_plots(run_id, self.plot_names):
                self.listbox.insert(tk.END, f"  {FIGURES[name][1]}")
                self._items.append(name)
        if self.include_trend and self.app.records:
            self.listbox.insert(tk.END, "  历次实验趋势（速率 / BER / SNR）")
            self._items.append(TREND_KEY)
        if self._items:
            self.listbox.selection_set(0)
            self._show(self._items[0])
        else:
            self.fig.clear()
            ax = self.fig.add_subplot(111)
            ax.text(0.5, 0.5, "当前实验没有该类别数据",
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
                build_records_trend(self.fig, self.app.records, "历次实验趋势")
            else:
                builder, title = FIGURES[name]
                npz = run_data_dir(self.app.current_run) / f"{name}.npz"
                builder(self.fig, npz, title)
        except Exception as exc:  # 数据缺失/格式异常时给出提示而非崩溃
            self.fig.clear()
            ax = self.fig.add_subplot(111)
            ax.text(0.5, 0.5, f"绘图失败：\n{exc}", ha="center", va="center",
                    transform=ax.transAxes, color="red")
        self.canvas.draw_idle()


class ResultsPanel(ttk.Frame):
    """子页面 3：上方实验记录表，下方该次实验的结果图."""

    COLUMNS = ("run_id", "time", "pilot", "vc", "nn", "est_rate",
               "final_rate", "ber", "ser", "snr")
    HEADINGS = {
        "run_id": ("实验编号", 170),
        "time": ("时间", 150),
        "pilot": ("导频", 90),
        "vc": ("虚拟信道", 70),
        "nn": ("NN", 50),
        "est_rate": ("探测速率 (Gbps)", 110),
        "final_rate": ("最终速率 (Gbps)", 110),
        "ber": ("BER", 100),
        "ser": ("SER", 100),
        "snr": ("平均 SNR (dB)", 100),
    }

    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app

        top = ttk.LabelFrame(self, text=" 传输通信实验记录（点击行切换实验） ")
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

        bottom = ttk.LabelFrame(self, text=" 当前实验结果图 ")
        bottom.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
        self.plot_panel = PlotPanel(bottom, app, TAB_RESULTS,
                                    include_trend=True)
        self.plot_panel.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        self._iid_to_run = {}

    def refresh(self):
        self.tree.delete(*self.tree.get_children())
        self._iid_to_run = {}
        for rec in reversed(self.app.records):   # 最新的在最上面
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
# 类型: bool=0/1 或 True/False 开关; int=整数; num=数值(支持科学计数法/表达式);
#       str=字符串(需带引号); ichoice=整数下拉; schoice=字符串下拉
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
        ("NORMALIZE_FLAG", "ichoice", ["0", "1"]),
    ]),
    ("导频图案", [
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
    ("硬件（AWG / 示波器）", [
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
    """安全求值：先 literal_eval，失败后允许无内置函数的简单算术表达式."""
    try:
        return ast.literal_eval(text)
    except Exception:
        return eval(text, {"__builtins__": {}}, {})


def _find_assignment(text, name):
    """在 config.py 源码中找到 `NAME = rhs  # comment`，返回 (rhs, gap, comment).

    gap 为 rhs 与行内注释之间的原始空白，用于写回时保持注释对齐。
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
    """config.py 可调参数编辑器：分组展示，仅把修改过的参数写回文件."""

    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        f = app.font_scale
        self.params = []   # 每个元素: dict(name, type, choices, var, orig, comment, widget)

        # ── 头部按钮 ───────────────────────────────────────────────
        head = tk.Frame(self, bg=COLOR_CARD)
        head.pack(fill=tk.X, padx=8, pady=(8, 4))
        ttk.Button(head, text="💾 保存到 config.py", style="Accent.TButton",
                   command=self.save).pack(side=tk.LEFT)
        ttk.Button(head, text="⟳ 重新读取", command=self.reload
                   ).pack(side=tk.LEFT, padx=8)
        self.info_var = tk.StringVar(
            value="修改参数后先保存再运行；仅改动过的参数会写回文件")
        ttk.Label(head, textvariable=self.info_var, style="DimCard.TLabel"
                  ).pack(side=tk.LEFT, padx=12)

        # ── 可滚动区域 ─────────────────────────────────────────────
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
        # 鼠标滚轮（仅悬停在参数区时启用）
        self._canvas.bind("<Enter>", lambda _e: self._canvas.bind_all(
            "<MouseWheel>", self._on_wheel))
        self._canvas.bind("<Leave>", lambda _e: self._canvas.unbind_all(
            "<MouseWheel>"))

        self.reload()

    def _on_wheel(self, event):
        self._canvas.yview_scroll(int(-event.delta / 120), "units")

    # ── 读取 / 构建 ────────────────────────────────────────────────

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
                    tk.Label(row, text="（config.py 中未找到）",
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
        self.info_var.set(f"已读取 {len(self.params)} 个参数（{CONFIG_PY.name}）")

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

    # ── 取值 / 写回 ────────────────────────────────────────────────

    def _literal_for(self, p, errors):
        """把控件当前值转换为可写入 config.py 的字面量文本."""
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
                errors.append(f"{name}: 需要整数，当前为 «{raw}»")
                return None
            return str(int(val))
        if ptype == "num":
            if not isinstance(val, (int, float)):
                errors.append(f"{name}: 需要数值，当前为 «{raw}»")
                return None
            return raw
        if ptype == "str":
            if not isinstance(val, str):
                errors.append(f"{name}: 字符串需带引号，如 \"TCPIP0::…\"")
                return None
            return raw
        return raw

    def get_value(self, name):
        """返回某参数当前控件值（数值化）."""
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
            messagebox.showerror("找不到 config.py", str(CONFIG_PY))
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
                                 "以下参数未能在 config.py 中定位："
                                 + ", ".join(errors))
            return False
        with CONFIG_PY.open("w", encoding="utf-8", newline="") as fp:
            fp.write(text)
        msg = f"已保存 {len(pending)} 个参数到 config.py" if pending \
            else "没有改动，config.py 保持不变"
        self.info_var.set(msg)
        if not silent and pending:
            messagebox.showinfo("保存成功", msg)
        return True


class RunPanel(ttk.Frame):
    """子页面 4：通过 GUI 调用 main.py 运行实验."""

    MODES = {
        "online":  ("在线（AWG + 示波器）", {"--offline": "0", "--use-awg": "1",
                                            "--use-virtual-channel": "0"}),
        "offline": ("离线（读取已有 RX 采集）", {"--offline": "1", "--use-awg": "0",
                                               "--use-virtual-channel": "0"}),
        "virtual": ("离线 + 虚拟信道（无硬件调试）", {"--offline": "1", "--use-awg": "0",
                                                    "--use-virtual-channel": "1"}),
    }
    STEPS = {
        "all": "完整流程（探测 + Bitloading）",
        "step1": "step1 生成 QPSK 探测 TX",
        "step2": "step2 接收 QPSK 并估计 SNR",
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

        ttk.Label(opt, text="运行模式：", style="Card.TLabel"
                  ).grid(row=0, column=0, sticky=tk.W, padx=12, pady=(10, 4))
        self.mode_var = tk.StringVar(value="virtual")
        col = 1
        for key, (label, _args) in self.MODES.items():
            ttk.Radiobutton(opt, text=label, value=key,
                            variable=self.mode_var
                            ).grid(row=0, column=col, sticky=tk.W,
                                   padx=(4, 16), pady=(10, 4))
            col += 1

        ttk.Label(opt, text="流程步骤：", style="Card.TLabel"
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
        ttk.Checkbutton(opt, text="启用 NN 后均衡（--use-nn，首次运行需训练）",
                        variable=self.use_nn_var
                        ).grid(row=2, column=1, columnspan=3, sticky=tk.W,
                               padx=(4, 16), pady=(4, 10))

        btn_bar = tk.Frame(opt, bg=COLOR_CARD)
        btn_bar.grid(row=3, column=0, columnspan=5, sticky=tk.W,
                     padx=12, pady=(0, 12))
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

        # ── 下方：左侧 config 参数面板 + 右侧运行日志 ───────────────
        paned = ttk.Panedwindow(self, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
        paned.after_idle(lambda: paned.sashpos(
            0, int(paned.winfo_width() * 0.56)))

        cfg_card = ttk.LabelFrame(paned, text=" config.py 实验参数 ")
        paned.add(cfg_card, weight=3)
        self.config_panel = ConfigPanel(cfg_card, app)
        self.config_panel.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        # ── 日志卡片 ───────────────────────────────────────────────
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
            "提示：选择运行模式与步骤后点击「开始测试」。\n"
            "在线模式需要连接 M8190A 与示波器；无硬件时建议使用虚拟信道模式。\n"
            "测试完成后会自动刷新数据并切换到最新实验。\n", "head")

    # ── 运行控制 ──────────────────────────────────────────────────

    def build_command(self):
        mode = self.mode_var.get()
        args = self.MODES[mode][1]
        step = self.step_combo.get().split()[0] or "all"
        cmd = [sys.executable, str(MAIN_PY)]
        for k, v in args.items():
            cmd += [k, v]
        cmd += ["--use-nn", "1" if self.use_nn_var.get() else "0",
                "--step", step]
        return cmd

    def start_run(self):
        if self.proc is not None:
            return
        if not MAIN_PY.is_file():
            messagebox.showerror("找不到 main.py", f"未找到 {MAIN_PY}")
            return
        # 先把参数面板的改动写回 config.py（校验失败则中止）
        if not self.config_panel.save(silent=True):
            return
        # main.py 中 step1/step3 的 AWG 下载由 config.OFFLINE_FLAG 控制，
        # 与 GUI 选择的运行模式可能不一致，这里提前检查避免误触硬件
        mode = self.mode_var.get()
        off_flag = self.config_panel.get_value("OFFLINE_FLAG")
        if mode in ("offline", "virtual") and off_flag == 0:
            if messagebox.askyesno(
                    "检测到 OFFLINE_FLAG=0",
                    "config.py 中 OFFLINE_FLAG=0，运行时 step1/step3 仍会向 "
                    "AWG 下载波形。\n\n是否先把 OFFLINE_FLAG 改为 1 再运行？\n"
                    "（选「否」则取消本次运行）"):
                self.config_panel.set_value("OFFLINE_FLAG", "1")
                self.config_panel.save(silent=True)
            else:
                return
        elif mode == "online" and off_flag not in (0, None):
            if not messagebox.askyesno(
                    "检测到 OFFLINE_FLAG=1",
                    "在线模式但 config.py 中 OFFLINE_FLAG=1，step1/step3 不会"
                    "向 AWG 下载波形，示波器可能采不到信号。\n\n仍要继续吗？"):
                return
        cmd = self.build_command()
        mode_label = self.MODES[self.mode_var.get()][0]
        if self.mode_var.get() == "online":
            if not messagebox.askyesno(
                    "确认在线模式",
                    "在线模式将控制 M8190A 与示波器，确认硬件已连接？"):
                return
        env = dict(os.environ)
        env["MPLBACKEND"] = "Agg"        # 子进程不弹 matplotlib 窗口
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
        self._append_log(f"模式：{mode_label}\n", "head")
        self.run_btn.configure(state=tk.DISABLED)
        self.stop_btn.configure(state=tk.NORMAL)
        self.run_status.configure(text="运行中…")
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

    def _on_done(self, code):
        self.proc = None
        self.run_btn.configure(state=tk.NORMAL)
        self.stop_btn.configure(state=tk.DISABLED)
        self.app.set_running(False)
        if code == 0:
            self.run_status.configure(text="测试完成 ✔")
            self._append_log("\n========== 测试完成 ==========\n", "head")
            self.app.reload_data(select_latest=True)
        else:
            self.run_status.configure(text=f"异常退出 (code={code})")
            self._append_log(f"\n进程退出，返回码 {code}\n", "err")

    def stop_run(self):
        if self.proc is not None:
            self.proc.terminate()
            self._append_log("\n（已请求停止进程…）\n", "err")

    def _append_log(self, text, tag=None):
        self.log_text.configure(state=tk.NORMAL)
        if tag:
            self.log_text.insert(tk.END, text, tag)
        else:
            self.log_text.insert(tk.END, text)
        self.log_text.see(tk.END)
        self.log_text.configure(state=tk.DISABLED)


# ═══════════════════════════════════════════════════════════════════════════════
# 主窗口
# ═══════════════════════════════════════════════════════════════════════════════

class DmtGuiApp(tk.Tk):
    def __init__(self):
        super().__init__()

        # DPI 缩放
        dpi = self.winfo_fpixels("1i")
        self.font_scale = max(dpi / 96.0, 1.0)
        self.tk.call("tk", "scaling", dpi / 72.0)

        self.title("DMT 通信系统实验平台")
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        w = min(int(sw * 0.82), int(1500 * self.font_scale))
        h = min(int(sh * 0.85), int(950 * self.font_scale))
        self.geometry(f"{w}x{h}+{(sw - w) // 2}+{(sh - h) // 2}")
        self.minsize(int(1050 * self.font_scale), int(680 * self.font_scale))
        self.configure(bg=COLOR_BG)

        apply_styles(self, self.font_scale)

        self.runs = []
        self.records = []
        self.current_run = None
        self._record_by_run = {}
        self._running = False

        # ── 顶部标题栏 ─────────────────────────────────────────────
        header = tk.Frame(self, bg=COLOR_BG)
        header.pack(fill=tk.X, padx=16, pady=(14, 6))
        ttk.Label(header, text="📡 DMT 通信系统实验平台",
                  style="Title.TLabel").pack(side=tk.LEFT)
        self.pill_runs = ttk.Label(header, style="Pill.TLabel")
        self.pill_runs.pack(side=tk.RIGHT, padx=(8, 0))
        self.pill_records = ttk.Label(header, style="Pill.TLabel")
        self.pill_records.pack(side=tk.RIGHT)

        # ── 实验选择栏 ─────────────────────────────────────────────
        sel = make_card(self)
        sel.pack(fill=tk.X, padx=16, pady=(4, 8))
        inner = tk.Frame(sel, bg=COLOR_CARD)
        inner.pack(fill=tk.X, padx=12, pady=10)
        f = self.font_scale
        ttk.Label(inner, text="实验编号 (run_id)", style="Section.TLabel"
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

        # ── 四个子页面 ─────────────────────────────────────────────
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=16, pady=(0, 8))

        tab1 = ttk.Frame(self.notebook)
        self.notebook.add(tab1, text="  📈 波形时频域  ")
        self.panel_wave = PlotPanel(tab1, self, TAB_WAVEFORM)
        self.panel_wave.pack(fill=tk.BOTH, expand=True, padx=2, pady=6)

        tab2 = ttk.Frame(self.notebook)
        self.notebook.add(tab2, text="  🎛 DMT 符号调制  ")
        self.panel_mod = PlotPanel(tab2, self, TAB_MODULATION)
        self.panel_mod.pack(fill=tk.BOTH, expand=True, padx=2, pady=6)

        tab3 = ResultsPanel(self.notebook, self)
        self.notebook.add(tab3, text="  📊 传输实验结果  ")
        self.results_panel = tab3

        tab4 = RunPanel(self.notebook, self)
        self.notebook.add(tab4, text="  ▶ 运行测试  ")
        self.panel_run = tab4

        # ── 状态栏 ─────────────────────────────────────────────────
        self.status_var = tk.StringVar()
        status = tk.Label(self, textvariable=self.status_var, anchor=tk.W,
                          bg=COLOR_BG, fg=COLOR_TEXT_DIM, bd=0,
                          font=(FONT_FAMILY, 9))
        status.pack(fill=tk.X, side=tk.BOTTOM, padx=16, pady=(0, 8))

        self.reload_data()

    # ── 数据 ──────────────────────────────────────────────────────

    def reload_data(self, select_latest=False):
        self.runs = list_runs()
        self.records = list_records()
        self._record_by_run = {r.get("run_id"): r for r in self.records}
        self.run_combo["values"] = self.runs
        self.pill_runs.configure(text=f"{len(self.runs)} 次实验")
        self.pill_records.configure(text=f"{len(self.records)} 条记录")
        if not self.runs:
            self.status_var.set(
                f"未找到实验数据（{ASSETS_DIR}）。请先在「运行测试」页完成一次测试。")
            return
        latest = self.runs[0]
        if select_latest or self.current_run not in self.runs:
            self.select_run(latest)
        else:
            self.select_run(self.current_run)
        self.status_var.set(f"数据目录：{ASSETS_DIR}")

    def select_run(self, run_id, source=None):
        if not run_id:
            return
        self.current_run = run_id
        if self.run_var.get() != run_id:
            self.run_var.set(run_id)
        rec = self._record_by_run.get(run_id)
        if rec:
            self.metrics_var.set(
                f"速率 {rec.get('final_rate_gbps', 0):.2f} Gbps ｜ "
                f"BER {rec.get('final_ber', 0):.3e} ｜ "
                f"SER {rec.get('final_ser', 0):.3e} ｜ "
                f"平均 SNR {rec.get('mean_recovered_snr_db', 0):.2f} dB ｜ "
                f"导频 {rec.get('pilot_pattern', '-')}")
        else:
            self.metrics_var.set("（该次实验没有记录文件）")
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


def main():
    enable_dpi_awareness()
    app = DmtGuiApp()
    app.mainloop()


if __name__ == "__main__":
    main()
