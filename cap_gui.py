# -*- coding: utf-8 -*-
"""CAP Communication System Experiment Platform GUI.

Six tabs:
    1. Waveform & Spectrum   —— TX/RX time-domain waveforms and spectra
    2. CAP Modulation        —— CAP constellation and modulation-related plots
    3. Transmission Results  —— experiment record table and run-history trend
    4. Run Test              —— run the CAP transceiver from the GUI with live log output
    5. Keithley 2400         —— RS-232/USB control of the Keithley 2400 source meter
    6. Grid Scan             —— automated bias vs SNR parameter sweep with CSV/contour output

Data sources:
    data/records/record_<run_id>.json          parameters and results for each run
    data/txdata/*.txt                          saved transmit waveforms
    data/rxdata/*.txt                          saved receive waveforms (when present)

How to run (from the project directory):
    python cap_gui.py

Depends only on numpy / matplotlib / tkinter (bundled with Python); no extra install required.
"""
import ast
import json
import os
import queue
import re
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
import config_cap as cfg
import main_cap
from constellation import modulate
from record import generate_run_id, save_record
from keithley2400_controller import (
    Keithley2400,
    refresh_port_list,
    parse_port_entry,
    K2400ConnectionError,
    K2400CommandError,
    K2400ConfigError,
)
from grid_scan_cap import GridScanner, GridScanConfig, list_grid_scans, load_summary

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

# Preferred UI / plot sans-serif font (ships with Raspberry Pi)
UI_FONT = "Liberation Sans"
UI_FONT_FALLBACKS = ["Microsoft YaHei UI", "Microsoft YaHei", "SimHei", "DejaVu Sans", "Liberation Sans"]

# Preferred monospace font for code/log areas (ships with Raspberry Pi)
MONO_FONT = "Liberation Mono Bold"
MONO_FONT_FALLBACKS = ["Liberation Mono", "DejaVu Sans Mono", "Courier"]

# 绘图字体必须支持中文，否则等高线图/坐标轴标题会乱码
PLOT_FONT_CANDIDATES = [
    "Microsoft YaHei", "SimHei", "SimSun", "STSong",
    "WenQuanYi Micro Hei", "Noto Sans CJK SC", "Source Han Sans SC",
    "DejaVu Sans",
]

import matplotlib.font_manager as fm


def _pick_available_font(candidates):
    """Return the first font from candidates that exists on the system."""
    for name in candidates:
        try:
            fm.findfont(name, fallback_to_default=False)
            return name
        except Exception:
            continue
    return candidates[-1] if candidates else "DejaVu Sans"


FONT_FAMILY = _pick_available_font([UI_FONT] + UI_FONT_FALLBACKS)
FONT_MONO = _pick_available_font([MONO_FONT] + MONO_FONT_FALLBACKS)
PLOT_FONT = _pick_available_font(PLOT_FONT_CANDIDATES)

plt.rcParams["font.sans-serif"] = [PLOT_FONT] + [f for f in PLOT_FONT_CANDIDATES if f != PLOT_FONT]
plt.rcParams["axes.unicode_minus"] = False

PROJECT_ROOT = Path(__file__).resolve().parent
ASSETS_DIR = PROJECT_ROOT / "data" / "codeplot_assets"
RECORDS_DIR = cfg.RECORD_DIR
TXDATA_DIR = cfg.TXDATA_DIR
RXDATA_DIR = cfg.RXDATA_DIR

APP_EMOJI = "🔷"  # emoji used for window icon and title


# ═══════════════════════════════════════════════════════════════════════════════
# High-DPI adaptation (must be called before creating Tk)
# ═══════════════════════════════════════════════════════════════════════════════

def enable_dpi_awareness():
    """Render Windows at actual DPI to avoid tiny/blurry UI on high-DPI screens."""
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
# Color palette (modern light dictionary style)
# ═══════════════════════════════════════════════════════════════════════════════

COLOR_BG = "#F3F5F7"          # window background (light gray)
COLOR_CARD = "#FFFFFF"        # card white
COLOR_BORDER = "#E2E8F0"      # card border
COLOR_PRIMARY = "#164E63"     # primary color (dark cyan)
COLOR_PRIMARY_HOVER = "#0E7490"
COLOR_TEXT = "#1F2937"        # primary text
COLOR_TEXT_DIM = "#64748B"    # secondary text
COLOR_DANGER = "#B91C1C"
COLOR_SELECT = "#164E63"      # selection color


# ═══════════════════════════════════════════════════════════════════════════════
# Data discovery
# ═══════════════════════════════════════════════════════════════════════════════

# Cache updated by CApGuiApp.reload_data so plot builders can locate record params.
_RECORD_BY_RUN = {}


def set_record_by_run(mapping):
    _RECORD_BY_RUN.clear()
    _RECORD_BY_RUN.update(mapping)


def _get_record(run_id):
    return _RECORD_BY_RUN.get(run_id, {})


def _record_tx_path(rec):
    """Return the saved TX waveform path for a CAP record."""
    mode = rec.get("mode", "singleband")
    if mode == "singleband":
        order = rec.get("order", cfg.SB_QAMORDER)
        constellation = rec.get("constellation", cfg.SB_CONSTELLATION)
        return TXDATA_DIR / f"data{order}{constellation}.txt"
    return TXDATA_DIR / "up123_data_for_dnn.txt"


def _record_rx_path(rec):
    """Return the expected RX waveform path for a CAP record."""
    mode = rec.get("mode", "singleband")
    if mode == "singleband":
        order = rec.get("order", cfg.SB_QAMORDER)
        constellation = rec.get("constellation", cfg.SB_CONSTELLATION)
        return RXDATA_DIR / f"OSC_data{order}{constellation}.txt"
    return RXDATA_DIR / "rx_multiband.txt"


def available_plots(run_id, names):
    """Return plot names that actually have data for this run (preserving names order)."""
    rec = _get_record(run_id)
    if not rec:
        return []
    tx_path = _record_tx_path(rec)
    has_tx = tx_path.is_file()
    has_rx = _record_rx_path(rec).is_file()
    out = []
    for n in names:
        if n in ("tx_waveform", "tx_spectrum", "tx_constellation",
                 "constellation_density", "cap_constellation"):
            if has_tx:
                out.append(n)
        elif n in ("rx_waveform", "rx_spectrum", "rx_constellation"):
            # RX plots fall back to TX placeholder when RX is not available
            if has_rx or has_tx:
                out.append(n)
    return out


def _load_signal(path):
    if not path.is_file():
        return None
    try:
        return np.loadtxt(path)
    except Exception:
        return None


def _load_run_signals(run_id):
    """Load TX/RX signals and metadata for a run_id."""
    rec = _get_record(run_id)
    tx_path = _record_tx_path(rec)
    rx_path = _record_rx_path(rec)
    tx = _load_signal(tx_path)
    rx = _load_signal(rx_path)
    if tx is None:
        tx = np.array([])
    if rx is None:
        rx = tx
    mode = rec.get("mode", "singleband")
    fs = cfg.SB_AWG_SAMPLE_RATE if mode == "singleband" else cfg.MB_FS
    return {"tx": tx, "rx": rx, "fs": fs, "rec": rec,
            "has_rx": rx_path.is_file() and rx is not tx}


def _regen_symbols(rec, max_count=5000):
    """Regenerate the TX symbol constellation for a singleband record."""
    if rec.get("mode", "singleband") != "singleband":
        return None
    order = rec.get("order", cfg.SB_QAMORDER)
    constellation = rec.get("constellation", cfg.SB_CONSTELLATION)
    seed = rec.get("seed", 100)
    try:
        np.random.seed(int(seed))
    except Exception:
        pass
    count = min(max_count, cfg.SB_NUMOFSYMBOLS)
    dec = np.random.randint(0, order, size=count)
    return modulate(dec, order, constellation)


def _safe_log10(x):
    return 10 * np.log10(np.maximum(np.asarray(x, dtype=float), 1e-12))


def _write_array_to_sheet(ws, arr, start_row=1, start_col=1):
    """Write a numpy array into an openpyxl sheet; complex numbers split into real/imag columns."""
    if arr.ndim == 0:
        val = arr.item()
        if np.iscomplexobj(arr):
            ws.cell(row=start_row, column=start_col, value=float(val.real))
            ws.cell(row=start_row, column=start_col + 1, value=float(val.imag))
        else:
            ws.cell(row=start_row, column=start_col,
                    value=float(val) if isinstance(val, (int, float, np.number)) else val)
        return
    if start_row > 1:
        if arr.dtype.kind == "c":
            if arr.ndim == 1:
                ws.cell(row=start_row - 1, column=start_col, value="real")
                ws.cell(row=start_row - 1, column=start_col + 1, value="imag")
            else:
                for c in range(arr.shape[1]):
                    ws.cell(row=start_row - 1, column=start_col + c * 2,
                            value=f"col_{c}_real")
                    ws.cell(row=start_row - 1, column=start_col + c * 2 + 1,
                            value=f"col_{c}_imag")
        else:
            if arr.ndim == 1:
                ws.cell(row=start_row - 1, column=start_col, value="value")
            else:
                for c in range(arr.shape[1]):
                    ws.cell(row=start_row - 1, column=start_col + c,
                            value=f"col_{c}")
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
    """Excel sheet names must be <= 31 chars and contain no special characters."""
    invalid = ["\\", "/", "?", "*", "[", "]", ":"]
    for ch in invalid:
        name = name.replace(ch, "_")
    return name[:31]


def _npz_to_xlsx(npz_path: Path, xlsx_path: Path):
    """Export each array in an NPZ file to a separate sheet in Excel."""
    if not _HAS_OPENPYXL:
        raise RuntimeError("openpyxl is missing; run: pip install openpyxl")
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
        ws.cell(row=1, column=1, value=f"Array: {key}")
        ws.cell(row=1, column=2, value=f"Shape: {arr.shape}")
        ws.cell(row=1, column=3, value=f"Dtype: {arr.dtype}")
        try:
            _write_array_to_sheet(ws, arr, start_row=3, start_col=1)
        except Exception as exc:
            errors.append(f"{key}: {exc}")
            ws.cell(row=3, column=1, value=f"Write failed: {exc}")
    wb.save(xlsx_path)
    if errors:
        raise RuntimeError("Some arrays failed to write: " + "; ".join(errors))


def list_records():
    """Read all record_*.json under data/records and return sorted by timestamp ascending."""
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
# Plotting functions
# ═══════════════════════════════════════════════════════════════════════════════

def _spectrum(sig, fs):
    n = len(sig)
    freqs = np.fft.fftshift(np.fft.fftfreq(n, d=1.0 / fs))
    spec = 10 * np.log10(np.abs(np.fft.fftshift(np.fft.fft(sig))) + 1e-12)
    return freqs, spec


def build_tx_waveform(fig, run_id, title):
    sigs = _load_run_signals(run_id)
    tx, fs = sigs["tx"], sigs["fs"]
    ax = fig.add_subplot(111)
    n = min(len(tx), 2000)
    t = np.arange(n) / fs
    ax.plot(t * 1e6, tx[:n], "b.-", linewidth=1, markersize=2)
    ax.set_title(title)
    ax.set_xlabel("时间 (us)")
    ax.set_ylabel("幅度")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()


def build_rx_waveform(fig, run_id, title):
    sigs = _load_run_signals(run_id)
    rx, fs = sigs["rx"], sigs["fs"]
    ax = fig.add_subplot(111)
    n = min(len(rx), 2000)
    t = np.arange(n) / fs
    label = "RX" if sigs["has_rx"] else "TX placeholder"
    ax.plot(t * 1e6, rx[:n], "b.-", linewidth=1, markersize=2)
    ax.set_title(f"{title} ({label})")
    ax.set_xlabel("时间 (us)")
    ax.set_ylabel("幅度")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()


def build_tx_spectrum(fig, run_id, title):
    sigs = _load_run_signals(run_id)
    tx, fs = sigs["tx"], sigs["fs"]
    ax = fig.add_subplot(111)
    n = min(len(tx), 8192)
    freqs, spec = _spectrum(tx[:n], fs)
    unit = "GHz" if fs >= 1e9 else "MHz"
    scale = 1e9 if fs >= 1e9 else 1e6
    ax.plot(freqs / scale, spec, "b-", linewidth=1)
    ax.set_title(title)
    ax.set_xlabel(f"Frequency ({unit})")
    ax.set_ylabel("幅度 (dB)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()


def build_rx_spectrum(fig, run_id, title):
    sigs = _load_run_signals(run_id)
    rx, fs = sigs["rx"], sigs["fs"]
    ax = fig.add_subplot(111)
    n = min(len(rx), 8192)
    freqs, spec = _spectrum(rx[:n], fs)
    unit = "GHz" if fs >= 1e9 else "MHz"
    scale = 1e9 if fs >= 1e9 else 1e6
    label = "RX" if sigs["has_rx"] else "TX placeholder"
    ax.plot(freqs / scale, spec, "b-", linewidth=1)
    ax.set_title(f"{title} ({label})")
    ax.set_xlabel(f"Frequency ({unit})")
    ax.set_ylabel("幅度 (dB)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()


def build_tx_constellation(fig, run_id, title):
    rec = _get_record(run_id)
    iq = _regen_symbols(rec)
    ax = fig.add_subplot(111)
    if iq is not None and len(iq):
        ax.plot(iq.real, iq.imag, "b.", alpha=0.4, markersize=4)
        ax.set_title(title)
    else:
        ax.text(0.5, 0.5, "多频带模式无星座图",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
    ax.set_xlabel("I")
    ax.set_ylabel("Q")
    ax.grid(True, alpha=0.3)
    ax.axis("equal")
    fig.tight_layout()


def build_rx_constellation(fig, run_id, title):
    # CAP does not save equalized RX symbols; show TX constellation as placeholder.
    build_tx_constellation(fig, run_id, title)


def build_constellation_density(fig, run_id, title):
    rec = _get_record(run_id)
    iq = _regen_symbols(rec)
    ax = fig.add_subplot(111)
    if iq is not None and len(iq):
        hb = ax.hexbin(iq.real, iq.imag, gridsize=80, cmap="GnBu", mincnt=1)
        fig.colorbar(hb, ax=ax, label="密度")
    else:
        ax.text(0.5, 0.5, "多频带模式无星座密度图",
                ha="center", va="center", transform=ax.transAxes)
    ax.set_title(title)
    ax.set_xlabel("I")
    ax.set_ylabel("Q")
    ax.axis("equal")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()


def build_records_trend(fig, records, title):
    """Trend across runs: SNR and BER/SER."""
    if not records:
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, "暂无实验记录", ha="center", va="center",
                transform=ax.transAxes)
        return
    xs = list(range(len(records)))
    labels = [r.get("run_id", "")[-6:] for r in records]
    snrs = [r.get("snr_db", 0) for r in records]

    def _ber(rec):
        mode = rec.get("mode", "singleband")
        if mode == "singleband":
            return rec.get("ber", np.nan)
        return rec.get("raw_ber_avg", rec.get("eq_ber_avg", rec.get("nn_ber_avg", np.nan)))

    def _ser(rec):
        mode = rec.get("mode", "singleband")
        if mode == "singleband":
            return rec.get("ser", np.nan)
        raw = rec.get("raw_ser")
        if isinstance(raw, list) and raw:
            return float(np.mean(raw))
        return raw if raw is not None else np.nan

    bers = [max(_ber(r) or 1e-12, 1e-12) for r in records]
    sers = [max(_ser(r) or 1e-12, 1e-12) for r in records]

    ax1 = fig.add_subplot(211)
    ax1.plot(xs, snrs, "g-o", markersize=4, linewidth=1.5, label="SNR (dB)")
    ax1.set_ylabel("SNR (dB)", color="g")
    ax1.set_title(title)
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(xs)
    ax1.set_xticklabels(labels, rotation=45, fontsize=8)

    ax2 = fig.add_subplot(212, sharex=ax1)
    ax2.semilogy(xs, bers, "r-x", markersize=4, linewidth=1.5, label="BER")
    ax2.semilogy(xs, sers, "b-s", markersize=4, linewidth=1.5, label="SER")
    ax2.set_xlabel("实验（按时间顺序）")
    ax2.set_ylabel("误码率")
    ax2.legend()
    ax2.grid(True, which="both", ls="--", alpha=0.3)
    ax2.set_xticks(xs)
    ax2.set_xticklabels(labels, rotation=45, fontsize=8)
    fig.tight_layout()


# ═══════════════════════════════════════════════════════════════════════════════
# Figure catalog: name -> (plot function, title)
# ═══════════════════════════════════════════════════════════════════════════════

FIGURES = {
    # Tab 1: Waveform & Spectrum
    "tx_waveform":   (build_tx_waveform,   "发射时域波形"),
    "rx_waveform":   (build_rx_waveform,   "接收时域波形"),
    "tx_spectrum":   (build_tx_spectrum,   "发射频谱"),
    "rx_spectrum":   (build_rx_spectrum,   "接收频谱"),
    # Tab 2: CAP Modulation
    "tx_constellation":   (build_tx_constellation,   "发射星座图"),
    "rx_constellation":   (build_rx_constellation,   "接收星座图"),
    "constellation_density": (build_constellation_density, "星座密度图"),
}

TAB_WAVEFORM = [
    "tx_waveform", "rx_waveform", "tx_spectrum", "rx_spectrum",
]
TAB_MODULATION = [
    "tx_constellation", "rx_constellation", "constellation_density",
]
TAB_RESULTS = [
    # Transmission Results tab uses the trend plot and any modulation plots.
    "tx_constellation",
]

TREND_KEY = "__records_trend__"   # virtual plot name for run-history trend inside tab 3


# ═══════════════════════════════════════════════════════════════════════════════
# Styles
# ═══════════════════════════════════════════════════════════════════════════════

def apply_styles(root, scale):
    """Modern light theme (customized on top of clam)."""
    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except tk.TclError:
        pass

    f = max(scale, 1.0)
    font_base = (FONT_FAMILY, 10)
    font_bold = (FONT_FAMILY, 10, "bold")
    font_tab = (FONT_FAMILY, 11)
    pad_x = int(round(14 * f))
    pad_y = int(round(8 * f))

    style.configure(".", font=font_base, background=COLOR_BG,
                    foreground=COLOR_TEXT)

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
    style.configure("Pill.TLabel", background=COLOR_PRIMARY,
                    foreground="#FFFFFF", font=font_bold,
                    padding=(pad_x, int(round(4 * f))))
    style.configure("Metrics.TLabel", background=COLOR_BG,
                    foreground=COLOR_PRIMARY_HOVER, font=font_bold)

    style.configure("TNotebook", background=COLOR_BG, borderwidth=0)
    style.configure("TNotebook.Tab", font=font_tab,
                    padding=(pad_x + 6, pad_y),
                    background="#E5EAEF", foreground=COLOR_TEXT)
    style.map("TNotebook.Tab",
              background=[("selected", COLOR_CARD)],
              foreground=[("selected", COLOR_PRIMARY)])

    style.configure("Accent.TButton", font=font_bold,
                    padding=(pad_x, pad_y),
                    background=COLOR_PRIMARY, foreground="#FFFFFF",
                    borderwidth=0, focusthickness=0)
    style.map("Accent.TButton",
              background=[("active", COLOR_PRIMARY_HOVER),
                          ("disabled", "#9FB3BC")],
              foreground=[("disabled", "#E5EAEF")])
    style.configure("TButton", font=font_base, padding=(pad_x, pad_y),
                    background="#E5EAEF", foreground=COLOR_TEXT,
                    borderwidth=0)
    style.map("TButton", background=[("active", "#D5DDE4")])
    style.configure("Danger.TButton", font=font_bold,
                    padding=(pad_x, pad_y),
                    background=COLOR_DANGER, foreground="#FFFFFF",
                    borderwidth=0)
    style.map("Danger.TButton",
              background=[("active", "#DC2626"), ("disabled", "#D1A5A5")])

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

    style.configure("TCombobox", padding=(int(round(8 * f)),
                                          int(round(4 * f))))

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

    style.configure("TSeparator", background=COLOR_BORDER)
    style.configure("Vertical.TScrollbar", background="#D5DDE4",
                    troughcolor=COLOR_BG, borderwidth=0, arrowsize=12)


def make_card(parent, **pack_kwargs):
    """White card container (thin border + padding)."""
    card = tk.Frame(parent, bg=COLOR_CARD,
                    highlightbackground=COLOR_BORDER, highlightthickness=1,
                    bd=0)
    if pack_kwargs:
        card.pack(**pack_kwargs)
    return card


def _set_windows_taskbar_icon():
    """Set Windows taskbar icon: an explicit AppUserModelID is needed to escape the default feather icon."""
    if sys.platform != "win32":
        return
    try:
        import ctypes
        app_id = "UW.APSK.CAP.GUI.v1"
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(app_id)
    except Exception:
        pass


def _create_emoji_icon(emoji: str, size: int = 64):
    """Render an emoji into a window icon, returning (PhotoImage, ico_path)."""
    if not _HAS_PIL:
        return None, None
    try:
        img = Image.new("RGBA", (size, size), (255, 255, 255, 0))
        draw = ImageDraw.Draw(img)
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
        png_path = cfg.DATA_DIR / ".gui_icon.png"
        ico_path = cfg.DATA_DIR / ".gui_icon.ico"
        img.save(png_path)
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
    """Left-side plot list + right-side matplotlib canvas."""

    def __init__(self, parent, app, plot_names, include_trend=False):
        super().__init__(parent)
        self.app = app
        self.plot_names = list(plot_names)
        self.include_trend = include_trend
        f = app.font_scale

        left = make_card(self)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 8))
        ttk.Label(left, text="图形列表", style="Section.TLabel"
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
        ttk.Button(left, text="导出当前图形数据",
                   command=self._export_data
                   ).pack(side=tk.BOTTOM, fill=tk.X, padx=8, pady=(0, 4))
        ttk.Button(left, text="导出全部图形数据",
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
        """Refill the list based on the current run."""
        run_id = self.app.current_run
        self.listbox.delete(0, tk.END)
        self._items = []
        if run_id:
            for name in available_plots(run_id, self.plot_names):
                self.listbox.insert(tk.END, f"  {FIGURES[name][1]}")
                self._items.append(name)
        if self.include_trend and self.app.records:
            self.listbox.insert(tk.END, "  实验趋势 (SNR / BER / SER)")
            self._items.append(TREND_KEY)
        if self._items:
            self.listbox.selection_set(0)
            self._show(self._items[0])
        else:
            self.fig.clear()
            ax = self.fig.add_subplot(111)
            ax.text(0.5, 0.5, "当前实验无此类别数据",
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
                build_records_trend(self.fig, self.app.records, "Experiment Trend")
            else:
                builder, title = FIGURES[name]
                builder(self.fig, self.app.current_run, title)
        except Exception as exc:
            self.fig.clear()
            ax = self.fig.add_subplot(111)
            ax.text(0.5, 0.5, f"Plot failed:\n{exc}", ha="center", va="center",
                    transform=ax.transAxes, color="red")
        self.canvas.draw_idle()

    def _safe_savefig(self, path: Path):
        """Save the figure; if bbox_inches='tight' raises IndexError, fall back to normal save."""
        try:
            self.fig.savefig(path, dpi=cfg.PLOT_DPI, bbox_inches="tight")
        except (IndexError, ValueError):
            self.fig.savefig(path, dpi=cfg.PLOT_DPI)

    def _export_image(self):
        """Export the selected plot as PNG; default save location is data/plots/<run_id>/."""
        sel = self.listbox.curselection()
        if not sel:
            messagebox.showinfo("提示", "请先选择一个图形")
            return
        name = self._items[sel[0]]
        run_id = self.app.current_run
        if not run_id:
            messagebox.showinfo("提示", "当前未选择实验")
            return

        plot_dir = cfg.PLOT_DIR / run_id
        default_name = f"{name}.png"
        path = filedialog.asksaveasfilename(
            title="导出图像",
            initialdir=str(plot_dir),
            initialfile=default_name,
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("PDF", "*.pdf"),
                       ("SVG", "*.svg"), ("All files", "*.*")])
        if not path:
            return
        path = Path(path)
        if path.is_file():
            if not messagebox.askyesno("确认覆盖",
                                       f"File already exists:\n{path}\n\nOverwrite?"):
                return
        try:
            plot_dir.mkdir(parents=True, exist_ok=True)
            self._safe_savefig(path)
            self._save_plot_metadata(name, path)
            messagebox.showinfo("导出成功", f"Saved to:\n{path}")
        except Exception as exc:
            messagebox.showerror("导出失败", str(exc))

    def _save_all_images(self):
        """Save all plots for the current run as PNG into data/plots/<run_id>/."""
        run_id = self.app.current_run
        if not run_id:
            messagebox.showinfo("提示", "当前未选择实验")
            return
        names = available_plots(run_id, self.plot_names)
        if not names:
            messagebox.showinfo("提示", "当前实验没有可保存的图形")
            return

        plot_dir = cfg.PLOT_DIR / run_id
        existing = sorted([p.name for p in plot_dir.glob("*.png")]) if plot_dir.is_dir() else []
        if existing:
            if not messagebox.askyesno(
                    "确认覆盖",
                    f"{len(existing)} image(s) already exist in {plot_dir}.\n\n"
                    f"Overwrite?"):
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
                                f"Saved {saved} image(s) to:\n{plot_dir}")

    def _save_plot_metadata(self, name, png_path: Path):
        """Write the current plot's parameters as JSON."""
        run_id = self.app.current_run
        params = {}
        try:
            arrays = self._gather_plot_arrays(name, run_id)
            params["arrays"] = {
                k: {"shape": list(v.shape), "dtype": str(v.dtype)}
                for k, v in arrays.items()
            }
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

    def _gather_plot_arrays(self, name, run_id):
        """Return a dict of numpy arrays for the given plot name and run."""
        if name == TREND_KEY:
            return {"records": np.array([r.get("run_id", "") for r in self.app.records])}
        sigs = _load_run_signals(run_id)
        tx, rx, fs = sigs["tx"], sigs["rx"], sigs["fs"]
        if name == "tx_waveform":
            return {"t": np.arange(len(tx)), "sig": tx}
        if name == "rx_waveform":
            return {"t": np.arange(len(rx)), "sig": rx}
        if name == "tx_spectrum":
            return {"sig": tx, "fs": np.array(fs)}
        if name == "rx_spectrum":
            return {"sig": rx, "fs": np.array(fs)}
        if name in ("tx_constellation", "rx_constellation", "constellation_density"):
            rec = _get_record(run_id)
            iq = _regen_symbols(rec)
            if iq is None:
                iq = np.array([])
            return {"iq": iq}
        return {}

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
            default_name = f"trend_{run_id}.json"
            path = filedialog.asksaveasfilename(
                title="导出趋势数据",
                initialfile=default_name,
                defaultextension=".json",
                filetypes=[("JSON", "*.json"), ("All files", "*.*")])
            if not path:
                return
            try:
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(self.app.records, f, indent=2,
                              ensure_ascii=False, default=str)
                messagebox.showinfo("导出成功", f"Saved to:\n{path}")
            except Exception as exc:
                messagebox.showerror("导出失败", str(exc))
            return

        arrays = self._gather_plot_arrays(name, run_id)
        if not arrays:
            messagebox.showerror("导出失败", "该图形无可用数据")
            return
        default_name = f"{name}_{run_id}.xlsx"
        path = filedialog.asksaveasfilename(
            title="导出图形数据",
            initialfile=default_name,
            defaultextension=".xlsx",
            filetypes=[("Excel", "*.xlsx"), ("All files", "*.*")])
        if not path:
            return
        try:
            if not _HAS_OPENPYXL:
                raise RuntimeError("openpyxl is missing; run: pip install openpyxl")
            wb = openpyxl.Workbook()
            wb.remove(wb.active)
            for key, arr in arrays.items():
                if arr.ndim > 2:
                    arr = arr.reshape(arr.shape[0], -1)
                sheet_name = _safe_sheet_name(key)
                ws = wb.create_sheet(title=sheet_name)
                ws.cell(row=1, column=1, value=f"Plot: {name}")
                ws.cell(row=1, column=2, value=f"Array: {key}")
                ws.cell(row=1, column=3, value=f"Shape: {arr.shape}")
                _write_array_to_sheet(ws, arr, start_row=3, start_col=1)
            wb.save(path)
            messagebox.showinfo("导出成功", f"Saved to:\n{path}")
        except Exception:
            tb = traceback.format_exc()
            print(tb)
            messagebox.showerror("导出失败", f"Failed to export current plot data:\n{tb}")

    def _export_all_data(self):
        """Export all plot data for the current run to Excel."""
        run_id = self.app.current_run
        if not run_id:
            messagebox.showinfo("提示", "当前未选择实验")
            return
        names = available_plots(run_id, self.plot_names)
        if not names:
            messagebox.showinfo("提示", "当前实验没有可导出的图形数据")
            return

        default_name = f"all_plots_{run_id}.xlsx"
        path = filedialog.asksaveasfilename(
            title="导出全部图形数据",
            initialfile=default_name,
            defaultextension=".xlsx",
            filetypes=[("Excel", "*.xlsx"), ("所有文件", "*.*")])
        if not path:
            return
        try:
            if not _HAS_OPENPYXL:
                raise RuntimeError("openpyxl is missing; run: pip install openpyxl")
            wb = openpyxl.Workbook()
            wb.remove(wb.active)
            for name in names:
                arrays = self._gather_plot_arrays(name, run_id)
                for key, arr in arrays.items():
                    if arr.ndim > 2:
                        arr = arr.reshape(arr.shape[0], -1)
                    sheet_name = _safe_sheet_name(f"{name}_{key}")
                    ws = wb.create_sheet(title=sheet_name)
                    ws.cell(row=1, column=1, value=f"Plot: {name}")
                    ws.cell(row=1, column=2, value=f"Array: {key}")
                    ws.cell(row=1, column=3, value=f"Shape: {arr.shape}")
                    _write_array_to_sheet(ws, arr, start_row=3, start_col=1)
            wb.save(path)
            messagebox.showinfo("导出成功", f"Saved to:\n{path}\n"
                                           f"Exported data for {len(names)} plot(s)")
        except Exception:
            tb = traceback.format_exc()
            print(tb)
            messagebox.showerror("导出失败", f"Failed to export all plot data:\n{tb}")


class ResultsPanel(ttk.Frame):
    """Tab 3: experiment record table on top, result plots for the selected run below."""

    COLUMNS = ("run_id", "time", "mode", "order", "constellation",
               "snr", "ber", "ser", "nn")
    HEADINGS = {
        "run_id": ("Experiment ID", 170),
        "time": ("Time", 150),
        "mode": ("Mode", 90),
        "order": ("Order", 60),
        "constellation": ("Constellation", 90),
        "snr": ("SNR (dB)", 80),
        "ber": ("BER", 110),
        "ser": ("SER", 110),
        "nn": ("NN", 50),
    }

    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app

        top = ttk.LabelFrame(self, text=" 传输实验记录（点击行切换实验） ")
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
        for rec in reversed(self.app.records):   # newest first
            run_id = rec.get("run_id", "")
            ts = rec.get("timestamp", "")[:19].replace("T", " ")
            mode = rec.get("mode", "singleband")
            if mode == "singleband":
                ber = rec.get("ber", np.nan)
                ser = rec.get("ser", np.nan)
            else:
                ber = rec.get("raw_ber_avg", rec.get("eq_ber_avg", rec.get("nn_ber_avg", np.nan)))
                raw_ser = rec.get("raw_ser")
                ser = float(np.mean(raw_ser)) if isinstance(raw_ser, list) and raw_ser else np.nan
            row = (
                run_id, ts,
                mode,
                rec.get("order", ""),
                rec.get("constellation", ""),
                f"{rec.get('snr_db', 0):.1f}",
                f"{ber:.3e}" if ber is not None and not np.isnan(ber) else "N/A",
                f"{ser:.3e}" if ser is not None and not np.isnan(ser) else "N/A",
                "Yes" if rec.get("use_nn") else "No",
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


class RunPanel(ttk.Frame):
    """Tab 4: run the CAP transceiver from the GUI with live log output."""

    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        self._queue = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        f = app.font_scale

        # ── Parameter card ───────────────────────────────────────────
        opt = ttk.LabelFrame(self, text=" 实验参数 ")
        opt.pack(fill=tk.X, padx=2, pady=(2, 8))

        ttk.Label(opt, text="模式:", style="Card.TLabel"
                  ).grid(row=0, column=0, sticky=tk.W, padx=12, pady=(10, 4))
        self.mode_var = tk.StringVar(value="singleband")
        self.mode_combo = ttk.Combobox(
            opt, textvariable=self.mode_var,
            values=["singleband", "multiband"],
            state="readonly", width=16)
        self.mode_combo.grid(row=0, column=1, sticky=tk.W, padx=(4, 16), pady=(10, 4))
        self.mode_combo.bind("<<ComboboxSelected>>", self._on_mode_change)

        ttk.Label(opt, text="调制阶数:", style="Card.TLabel"
                  ).grid(row=1, column=0, sticky=tk.W, padx=12, pady=4)
        self.order_var = tk.IntVar(value=cfg.SB_QAMORDER)
        tk.Spinbox(opt, from_=4, to=128, textvariable=self.order_var, width=16
                   ).grid(row=1, column=1, sticky=tk.W, padx=(4, 16), pady=4)

        ttk.Label(opt, text="星座类型:", style="Card.TLabel"
                  ).grid(row=2, column=0, sticky=tk.W, padx=12, pady=4)
        self.const_var = tk.StringVar(value=cfg.SB_CONSTELLATION)
        ttk.Combobox(opt, textvariable=self.const_var,
                     values=["QAM", "APSK"], state="readonly", width=16
                     ).grid(row=2, column=1, sticky=tk.W, padx=(4, 16), pady=4)

        ttk.Label(opt, text="信噪比 (dB):", style="Card.TLabel"
                  ).grid(row=3, column=0, sticky=tk.W, padx=12, pady=4)
        self.snr_var = tk.DoubleVar(value=cfg.SB_SNR_DB)
        tk.Spinbox(opt, from_=0.0, to=50.0, textvariable=self.snr_var, width=16
                   ).grid(row=3, column=1, sticky=tk.W, padx=(4, 16), pady=4)

        ttk.Label(opt, text="随机种子:", style="Card.TLabel"
                  ).grid(row=4, column=0, sticky=tk.W, padx=12, pady=4)
        self.seed_var = tk.IntVar(value=100)
        tk.Spinbox(opt, from_=0, to=10000, textvariable=self.seed_var, width=16
                   ).grid(row=4, column=1, sticky=tk.W, padx=(4, 16), pady=4)

        self.virtual_var = tk.BooleanVar(value=bool(cfg.USE_VIRTUAL_CHANNEL))
        ttk.Checkbutton(opt, text="使用虚拟信道",
                        variable=self.virtual_var
                        ).grid(row=5, column=1, sticky=tk.W, padx=(4, 16), pady=4)

        self.nn_var = tk.BooleanVar(value=False)
        self.nn_check = ttk.Checkbutton(opt, text="使用 NN 后均衡器（仅多频带）",
                                        variable=self.nn_var)
        self.nn_check.grid(row=6, column=1, sticky=tk.W, padx=(4, 16), pady=(4, 10))

        btn_bar = tk.Frame(opt, bg=COLOR_CARD)
        btn_bar.grid(row=7, column=0, columnspan=2, sticky=tk.W,
                     padx=12, pady=(0, 8))
        self.run_btn = ttk.Button(btn_bar, text="▶  运行仿真",
                                  style="Accent.TButton",
                                  command=self.start_run)
        self.run_btn.pack(side=tk.LEFT, padx=(0, 8))
        self.quick_btn = ttk.Button(btn_bar, text="⚡ 快速绘图（仅发射）",
                                    command=self._quick_plot)
        self.quick_btn.pack(side=tk.LEFT, padx=(0, 8))
        self.run_status = ttk.Label(btn_bar, text="就绪", style="DimCard.TLabel")
        self.run_status.pack(side=tk.LEFT, padx=16)

        # ── Bottom: log card ─────────────────────────────────────────
        log_card = ttk.LabelFrame(self, text=" 运行日志 ")
        log_card.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
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
            "Hint: configure CAP parameters, then click Run Simulation.\n"
            "Singleband mode runs one CAP carrier; multiband runs three bands.\n"
            "Use virtual channel when no hardware is connected.\n"
            "Data will auto-refresh and switch to the latest experiment after the run completes.\n", "head")

        self._on_mode_change()

    def _on_mode_change(self, _event=None):
        mode = self.mode_var.get()
        if mode == "singleband":
            self.order_var.set(cfg.SB_QAMORDER)
            self.const_var.set(cfg.SB_CONSTELLATION)
            self.snr_var.set(cfg.SB_SNR_DB)
            self.nn_check.configure(state=tk.DISABLED)
            self.nn_var.set(False)
        else:
            self.order_var.set(cfg.MB_M)
            self.const_var.set(cfg.MB_CONSTELLATION)
            self.snr_var.set(cfg.MB_SNR_DB)
            self.nn_check.configure(state=tk.NORMAL)

    def _append_log(self, text, tag=None):
        self.log_text.configure(state=tk.NORMAL)
        if tag:
            self.log_text.insert(tk.END, text, tag)
        else:
            self.log_text.insert(tk.END, text)
        self.log_text.see(tk.END)
        self.log_text.configure(state=tk.DISABLED)

    def start_run(self):
        if self._thread is not None and self._thread.is_alive():
            return
        self.run_btn.configure(state=tk.DISABLED)
        self.run_status.configure(text="运行中…")
        self.app.set_running(True)
        self._append_log("\n========== Starting CAP Simulation ==========\n", "head")
        self._thread = threading.Thread(target=self._run_thread, daemon=True)
        self._thread.start()
        self.after(100, self._poll)

    def _run_thread(self):
        try:
            mode = self.mode_var.get()
            order = self.order_var.get()
            constellation = self.const_var.get()
            snr = self.snr_var.get()
            seed = self.seed_var.get()
            cfg.USE_VIRTUAL_CHANNEL = 1 if self.virtual_var.get() else 0

            run_id = generate_run_id()
            if mode == "singleband":
                record = main_cap.run_singleband(
                    numofsymbols=cfg.SB_NUMOFSYMBOLS,
                    order=order,
                    constellation=constellation,
                    snr_db=snr,
                    seed=seed,
                )
                record.update({
                    "run_id": run_id,
                    "mode": "singleband",
                    "order": order,
                    "constellation": constellation,
                    "snr_db": snr,
                    "seed": seed,
                    "use_virtual_channel": bool(cfg.USE_VIRTUAL_CHANNEL),
                    "use_nn": False,
                })
                self._queue.put(("log", f"Single-band done: SER={record['ser']:.4e}, BER={record['ber']:.4e}"))
            else:
                use_nn = self.nn_var.get()
                record = main_cap.run_multiband(
                    numofsymbols=cfg.MB_NUMOFSYMBOLS,
                    order=order,
                    constellation=constellation,
                    snr_db=snr,
                    seed=seed,
                    use_nn=use_nn,
                )
                record.update({
                    "run_id": run_id,
                    "mode": "multiband",
                    "order": order,
                    "constellation": constellation,
                    "snr_db": snr,
                    "seed": seed,
                    "use_virtual_channel": bool(cfg.USE_VIRTUAL_CHANNEL),
                    "use_nn": use_nn,
                })
                raw_avg = record.get("raw_ber_avg", 1.0)
                lms_avg = record.get("eq_ber_avg", None)
                nn_avg = record.get("nn_ber_avg", None)
                log_msg = f"Multi-band done: raw BER avg={raw_avg:.4e}"
                if lms_avg is not None:
                    log_msg += f", LMS BER avg={lms_avg:.4e}"
                if nn_avg is not None:
                    log_msg += f", NN BER avg={nn_avg:.4e}"
                self._queue.put(("log", log_msg))

            save_record(run_id, record, cfg.RECORD_DIR)
            self._queue.put(("done", run_id))
        except Exception as exc:
            self._queue.put(("error", exc))

    def _poll(self):
        try:
            while True:
                kind, payload = self._queue.get_nowait()
                if kind == "log":
                    self._append_log(payload)
                elif kind == "error":
                    self._append_log(f"\nError: {payload}\n", "err")
                    self._on_done(None)
                elif kind == "done":
                    run_id = payload
                    self._append_log("\n========== Simulation Completed ==========\n", "head")
                    self._on_done(run_id)
                    return
        except queue.Empty:
            pass
        if self._thread is not None and self._thread.is_alive():
            self.after(100, self._poll)

    def _on_done(self, run_id):
        self.run_btn.configure(state=tk.NORMAL)
        self.app.set_running(False)
        if run_id:
            self.run_status.configure(text="测试完成 ✔")
            self.app.reload_data(select_latest=True)
        else:
            self.run_status.configure(text="测试失败")

    def _quick_plot(self):
        """Generate a TX-only preview without running a full simulation."""
        try:
            mode = self.mode_var.get()
            order = self.order_var.get()
            constellation = self.const_var.get()
            np.random.seed(self.seed_var.get())
            if mode == "singleband":
                num = min(cfg.SB_NUMOFSYMBOLS, 5000)
                dec = np.random.randint(0, order, size=num)
                sym = modulate(dec, order, constellation)
                tx, _, _, _ = main_cap.cap_core.generate_singleband_cap(
                    sym, cfg.SB_AWG_SAMPLE_RATE, cfg.SB_SYMBOL_RATE,
                    cfg.SB_ALPHA, cfg.SB_SUBCAR, cfg.SB_STARTFREQ, cfg.SB_TAPS
                )
                data = {"tx": tx, "rx": tx, "sym": sym, "mode": mode, "fs": cfg.SB_AWG_SAMPLE_RATE}
            else:
                num = min(cfg.MB_NUMOFSYMBOLS, 2048)
                rng = np.random.default_rng(self.seed_var.get())
                syms = [modulate(rng.integers(0, order, size=num), order, constellation) for _ in range(3)]
                tx, _, _, _, _ = main_cap.cap_core.generate_multiband_cap(
                    syms, cfg.MB_RS, cfg.MB_FS, cfg.MB_ROLLOFF, cfg.MB_CF, cfg.MB_SPAN, cfg.MB_SHAPE
                )
                data = {"tx": tx, "rx": tx, "sym": None, "mode": mode, "fs": cfg.MB_FS}
            self.app.show_quick_plots(data)
            self._append_log("Quick plot (TX only) generated.")
        except Exception as exc:
            messagebox.showerror("绘图错误", str(exc))
            self._append_log(f"Quick plot error: {exc}", "err")


class Keithley2400Panel(ttk.Frame):
    """GUI panel for controlling a Keithley 2400 over RS-232 or USB-to-RS232."""

    SOURCE_MODES = ["voltage", "current"]

    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        self.instrument: Optional[Keithley2400] = None

        self.port_var = tk.StringVar(value=cfg.K2400_PORT)
        self.interface_var = tk.StringVar(value=getattr(cfg, "K2400_INTERFACE", "rs232"))
        self.source_mode_var = tk.StringVar(value=cfg.K2400_SOURCE_MODE)
        self.level_var = tk.StringVar(value=str(cfg.K2400_LEVEL))
        self.compliance_var = tk.StringVar(value=str(cfg.K2400_COMPLIANCE))
        self.nplc_var = tk.StringVar(value=str(cfg.K2400_NPLC))
        self.auto_range_var = tk.BooleanVar(value=True)
        self.output_var = tk.BooleanVar(value=False)

        self._build_ui()
        self._refresh_ports()

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

        hdr = tk.Frame(scrollable, bg=COLOR_BG)
        hdr.pack(fill=tk.X, padx=16, pady=(14, 6))
        ttk.Label(hdr, text="Keithley 2400 源表控制",
                  style="Title.TLabel").pack(side=tk.LEFT)
        self.status_lbl = ttk.Label(hdr, text="未连接", style="Pill.TLabel")
        self.status_lbl.pack(side=tk.RIGHT)

        card = make_card(scrollable)
        card.pack(fill=tk.X, padx=16, pady=6)
        inner = tk.Frame(card, bg=COLOR_CARD)
        inner.pack(fill=tk.X, padx=12, pady=12)

        ttk.Label(inner, text="通信接口", style="Section.TLabel").grid(
            row=0, column=0, sticky=tk.W, padx=(0, 8), pady=4)
        self.interface_combo = ttk.Combobox(inner, textvariable=self.interface_var,
                                            values=["rs232", "gpib"], state="readonly", width=12)
        self.interface_combo.bind("<<ComboboxSelected>>", self._on_interface_change)
        self.interface_combo.grid(row=0, column=1, sticky=tk.W, padx=(0, 8), pady=4)

        ttk.Label(inner, text="端口 / GPIB", style="Section.TLabel").grid(
            row=1, column=0, sticky=tk.W, padx=(0, 8), pady=4)
        self.port_combo = ttk.Combobox(inner, textvariable=self.port_var,
                                       values=[], width=40, state="readonly")
        self.port_combo.grid(row=1, column=1, sticky=tk.W, padx=(0, 8), pady=4)
        ttk.Button(inner, text="⟳ 刷新", command=self._refresh_ports).grid(
            row=1, column=2, padx=(0, 8), pady=4)

        ttk.Label(inner, text="波特率", style="Section.TLabel").grid(
            row=2, column=0, sticky=tk.W, padx=(0, 8), pady=4)
        self.baud_combo = ttk.Combobox(inner, values=[9600, 19200, 38400, 57600, 115200],
                                       width=12, state="readonly")
        self.baud_combo.set(str(cfg.K2400_BAUDRATE))
        self.baud_combo.grid(row=2, column=1, sticky=tk.W, padx=(0, 8), pady=4)

        self.conn_btn = ttk.Button(inner, text="连接", command=self._toggle_connect)
        self.conn_btn.grid(row=2, column=2, padx=(0, 8), pady=4)

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

    def _log(self, text: str, tag: Optional[str] = None):
        self.log_text.configure(state=tk.NORMAL)
        if tag:
            self.log_text.insert(tk.END, text + "\n", tag)
        else:
            self.log_text.insert(tk.END, text + "\n")
        self.log_text.see(tk.END)
        self.log_text.configure(state=tk.DISABLED)

    def _refresh_ports(self):
        ports = refresh_port_list(interface=self.interface_var.get())
        self.port_combo["values"] = ports
        if ports and not self.port_var.get():
            self.port_var.set(parse_port_entry(ports[0]))
        elif ports:
            current = self.port_var.get()
            matching = [p for p in ports if parse_port_entry(p) == current]
            if not matching:
                self.port_var.set(parse_port_entry(ports[0]))

    def _on_interface_change(self, _event=None):
        self._refresh_ports()

    def _on_mode_change(self, _event=None):
        mode = self.source_mode_var.get()
        if mode == "voltage":
            self.level_unit_lbl.configure(text="V")
            self.comp_unit_lbl.configure(text="A")
        else:
            self.level_unit_lbl.configure(text="A")
            self.comp_unit_lbl.configure(text="V")

    def _toggle_connect(self):
        if self.instrument is not None and self.instrument.connected:
            self._disconnect()
        else:
            self._connect()

    def _connect(self):
        port = self.port_var.get()
        if not port:
            messagebox.showwarning("未选端口", "请先选择端口 / GPIB 资源。")
            return
        baud = int(self.baud_combo.get() or cfg.K2400_BAUDRATE)
        interface = self.interface_var.get()
        try:
            self.instrument = Keithley2400(port=port, baudrate=baud,
                                           timeout=cfg.K2400_TIMEOUT,
                                           interface=interface)
            self.instrument.connect()
            self.conn_btn.configure(text="断开连接")
            self.status_lbl.configure(text=f"Connected ({port})")
            self._log(f"Connected to {port} at {baud} baud")
            self._apply_settings()
        except K2400ConnectionError as exc:
            messagebox.showerror("连接失败", str(exc))
            self._log(f"Connection failed: {exc}", "err")
            self.instrument = None
        except Exception as exc:
            messagebox.showerror("连接错误", str(exc))
            self._log(f"Connection error: {exc}", "err")
            self.instrument = None

    def _disconnect(self):
        if self.instrument is not None:
            try:
                self.instrument.disconnect()
            except Exception as exc:
                self._log(f"Disconnect error: {exc}", "err")
            finally:
                self.instrument = None
        self.conn_btn.configure(text="连接")
        self.status_lbl.configure(text="未连接")
        self.output_var.set(False)
        self.out_btn.configure(text="输出开")
        self._log("Disconnected")

    def _apply_settings(self):
        if self.instrument is None or not self.instrument.connected:
            self._log("Not connected; settings not applied.")
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

            self._log(f"Settings applied: {mode} source, level={level}, "
                      f"compliance={compliance}, NPLC={nplc}, auto_range={auto_range}")
        except ValueError:
            messagebox.showerror("数值无效", "电平、限值和 NPLC 必须是数字。")
        except K2400ConfigError as exc:
            messagebox.showerror("配置错误", str(exc))
        except Exception as exc:
            messagebox.showerror("应用设置失败", str(exc))
            self._log(f"Apply settings failed: {exc}", "err")

    def _toggle_output(self):
        if self.instrument is None or not self.instrument.connected:
            self._log("Not connected.")
            return
        try:
            if self.output_var.get():
                self.instrument.output_off()
                self.output_var.set(False)
                self.out_btn.configure(text="输出开")
                self._log("Output OFF")
            else:
                self.instrument.output_on()
                self.output_var.set(True)
                self.out_btn.configure(text="输出关")
                self._log("Output ON")
        except Exception as exc:
            messagebox.showerror("输出控制失败", str(exc))
            self._log(f"Output control failed: {exc}", "err")

    def _measure(self):
        if self.instrument is None or not self.instrument.connected:
            self._log("Not connected.")
            return
        try:
            data = self.instrument.measure()
            text = (f"V={data['voltage']:.6e} V, "
                    f"I={data['current']:.6e} A, "
                    f"R={data['resistance']:.6e} Ω, "
                    f"t={data['timestamp']:.6f} s")
            self.last_measure_lbl.configure(text=f"Last measure: {text}")
            self._log(f"Measure: {text}")
        except K2400CommandError as exc:
            messagebox.showerror("测量失败", str(exc))
            self._log(f"Measurement failed: {exc}", "err")
        except Exception as exc:
            messagebox.showerror("测量错误", str(exc))
            self._log(f"Measurement error: {exc}", "err")

    def _reset(self):
        if self.instrument is None or not self.instrument.connected:
            self._log("Not connected.")
            return
        try:
            self.instrument.reset()
            self._log("Instrument reset.")
            self._apply_settings()
        except Exception as exc:
            messagebox.showerror("复位失败", str(exc))
            self._log(f"Reset failed: {exc}", "err")

    def on_close(self):
        """Call when the application exits to safely turn off output."""
        self._disconnect()


class GridScanPanel(ttk.Frame):
    """GUI panel for automated 2-D parameter sweeps (Keithley bias vs virtual-channel SNR)."""

    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        self.scanner: Optional[GridScanner] = None
        self.scan_thread: Optional[threading.Thread] = None
        self.stop_requested = False

        self.p1_name_var = tk.StringVar(value=cfg.GRID_SCAN_PARAM1_NAME)
        self.p1_mode_var = tk.StringVar(value=cfg.GRID_SCAN_PARAM1_MODE)
        self.p1_start_var = tk.StringVar(value=str(cfg.GRID_SCAN_PARAM1_START))
        self.p1_stop_var = tk.StringVar(value=str(cfg.GRID_SCAN_PARAM1_STOP))
        self.p1_step_var = tk.StringVar(value=str(cfg.GRID_SCAN_PARAM1_STEP))
        self.p2_start_var = tk.StringVar(value="20.0")
        self.p2_stop_var = tk.StringVar(value="35.0")
        self.p2_step_var = tk.StringVar(value="1.0")
        self.run_mode_var = tk.StringVar(value=cfg.GRID_SCAN_RUN_MODE)
        self.repeats_var = tk.StringVar(value=str(cfg.GRID_SCAN_REPEATS))
        self.order_var = tk.StringVar(value=str(cfg.SB_QAMORDER))
        self.constellation_var = tk.StringVar(value=cfg.SB_CONSTELLATION)
        self.use_nn_var = tk.BooleanVar(value=False)
        self.use_virtual_channel_var = tk.BooleanVar(value=True)
        self.port_var = tk.StringVar(value=cfg.K2400_PORT)
        self.interface_var = tk.StringVar(value=getattr(cfg, "K2400_INTERFACE", "rs232"))
        self.baud_var = tk.StringVar(value=str(cfg.K2400_BAUDRATE))
        self.compliance_var = tk.StringVar(value=str(cfg.K2400_COMPLIANCE))
        self.nplc_var = tk.StringVar(value=str(cfg.K2400_NPLC))

        self._build_ui()
        self._refresh_ports()
        self._refresh_scan_list()
        self._on_p1_mode_change()

    def _build_ui(self):
        paned = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=12, pady=12)

        left = tk.Frame(paned, bg=COLOR_BG)
        paned.add(left, weight=1)

        hdr = tk.Frame(left, bg=COLOR_BG)
        hdr.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(hdr, text="网格扫描配置", style="Title.TLabel").pack(anchor=tk.W)

        card = make_card(left)
        card.pack(fill=tk.X, pady=6)
        inner = tk.Frame(card, bg=COLOR_CARD)
        inner.pack(fill=tk.X, padx=12, pady=12)

        ttk.Label(inner, text="参数 1 (Keithley)", style="Section.TLabel").grid(
            row=0, column=0, columnspan=4, sticky=tk.W, pady=(0, 6))

        ttk.Label(inner, text="名称").grid(row=1, column=0, sticky=tk.W, padx=(0, 4))
        ttk.Entry(inner, textvariable=self.p1_name_var, width=14).grid(
            row=1, column=1, sticky=tk.W, padx=(0, 8), pady=2)
        ttk.Label(inner, text="模式").grid(row=1, column=2, sticky=tk.W, padx=(0, 4))
        self.p1_mode_combo = ttk.Combobox(inner, textvariable=self.p1_mode_var,
                                          values=["voltage", "current"], state="readonly", width=10)
        self.p1_mode_combo.bind("<<ComboboxSelected>>", self._on_p1_mode_change)
        self.p1_mode_combo.grid(row=1, column=3, sticky=tk.W, pady=2)
        self.p1_unit_lbl = ttk.Label(inner, text="V")
        self.p1_unit_lbl.grid(row=1, column=4, sticky=tk.W, padx=(4, 0))

        ttk.Label(inner, text="起始").grid(row=2, column=0, sticky=tk.W, padx=(0, 4))
        ttk.Entry(inner, textvariable=self.p1_start_var, width=10).grid(
            row=2, column=1, sticky=tk.W, padx=(0, 8), pady=2)
        ttk.Label(inner, text="终止").grid(row=2, column=2, sticky=tk.W, padx=(0, 4))
        ttk.Entry(inner, textvariable=self.p1_stop_var, width=10).grid(
            row=2, column=3, sticky=tk.W, pady=2)

        ttk.Label(inner, text="步进").grid(row=3, column=0, sticky=tk.W, padx=(0, 4))
        ttk.Entry(inner, textvariable=self.p1_step_var, width=10).grid(
            row=3, column=1, sticky=tk.W, padx=(0, 8), pady=2)

        ttk.Label(inner, text="参数 2 (SNR dB)", style="Section.TLabel").grid(
            row=4, column=0, columnspan=4, sticky=tk.W, pady=(12, 6))

        ttk.Label(inner, text="起始").grid(row=5, column=0, sticky=tk.W, padx=(0, 4))
        ttk.Entry(inner, textvariable=self.p2_start_var, width=10).grid(
            row=5, column=1, sticky=tk.W, padx=(0, 8), pady=2)
        ttk.Label(inner, text="终止").grid(row=5, column=2, sticky=tk.W, padx=(0, 4))
        ttk.Entry(inner, textvariable=self.p2_stop_var, width=10).grid(
            row=5, column=3, sticky=tk.W, pady=2)

        ttk.Label(inner, text="步进").grid(row=6, column=0, sticky=tk.W, padx=(0, 4))
        ttk.Entry(inner, textvariable=self.p2_step_var, width=10).grid(
            row=6, column=1, sticky=tk.W, padx=(0, 8), pady=2)

        ttk.Label(inner, text="处理流程", style="Section.TLabel").grid(
            row=7, column=0, columnspan=4, sticky=tk.W, pady=(12, 6))

        ttk.Label(inner, text="运行模式").grid(row=8, column=0, sticky=tk.W, padx=(0, 4))
        ttk.Combobox(inner, textvariable=self.run_mode_var,
                     values=["singleband", "multiband"], state="readonly", width=12).grid(
            row=8, column=1, sticky=tk.W, padx=(0, 8), pady=2)
        ttk.Label(inner, text="重复次数").grid(row=8, column=2, sticky=tk.W, padx=(0, 4))
        ttk.Entry(inner, textvariable=self.repeats_var, width=10).grid(
            row=8, column=3, sticky=tk.W, pady=2)

        ttk.Label(inner, text="调制阶数").grid(row=9, column=0, sticky=tk.W, padx=(0, 4))
        ttk.Entry(inner, textvariable=self.order_var, width=10).grid(
            row=9, column=1, sticky=tk.W, padx=(0, 8), pady=2)
        ttk.Label(inner, text="星座类型").grid(row=9, column=2, sticky=tk.W, padx=(0, 4))
        ttk.Combobox(inner, textvariable=self.constellation_var,
                     values=["QAM", "APSK"], state="readonly", width=10).grid(
            row=9, column=3, sticky=tk.W, pady=2)

        ttk.Checkbutton(inner, text="虚拟信道", variable=self.use_virtual_channel_var).grid(
            row=10, column=0, columnspan=2, sticky=tk.W, pady=2)
        ttk.Checkbutton(inner, text="使用 NN 后均衡器", variable=self.use_nn_var).grid(
            row=10, column=2, columnspan=2, sticky=tk.W, pady=2)

        ttk.Label(inner, text="Keithley 连接", style="Section.TLabel").grid(
            row=11, column=0, columnspan=4, sticky=tk.W, pady=(12, 6))

        ttk.Label(inner, text="通信接口").grid(row=12, column=0, sticky=tk.W, padx=(0, 4))
        self.interface_combo = ttk.Combobox(inner, textvariable=self.interface_var,
                                            values=["rs232", "gpib"], state="readonly", width=10)
        self.interface_combo.bind("<<ComboboxSelected>>", self._on_interface_change)
        self.interface_combo.grid(row=12, column=1, sticky=tk.W, padx=(0, 8), pady=2)

        ttk.Label(inner, text="端口 / GPIB").grid(row=13, column=0, sticky=tk.W, padx=(0, 4))
        self.port_combo = ttk.Combobox(inner, textvariable=self.port_var,
                                       values=[], width=18, state="readonly")
        self.port_combo.grid(row=13, column=1, sticky=tk.W, padx=(0, 8), pady=2)
        ttk.Button(inner, text="⟳ 刷新", command=self._refresh_ports).grid(
            row=13, column=2, columnspan=2, sticky=tk.W, pady=2)

        ttk.Label(inner, text="波特率").grid(row=14, column=0, sticky=tk.W, padx=(0, 4))
        ttk.Entry(inner, textvariable=self.baud_var, width=10).grid(
            row=14, column=1, sticky=tk.W, padx=(0, 8), pady=2)
        ttk.Label(inner, text="限值").grid(row=14, column=2, sticky=tk.W, padx=(0, 4))
        ttk.Entry(inner, textvariable=self.compliance_var, width=10).grid(
            row=14, column=3, sticky=tk.W, pady=2)
        self.p1_comp_unit_lbl = ttk.Label(inner, text="A")
        self.p1_comp_unit_lbl.grid(row=14, column=4, sticky=tk.W, padx=(4, 0))

        ttk.Label(inner, text="NPLC").grid(row=15, column=0, sticky=tk.W, padx=(0, 4))
        ttk.Entry(inner, textvariable=self.nplc_var, width=10).grid(
            row=15, column=1, sticky=tk.W, padx=(0, 8), pady=2)

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

        right = tk.Frame(paned, bg=COLOR_BG)
        paned.add(right, weight=2)

        ttk.Label(right, text="历史网格扫描", style="Title.TLabel").pack(anchor=tk.W, pady=(0, 8))

        sel = tk.Frame(right, bg=COLOR_BG)
        sel.pack(fill=tk.X, pady=(0, 6))
        self.scan_var = tk.StringVar()
        self.scan_combo = ttk.Combobox(sel, textvariable=self.scan_var,
                                       values=[], state="readonly", width=40)
        self.scan_combo.pack(side=tk.LEFT, padx=(0, 8))
        self.scan_combo.bind("<<ComboboxSelected>>", self._on_scan_selected)
        ttk.Button(sel, text="⟳ 刷新", command=self._refresh_scan_list).pack(side=tk.LEFT)

        self.tree = ttk.Treeview(right, show="headings", height=8)
        self.tree.pack(fill=tk.X, pady=(0, 8))
        self.tree["columns"] = ("point", "param1", "vpp", "ber", "ser", "snr_db", "rate_gbps")
        for col in self.tree["columns"]:
            self.tree.heading(col, text=col.replace("_", " ").title())
            self.tree.column(col, width=80, anchor=tk.CENTER)
        self.tree.column("param1", width=100)
        self.tree.column("vpp", width=80)

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

    def _refresh_ports(self):
        ports = refresh_port_list(interface=self.interface_var.get())
        self.port_combo["values"] = ports
        if ports and not self.port_var.get():
            self.port_var.set(parse_port_entry(ports[0]))

    def _on_interface_change(self, _event=None):
        self._refresh_ports()

    def _on_p1_mode_change(self, _event=None):
        mode = self.p1_mode_var.get()
        if mode == "voltage":
            self.p1_unit_lbl.configure(text="V")
            self.p1_comp_unit_lbl.configure(text="A")
        else:
            self.p1_unit_lbl.configure(text="A")
            self.p1_comp_unit_lbl.configure(text="V")

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

        # colorbar 是独立的 Axes，ax.clear() 不会移除它，必须先手动删除，
        # 否则每次切换指标都会多一条 colorbar，主图被越挤越小
        if getattr(self, "_cbar", None) is not None:
            try:
                self._cbar.remove()
            except Exception:
                pass
            self._cbar = None
        self.ax.clear()
        header, rows = load_summary(scan_id)
        if header is None or not rows:
            self.ax.set_title("无数据")
            self.canvas.draw()
            return

        col_idx = {h: i for i, h in enumerate(header)}
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

        metric_labels = {"snr_db": "SNR (dB)", "ber": "BER", "ser": "SER"}
        metric_label = metric_labels.get(metric, metric)

        if np.any(np.isfinite(Zi)):
            levels = np.linspace(np.nanmin(Zi), np.nanmax(Zi), 20)
            im = self.ax.contourf(Xi, Yi, Zi, levels=levels, cmap="viridis", extend="both")
            self._cbar = self.fig.colorbar(im, ax=self.ax, label=metric_label)
        self.ax.set_xlabel(header[1] if len(header) > 1 else "param1")
        self.ax.set_ylabel("SNR (dB)")
        self.ax.set_title(f"{metric_label} 网格扫描 ({scan_id})")
        self.fig.tight_layout()
        self.canvas.draw()

    def _build_config(self) -> GridScanConfig:
        return GridScanConfig(
            scan_id="",
            param1_name=self.p1_name_var.get().strip() or "bias_voltage",
            param1_mode=self.p1_mode_var.get(),
            param1_start=float(self.p1_start_var.get()),
            param1_stop=float(self.p1_stop_var.get()),
            param1_step=float(self.p1_step_var.get()),
            param2_name="snr_db",
            param2_start=float(self.p2_start_var.get()),
            param2_stop=float(self.p2_stop_var.get()),
            param2_step=float(self.p2_step_var.get()),
            run_mode=self.run_mode_var.get(),
            step_repeats=int(self.repeats_var.get()),
            order=int(self.order_var.get()),
            constellation=self.constellation_var.get(),
            use_nn=self.use_nn_var.get(),
            use_virtual_channel=self.use_virtual_channel_var.get(),
            keithley_port=self.port_var.get(),
            keithley_baudrate=int(self.baud_var.get()),
            keithley_timeout=cfg.K2400_TIMEOUT,
            keithley_compliance=float(self.compliance_var.get()),
            keithley_nplc=float(self.nplc_var.get()),
            keithley_interface=self.interface_var.get(),
        )

    def _start_scan(self):
        if self.scan_thread is not None and self.scan_thread.is_alive():
            return
        try:
            gcfg = self._build_config()
        except Exception as exc:
            messagebox.showerror("配置无效", str(exc))
            return

        self.stop_requested = False
        self.run_btn.configure(state=tk.DISABLED)
        self.stop_btn.configure(state=tk.NORMAL)
        self.progress_var.set(0.0)
        self.progress_lbl.configure(text=f"Scan {gcfg.scan_id} starting...")

        def progress_cb(fraction):
            self.after(0, lambda: self.progress_var.set(fraction))

        def run():
            try:
                scanner = GridScanner(gcfg)
                self.scanner = scanner
                csv_path = scanner.run(progress_callback=progress_cb)
                self.after(0, lambda: self._scan_done(csv_path, gcfg.scan_id))
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
        self.progress_lbl.configure(text=f"Done: {csv_path}")
        self.progress_var.set(1.0)
        self._refresh_scan_list()
        self.scan_var.set(scan_id)
        self._on_scan_selected()
        messagebox.showinfo("网格扫描完成", f"Summary saved to:\n{csv_path}")

    def _scan_error(self, exc: Exception):
        self.run_btn.configure(state=tk.NORMAL)
        self.stop_btn.configure(state=tk.DISABLED)
        self.progress_lbl.configure(text=f"Error: {exc}")
        messagebox.showerror("网格扫描失败", str(exc))


# ═══════════════════════════════════════════════════════════════════════════════
# Main window
# ═══════════════════════════════════════════════════════════════════════════════

class CApGuiApp(tk.Tk):
    def __init__(self):
        super().__init__()

        _set_windows_taskbar_icon()

        dpi = self.winfo_fpixels("1i")
        self.font_scale = max(dpi / 96.0, 1.0)
        self.tk.call("tk", "scaling", dpi / 72.0)

        self.title(f"{APP_EMOJI} CAP Communication System Experiment Platform")
        self._icon, self._icon_ico = _create_emoji_icon(APP_EMOJI)
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

        self.runs = []
        self.records = []
        self.current_run = None
        self._record_by_run = {}
        self._running = False

        header = tk.Frame(self, bg=COLOR_BG)
        header.pack(fill=tk.X, padx=16, pady=(14, 6))
        ttk.Label(header, text=f"{APP_EMOJI} CAP Communication System Experiment Platform",
                  style="Title.TLabel").pack(side=tk.LEFT)
        self.pill_runs = ttk.Label(header, style="Pill.TLabel")
        self.pill_runs.pack(side=tk.RIGHT, padx=(8, 0))
        self.pill_records = ttk.Label(header, style="Pill.TLabel")
        self.pill_records.pack(side=tk.RIGHT)

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

        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=16, pady=(0, 8))

        tab1 = ttk.Frame(self.notebook)
        self.notebook.add(tab1, text="  📈 波形与频谱  ")
        self.panel_wave = PlotPanel(tab1, self, TAB_WAVEFORM)
        self.panel_wave.pack(fill=tk.BOTH, expand=True, padx=2, pady=6)

        tab2 = ttk.Frame(self.notebook)
        self.notebook.add(tab2, text="  🎛 CAP 调制  ")
        self.panel_mod = PlotPanel(tab2, self, TAB_MODULATION)
        self.panel_mod.pack(fill=tk.BOTH, expand=True, padx=2, pady=6)

        tab3 = ResultsPanel(self.notebook, self)
        self.notebook.add(tab3, text="  📊 传输实验结果  ")
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

        self.notebook.select(3)

        self.status_var = tk.StringVar()
        status = tk.Label(self, textvariable=self.status_var, anchor=tk.W,
                          bg=COLOR_BG, fg=COLOR_TEXT_DIM, bd=0,
                          font=(FONT_FAMILY, 9))
        status.pack(fill=tk.X, side=tk.BOTTOM, padx=16, pady=(0, 8))

        self.protocol("WM_DELETE_WINDOW", self._on_close)

        self.reload_data()

    def reload_data(self, select_latest=False):
        self.records = list_records()
        self._record_by_run = {r.get("run_id"): r for r in self.records}
        set_record_by_run(self._record_by_run)
        seen = set()
        self.runs = []
        for rec in reversed(self.records):
            run_id = rec.get("run_id", "")
            if run_id and run_id not in seen:
                seen.add(run_id)
                self.runs.append(run_id)
        self.run_combo["values"] = self.runs
        self.pill_runs.configure(text=f"{len(self.runs)} experiments")
        self.pill_records.configure(text=f"{len(self.records)} records")
        if not self.runs:
            self.status_var.set(
                f"No experiment data found ({RECORDS_DIR}). Please run a test on the \"Run Test\" page first.")
            return
        latest = self.runs[0]
        if select_latest or self.current_run not in self.runs:
            self.select_run(latest)
        else:
            self.select_run(self.current_run)
        self.status_var.set(f"Data directory: {cfg.DATA_DIR}; Records directory: {RECORDS_DIR}")

    def select_run(self, run_id, source=None):
        if not run_id:
            return
        self.current_run = run_id
        if self.run_var.get() != run_id:
            self.run_var.set(run_id)
        rec = self._record_by_run.get(run_id)
        if rec:
            mode = rec.get("mode", "singleband")
            if mode == "singleband":
                ber = rec.get("ber", 0)
                ser = rec.get("ser", 0)
            else:
                ber = rec.get("raw_ber_avg", rec.get("eq_ber_avg", rec.get("nn_ber_avg", 0)))
                raw_ser = rec.get("raw_ser")
                ser = float(np.mean(raw_ser)) if isinstance(raw_ser, list) and raw_ser else 0
            self.metrics_var.set(
                f"Mode {mode} | "
                f"Order {rec.get('order', '-')} | "
                f"{rec.get('constellation', '-')} | "
                f"SNR {rec.get('snr_db', 0):.1f} dB | "
                f"BER {ber:.3e} | "
                f"SER {ser:.3e} | "
                f"NN {'Yes' if rec.get('use_nn') else 'No'}")
        else:
            self.metrics_var.set("(No record file for this experiment)")
        self.panel_wave.refresh()
        self.panel_mod.refresh()
        if source != "results":
            self.results_panel.refresh()
        else:
            self.results_panel.plot_panel.refresh()

    def show_quick_plots(self, data: dict):
        """Display TX-only quick plot data in a modal preview window."""
        win = tk.Toplevel(self)
        win.title("快速绘图（仅发射）")
        win.geometry("900x650")
        win.transient(self)
        fig = Figure(figsize=(8, 6), dpi=100)
        canvas = FigureCanvasTkAgg(fig, master=win)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        toolbar = NavigationToolbar2Tk(canvas, win)
        toolbar.update()

        tx = data["tx"]
        sym = data["sym"]
        mode = data["mode"]
        fs = data["fs"]

        ax_tx = fig.add_subplot(2, 2, 1)
        ax_rx = fig.add_subplot(2, 2, 2)
        ax_spec = fig.add_subplot(2, 2, 3)
        ax_const = fig.add_subplot(2, 2, 4)

        sample_len = min(len(tx), 2000)
        t_axis = np.arange(sample_len) / fs

        ax_tx.plot(t_axis * 1e6, tx[:sample_len])
        ax_tx.set_title("发射波形（前 2000 个采样点）")
        ax_tx.set_xlabel("时间 (us)")
        ax_tx.set_ylabel("幅度")

        ax_rx.plot(t_axis * 1e6, tx[:sample_len])
        ax_rx.set_title("接收波形（发射占位）")
        ax_rx.set_xlabel("时间 (us)")
        ax_rx.set_ylabel("幅度")

        nfft = 2 ** int(np.ceil(np.log2(min(len(tx), 8192))))
        f = (np.arange(nfft) - nfft // 2) * fs / nfft / 1e6
        spec = 20 * np.log10(np.abs(np.fft.fftshift(np.fft.fft(tx[:nfft]))) + 1e-12)
        ax_spec.plot(f, spec)
        ax_spec.set_title("发射频谱")
        ax_spec.set_xlabel("频率 (MHz)")
        ax_spec.set_ylabel("幅度 (dB)")

        if sym is not None:
            ax_const.plot(sym.real, sym.imag, "b.", alpha=0.5)
            ax_const.set_title("发射星座图")
            ax_const.set_xlabel("同相 I")
            ax_const.set_ylabel("正交 Q")
            ax_const.grid(True)
        else:
            ax_const.text(0.5, 0.5, "多频带模式\n无星座图",
                          ha="center", va="center", transform=ax_const.transAxes)

        fig.tight_layout()
        canvas.draw()

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
    app = CApGuiApp()
    app.mainloop()


if __name__ == "__main__":
    main()
