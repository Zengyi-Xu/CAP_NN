# -*- coding: utf-8 -*-
"""DMT Communication System Experiment Platform GUI.

Six tabs:
    1. Waveform & Spectrum   —— TX/RX time-domain waveforms and spectra
    2. DMT Modulation        —— bit/power loading, constellation, constellation density
    3. Transmission Results  —— experiment record table, SNR comparison,
                               per-carrier SER/BER, nonlinearity, run history trend
    4. Run Test              —— invoke main.py from the GUI with live log output
    5. Keithley 2400         —— RS-232/USB control of the Keithley 2400 source meter
    6. Grid Scan             —— automated bias vs Vpp parameter sweep with CSV/CodePlot output

Data sources (same auto-save locations as main.py):
    data/records/record_<run_id>.json          parameters and results for each run
                                               (primary source for the drop-down)
    data/codeplot_assets/<run_id>/data/*.npz   plot data for each run (displayed when present)

How to run (from the project directory):
    python dmt_gui.py

Depends only on numpy / matplotlib / tkinter (bundled with Python); no extra install required.
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

# Preferred UI / plot sans-serif font (ships with Raspberry Pi)
UI_FONT = "Liberation Sans"
UI_FONT_FALLBACKS = ["DejaVu Sans", "Liberation Sans"]

# Preferred monospace font for code/log areas (ships with Raspberry Pi)
MONO_FONT = "Liberation Mono Bold"
MONO_FONT_FALLBACKS = ["Liberation Mono", "DejaVu Sans Mono", "Courier"]

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

plt.rcParams["font.sans-serif"] = [FONT_FAMILY] + [f for f in UI_FONT_FALLBACKS if f != FONT_FAMILY]
plt.rcParams["axes.unicode_minus"] = False

PROJECT_ROOT = Path(__file__).resolve().parent
ASSETS_DIR = PROJECT_ROOT / "data" / "codeplot_assets"
RECORDS_DIR = PROJECT_ROOT / "data" / "records"
MAIN_PY = PROJECT_ROOT / "main.py"
CONFIG_PY = PROJECT_ROOT / "config.py"

APP_EMOJI = "📶"  # emoji used for window icon and title; change to your preference


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

def run_data_dir(run_id):
    return ASSETS_DIR / run_id / "data"


def available_plots(run_id, names):
    """Return list of plot names that actually exist for this run (preserving names order)."""
    d = run_data_dir(run_id)
    return [n for n in names if (d / f"{n}.npz").is_file()]


def _resolve_source_run_id(step: str, run_suffix: str) -> str:
    """Resolve the full run_id of the source RX file from step and run_suffix."""
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
    """Write a numpy array into an openpyxl sheet; complex numbers split into real/imag columns."""
    # Handle 0-D scalars (avoid tuple index out of range from arr.shape[0])
    if arr.ndim == 0:
        val = arr.item()
        if np.iscomplexobj(arr):
            ws.cell(row=start_row, column=start_col, value=float(val.real))
            ws.cell(row=start_row, column=start_col + 1, value=float(val.imag))
        else:
            ws.cell(row=start_row, column=start_col,
                    value=float(val) if isinstance(val, (int, float, np.number)) else val)
        return
    # Write header
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
    # Write data
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


def _cleanup_empty_plot_dirs():
    """Remove all empty folders under data/plots (keep directories that contain files)."""
    try:
        if not config.PLOT_DIR.is_dir():
            return
        for d in config.PLOT_DIR.iterdir():
            if d.is_dir() and not any(d.iterdir()):
                d.rmdir()
    except Exception:
        pass


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
# Plotting functions (keep in sync with the CodePlot templates in plot_adapter.py)
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
    """Trend across runs: rate and BER."""
    recs = [r for r in records if r.get("final_rate_gbps") is not None]
    if not recs:
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, "No experiment records yet", ha="center", va="center",
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
    ax2.set_xlabel("Run (chronological)")
    ax2.set_ylabel("BER")
    ax2.grid(True, which="both", ls="--", alpha=0.3)
    ax2.set_xticks(xs)
    ax2.set_xticklabels(labels, rotation=45, fontsize=8)
    fig.tight_layout()


# ═══════════════════════════════════════════════════════════════════════════════
# Figure catalog: name -> (plot function, title)
# ═══════════════════════════════════════════════════════════════════════════════

FIGURES = {
    # Tab 1: Waveform & Spectrum
    "SNRest_QPSK_time":          (build_time,          "QPSK Probe TX Time Waveform"),
    "SNRest_QPSK_spec":          (build_spectrum,      "QPSK Probe TX Spectrum"),
    "SNRest_QPSK_rx_spec":       (build_spectrum,      "QPSK Probe RX Spectrum"),
    "DMT_bitloading_Tx_time":    (build_time,          "DMT Bitloading TX Time Waveform"),
    "DMT_bitloading_Tx_spec":    (build_spectrum,      "DMT Bitloading TX Spectrum"),
    "DMT_bitloading_Rx_spec":    (build_spectrum,      "DMT Bitloading RX Spectrum"),
    # Tab 2: DMT Modulation
    "bit_power_loading":         (build_bit_power_loading, "Bit / Power Loading"),
    "SNRest_QPSK_constellation": (build_constellation, "QPSK Probe Constellation"),
    "bitloading_constellation":  (build_constellation, "DMT Bitloading Constellation"),
    "constellation_by_order":    (build_const_by_order,    "RX Constellation (by modulation order)"),
    "constellation_density":     (build_const_density,     "Constellation Density (by modulation order)"),
    # Tab 3: Transmission Results
    "SNR_QPSK":                  (build_snr,           "QPSK Estimated SNR vs Measured SNR"),
    "SNR_compare":               (build_snr,           "SNR Comparison (Estimated vs Recovered)"),
    "ser_ber_per_carrier":       (build_ser_ber,       "Per-Subcarrier SER / BER"),
    "SNRest_QPSK_nonlinearity":  (build_nonlinearity,  "QPSK TX-RX Amplitude Nonlinearity"),
    "DMT_bitloading_nonlinearity": (build_nonlinearity, "DMT TX-RX Amplitude Nonlinearity"),
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
    # Fonts use point sizes (tk scaling already handles DPI); multiply only pixel sizes by f
    font_base = (FONT_FAMILY, 10)
    font_bold = (FONT_FAMILY, 10, "bold")
    font_tab = (FONT_FAMILY, 11)
    pad_x = int(round(14 * f))
    pad_y = int(round(8 * f))

    style.configure(".", font=font_base, background=COLOR_BG,
                    foreground=COLOR_TEXT)

    # Frame / label
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
    # Pill badges (statistics)
    style.configure("Pill.TLabel", background=COLOR_PRIMARY,
                    foreground="#FFFFFF", font=font_bold,
                    padding=(pad_x, int(round(4 * f))))
    style.configure("Metrics.TLabel", background=COLOR_BG,
                    foreground=COLOR_PRIMARY_HOVER, font=font_bold)

    # Notebook tabs: large padding, white when selected
    style.configure("TNotebook", background=COLOR_BG, borderwidth=0)
    style.configure("TNotebook.Tab", font=font_tab,
                    padding=(pad_x + 6, pad_y),
                    background="#E5EAEF", foreground=COLOR_TEXT)
    style.map("TNotebook.Tab",
              background=[("selected", COLOR_CARD)],
              foreground=[("selected", COLOR_PRIMARY)])

    # Primary button (dark cyan background, white text)
    style.configure("Accent.TButton", font=font_bold,
                    padding=(pad_x, pad_y),
                    background=COLOR_PRIMARY, foreground="#FFFFFF",
                    borderwidth=0, focusthickness=0)
    style.map("Accent.TButton",
              background=[("active", COLOR_PRIMARY_HOVER),
                          ("disabled", "#9FB3BC")],
              foreground=[("disabled", "#E5EAEF")])
    # Normal button
    style.configure("TButton", font=font_base, padding=(pad_x, pad_y),
                    background="#E5EAEF", foreground=COLOR_TEXT,
                    borderwidth=0)
    style.map("TButton", background=[("active", "#D5DDE4")])
    # Danger button
    style.configure("Danger.TButton", font=font_bold,
                    padding=(pad_x, pad_y),
                    background=COLOR_DANGER, foreground="#FFFFFF",
                    borderwidth=0)
    style.map("Danger.TButton",
              background=[("active", "#DC2626"), ("disabled", "#D1A5A5")])

    # Radio / check (indicator scales with DPI so it doesn't stay tiny on high-DPI screens)
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

    # Dropdown
    style.configure("TCombobox", padding=(int(round(8 * f)),
                                          int(round(4 * f))))

    # Table
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

    # Separator / scrollbar
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
        # Any unique ID works; it has nothing to do with the .ico file
        app_id = "DMT.PY.NN.GUI.v1"
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
        # Prefer Linux/Raspberry Pi fonts, then Windows emoji fonts, then fallback
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
        # Generate a multi-size ICO usable by the Windows taskbar
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
        ttk.Label(left, text="Plot List", style="Section.TLabel"
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

        ttk.Button(left, text="Export Current Image",
                   command=self._export_image
                   ).pack(side=tk.BOTTOM, fill=tk.X, padx=8, pady=(0, 4))
        ttk.Button(left, text="Save All Images",
                   command=self._save_all_images
                   ).pack(side=tk.BOTTOM, fill=tk.X, padx=8, pady=(0, 4))
        ttk.Button(left, text="Export Current Plot Data",
                   command=self._export_data
                   ).pack(side=tk.BOTTOM, fill=tk.X, padx=8, pady=(0, 4))
        ttk.Button(left, text="Export All Plot Data",
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
            self.listbox.insert(tk.END, "  Experiment Trend (Rate / BER / SNR)")
            self._items.append(TREND_KEY)
        if self._items:
            self.listbox.selection_set(0)
            self._show(self._items[0])
        else:
            self.fig.clear()
            ax = self.fig.add_subplot(111)
            ax.text(0.5, 0.5, "No data of this category for current experiment",
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
                npz = run_data_dir(self.app.current_run) / f"{name}.npz"
                builder(self.fig, npz, title)
        except Exception as exc:  # show a hint instead of crashing when data is missing/malformed
            self.fig.clear()
            ax = self.fig.add_subplot(111)
            ax.text(0.5, 0.5, f"Plot failed:\n{exc}", ha="center", va="center",
                    transform=ax.transAxes, color="red")
        self.canvas.draw_idle()

    def _safe_savefig(self, path: Path):
        """Save the figure; if bbox_inches='tight' raises IndexError (e.g. bit/power loading
        twin-y plots on some matplotlib versions), fall back to normal save."""
        try:
            self.fig.savefig(path, dpi=config.PLOT_DPI, bbox_inches="tight")
        except (IndexError, ValueError):
            self.fig.savefig(path, dpi=config.PLOT_DPI)

    def _export_image(self):
        """Export the selected plot as PNG; default save location is data/plots/<run_id>/."""
        sel = self.listbox.curselection()
        if not sel:
            messagebox.showinfo("Note", "Please select a plot first")
            return
        name = self._items[sel[0]]
        run_id = self.app.current_run
        if not run_id:
            messagebox.showinfo("Note", "No experiment currently selected")
            return

        plot_dir = config.PLOT_DIR / run_id
        default_name = f"{name}.png"
        path = filedialog.asksaveasfilename(
            title="Export Image",
            initialdir=str(plot_dir),
            initialfile=default_name,
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("PDF", "*.pdf"),
                       ("SVG", "*.svg"), ("All files", "*.*")])
        if not path:
            return
        path = Path(path)
        if path.is_file():
            if not messagebox.askyesno("Confirm Overwrite",
                                       f"File already exists:\n{path}\n\nOverwrite?"):
                return
        try:
            plot_dir.mkdir(parents=True, exist_ok=True)
            self._safe_savefig(path)
            # Save plot parameter metadata
            self._save_plot_metadata(name, path)
            messagebox.showinfo("Export Successful", f"Saved to:\n{path}")
        except Exception as exc:
            messagebox.showerror("Export Failed", str(exc))

    def _save_all_images(self):
        """Save all plots for the current run as PNG into data/plots/<run_id>/."""
        run_id = self.app.current_run
        if not run_id:
            messagebox.showinfo("Note", "No experiment currently selected")
            return
        names = available_plots(run_id, self.plot_names)
        if not names:
            messagebox.showinfo("Note", "No plots available to save for current experiment")
            return

        plot_dir = config.PLOT_DIR / run_id
        existing = sorted([p.name for p in plot_dir.glob("*.png")]) if plot_dir.is_dir() else []
        if existing:
            if not messagebox.askyesno(
                    "Confirm Overwrite",
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
            # Restore the previously displayed plot
            if current_name in self._items:
                self._show(current_name)
            else:
                self._show(self._items[0])
        except Exception as exc:
            messagebox.showerror("Save Failed", str(exc))
            return
        if errors:
            messagebox.showerror("Partial Save Failed", "\n".join(errors))
        else:
            messagebox.showinfo("Save Successful",
                                f"Saved {saved} image(s) to:\n{plot_dir}")

    def _save_plot_metadata(self, name, png_path: Path):
        """Write the current plot's parameters (array names, shapes, data path) as JSON."""
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
            messagebox.showinfo("Note", "Please select a plot first")
            return
        name = self._items[sel[0]]
        run_id = self.app.current_run
        if not run_id:
            messagebox.showinfo("Note", "No experiment currently selected")
            return

        if name == TREND_KEY:
            # Export run-history trend as JSON
            default_name = f"trend_{run_id}.json"
            path = filedialog.asksaveasfilename(
                title="Export Trend Data",
                initialfile=default_name,
                defaultextension=".json",
                filetypes=[("JSON", "*.json"), ("All files", "*.*")])
            if not path:
                return
            try:
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(self.app.records, f, indent=2,
                              ensure_ascii=False, default=str)
                messagebox.showinfo("Export Successful", f"Saved to:\n{path}")
            except Exception as exc:
                messagebox.showerror("Export Failed", str(exc))
            return

        npz = run_data_dir(run_id) / f"{name}.npz"
        if not npz.is_file():
            messagebox.showerror("Export Failed", f"Data file not found:\n{npz}")
            return
        default_name = f"{name}_{run_id}.xlsx"
        path = filedialog.asksaveasfilename(
            title="Export Plot Data",
            initialfile=default_name,
            defaultextension=".xlsx",
            filetypes=[("Excel", "*.xlsx"), ("All files", "*.*")])
        if not path:
            return
        try:
            _npz_to_xlsx(npz, Path(path))
            messagebox.showinfo("Export Successful", f"Saved to:\n{path}")
        except Exception:
            tb = traceback.format_exc()
            print(tb)
            messagebox.showerror("Export Failed", f"Failed to export current plot data:\n{tb}")

    def _export_all_data(self):
        """Export all plot data for the current run to Excel (each array in its own sheet)."""
        run_id = self.app.current_run
        if not run_id:
            messagebox.showinfo("Note", "No experiment currently selected")
            return
        names = available_plots(run_id, self.plot_names)
        if not names:
            messagebox.showinfo("Note", "No plot data available to export for current experiment")
            return

        default_name = f"all_plots_{run_id}.xlsx"
        path = filedialog.asksaveasfilename(
            title="Export All Plot Data",
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
                    ws.cell(row=1, column=1, value=f"Plot: {name}")
                    ws.cell(row=1, column=2, value=f"Array: {key}")
                    ws.cell(row=1, column=3, value=f"Shape: {arr.shape}")
                    _write_array_to_sheet(ws, arr, start_row=3, start_col=1)
            wb.save(path)
            messagebox.showinfo("Export Successful", f"Saved to:\n{path}\n"
                                           f"Exported data for {len(names)} plot(s)")
        except Exception:
            tb = traceback.format_exc()
            print(tb)
            messagebox.showerror("Export Failed", f"Failed to export all plot data:\n{tb}")


class ResultsPanel(ttk.Frame):
    """Tab 3: experiment record table on top, result plots for the selected run below."""

    COLUMNS = ("run_id", "time", "pilot", "vc", "nn", "est_rate",
               "final_rate", "ber", "ser", "snr")
    HEADINGS = {
        "run_id": ("Experiment ID", 170),
        "time": ("Time", 150),
        "pilot": ("Pilot", 90),
        "vc": ("Virtual Channel", 70),
        "nn": ("NN", 50),
        "est_rate": ("Estimated Rate (Gbps)", 110),
        "final_rate": ("Final Rate (Gbps)", 110),
        "ber": ("BER", 100),
        "ser": ("SER", 100),
        "snr": ("Avg SNR (dB)", 100),
    }

    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app

        top = ttk.LabelFrame(self, text=" Transmission Experiment Records (click row to switch experiment) ")
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

        bottom = ttk.LabelFrame(self, text=" Current Experiment Results ")
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
            row = (
                run_id, ts,
                rec.get("pilot_pattern", ""),
                "Yes" if rec.get("use_virtual_channel") else "No",
                "Yes" if rec.get("use_nn") else "No",
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
# config.py parameter panel
# Types: bool=0/1 or True/False toggle; int=integer; num=numeric (scientific/expression);
#        str=string (must be quoted); ichoice=integer dropdown; schoice=string dropdown
# ═══════════════════════════════════════════════════════════════════════════════

PARAM_GROUPS = [
    ("Run Mode", [
        ("OFFLINE_FLAG", "bool", None),
        ("USE_VIRTUAL_CHANNEL", "bool", None),
        ("POSTEQ_FLAG", "ichoice", ["0", "1", "2", "3"]),
        ("RANDOM_SEED", "int", None),
    ]),
    ("DMT Signal Parameters", [
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
    ("Pilot Pattern", [
        ("PILOT_PATTERN", "schoice", ["training_only", "comb", "mesh"]),
        ("PILOT_COMB_START", "int", None),
        ("PILOT_COMB_SPACING", "int", None),
        ("PILOT_MESH_START_FREQ", "int", None),
        ("PILOT_MESH_FREQ_SPACING", "int", None),
        ("PILOT_MESH_START_TIME", "int", None),
        ("PILOT_MESH_TIME_SPACING", "int", None),
    ]),
    ("Pre-Equalization", [
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
    ("Hardware (AWG / Oscilloscope)", [
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
    ("Virtual Channel", [
        ("VIRTUAL_CHANNEL_FC", "num", None),
        ("VIRTUAL_CHANNEL_SNR_DB", "num", None),
        ("VIRTUAL_CHANNEL_NONLINEARITY", "num", None),
        ("VIRTUAL_CHANNEL_DELAY", "int", None),
        ("VIRTUAL_CHANNEL_ATTENUATION", "num", None),
    ]),
    ("Plotting", [
        ("PLOT_SHOW", "bool", None),
        ("PLOT_SAVE", "bool", None),
        ("PLOT_DPI", "int", None),
    ]),
]


def _safe_eval(text):
    """Safe evaluation: try literal_eval first, then simple arithmetic without builtins."""
    try:
        return ast.literal_eval(text)
    except Exception:
        return eval(text, {"__builtins__": {}}, {})


def _find_assignment(text, name):
    """Find `NAME = rhs  # comment` in config.py source and return (rhs, gap, comment).

    gap is the original whitespace between rhs and the inline comment, used to keep
    comment alignment when writing back.
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
    """Editable config.py parameter editor: grouped display, writes only changed parameters back."""

    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        f = app.font_scale
        self.params = []   # each entry: dict(name, type, choices, var, orig, comment, widget)

        # ── Header buttons ───────────────────────────────────────────
        head = tk.Frame(self, bg=COLOR_CARD)
        head.pack(fill=tk.X, padx=8, pady=(8, 4))
        ttk.Button(head, text="💾 Save to config.py", style="Accent.TButton",
                   command=self.save).pack(side=tk.LEFT)
        ttk.Button(head, text="⟳ Reload", command=self.reload
                   ).pack(side=tk.LEFT, padx=8)
        self.info_var = tk.StringVar(
            value="Save parameters before running; only changed parameters are written back to file")
        ttk.Label(head, textvariable=self.info_var, style="DimCard.TLabel"
                  ).pack(side=tk.LEFT, padx=12)

        # ── Scrollable area ──────────────────────────────────────────
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
        # Mouse wheel (enabled only while cursor is over the parameter area)
        self._canvas.bind("<Enter>", lambda _e: self._canvas.bind_all(
            "<MouseWheel>", self._on_wheel))
        self._canvas.bind("<Leave>", lambda _e: self._canvas.unbind_all(
            "<MouseWheel>"))

        self.reload()

    def _on_wheel(self, event):
        self._canvas.yview_scroll(int(-event.delta / 120), "units")

    # ── Read / build ───────────────────────────────────────────────

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
                    tk.Label(row, text="(not found in config.py)",
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
        self.info_var.set(f"Read {len(self.params)} parameters ({CONFIG_PY.name})")

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

    # ── Get values / write back ────────────────────────────────────

    def _literal_for(self, p, errors):
        """Convert the widget's current value to a literal string writable to config.py."""
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
            errors.append(f"{name}: cannot be empty")
            return None
        try:
            val = _safe_eval(raw)
        except Exception:
            errors.append(f"{name}: unable to parse «{raw}»")
            return None
        if ptype == "int":
            if not isinstance(val, (int, float)) or int(val) != val:
                errors.append(f"{name}: integer required, got «{raw}»")
                return None
            return str(int(val))
        if ptype == "num":
            if not isinstance(val, (int, float)):
                errors.append(f"{name}: numeric value required, got «{raw}»")
                return None
            return raw
        if ptype == "str":
            if not isinstance(val, str):
                errors.append(f"{name}: string must be quoted, e.g. \"TCPIP0::…\"")
                return None
            return raw
        return raw

    def get_value(self, name):
        """Return the current widget value of a parameter (as a Python value)."""
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
            messagebox.showerror("config.py not found", str(CONFIG_PY))
            return False
        errors = []
        pending = []
        for p in self.params:
            lit = self._literal_for(p, errors)
            if lit is not None and lit != p["orig"]:
                pending.append((p, lit))
        if errors:
            messagebox.showerror("Parameter Validation Failed", "\n".join(errors))
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
            messagebox.showerror("Write-back Failed",
                                 "The following parameters could not be located in config.py: "
                                 + ", ".join(errors))
            return False
        with CONFIG_PY.open("w", encoding="utf-8", newline="") as fp:
            fp.write(text)
        msg = f"Saved {len(pending)} parameters to config.py" if pending \
            else "No changes, config.py unchanged"
        self.info_var.set(msg)
        if not silent and pending:
            messagebox.showinfo("Save Successful", msg)
        return True


class RunPanel(ttk.Frame):
    """Tab 4: invoke main.py through the GUI to run experiments."""

    MODES = {
        "online":  ("Online (AWG + Oscilloscope)", {"--offline": "0", "--use-awg": "1",
                                                     "--use-virtual-channel": "0"}),
        "offline": ("Offline (read existing RX capture)", {"--offline": "1", "--use-awg": "0",
                                                           "--use-virtual-channel": "0"}),
        "virtual": ("Offline + Virtual Channel (no hardware)", {"--offline": "1", "--use-awg": "0",
                                                                "--use-virtual-channel": "1"}),
    }
    STEPS = {
        "all": "Full pipeline (probe + Bitloading)",
        "step1": "step1 Generate QPSK probe TX",
        "step2": "step2 Receive QPSK and estimate SNR",
        "step3": "step3 Generate Bitloading TX",
        "step4": "step4 Receive Bitloading and demodulate",
    }

    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        self.proc = None
        self._queue = queue.Queue()
        f = app.font_scale

        # ── Parameter card ───────────────────────────────────────────
        opt = ttk.LabelFrame(self, text=" Experiment Parameters ")
        opt.pack(fill=tk.X, padx=2, pady=(2, 8))

        ttk.Label(opt, text="Run mode:", style="Card.TLabel"
                  ).grid(row=0, column=0, sticky=tk.W, padx=12, pady=(10, 4))
        self.mode_var = tk.StringVar(value="virtual")
        col = 1
        for key, (label, _args) in self.MODES.items():
            ttk.Radiobutton(opt, text=label, value=key,
                            variable=self.mode_var
                            ).grid(row=0, column=col, sticky=tk.W,
                                   padx=(4, 16), pady=(10, 4))
            col += 1

        ttk.Label(opt, text="Pipeline step:", style="Card.TLabel"
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
        ttk.Checkbutton(opt, text="Enable NN post-equalizer (--use-nn, training required on first run)",
                        variable=self.use_nn_var
                        ).grid(row=2, column=1, columnspan=3, sticky=tk.W,
                               padx=(4, 16), pady=(4, 10))

        # ── Offline positioning options ──────────────────────────────
        offline_frame = ttk.LabelFrame(opt, text=" Offline Positioning Options (optional) ")
        offline_frame.grid(row=3, column=0, columnspan=5, sticky=tk.EW,
                           padx=12, pady=(0, 8))

        ttk.Label(offline_frame, text="Last 6 digits of STEP2 waveform ID:",
                  style="Card.TLabel"
                  ).grid(row=0, column=0, sticky=tk.W, padx=(8, 4), pady=(8, 4))
        self.qpsk_suffix_var = tk.StringVar(value="")
        ttk.Entry(offline_frame, textvariable=self.qpsk_suffix_var, width=14
                  ).grid(row=0, column=1, sticky=tk.W, padx=4, pady=(8, 4))
        ttk.Label(offline_frame, text="Leave blank to use the latest QPSK RX file",
                  style="DimCard.TLabel"
                  ).grid(row=0, column=2, sticky=tk.W, padx=4, pady=(8, 4))

        ttk.Label(offline_frame, text="Last 6 digits of STEP4 waveform ID:",
                  style="Card.TLabel"
                  ).grid(row=1, column=0, sticky=tk.W, padx=(8, 4), pady=4)
        self.bpl_suffix_var = tk.StringVar(value="")
        ttk.Entry(offline_frame, textvariable=self.bpl_suffix_var, width=14
                  ).grid(row=1, column=1, sticky=tk.W, padx=4, pady=4)
        ttk.Label(offline_frame, text="Leave blank to use the latest Bitloading RX file",
                  style="DimCard.TLabel"
                  ).grid(row=1, column=2, sticky=tk.W, padx=4, pady=4)

        ttk.Label(offline_frame, text="Full run-id / filename:",
                  style="Card.TLabel"
                  ).grid(row=2, column=0, sticky=tk.W, padx=(8, 4), pady=(0, 8))
        self.full_run_id_var = tk.StringVar(value="")
        ttk.Entry(offline_frame, textvariable=self.full_run_id_var, width=36
                  ).grid(row=2, column=1, sticky=tk.W, padx=4, pady=(0, 8))
        ttk.Label(offline_frame, text="Paste full id/filename to auto-detect stage and strip extension",
                  style="DimCard.TLabel"
                  ).grid(row=2, column=2, sticky=tk.W, padx=4, pady=(0, 8))

        self.full_run_id_var.trace_add("write", self._on_full_run_id_change)

        btn_bar = tk.Frame(opt, bg=COLOR_CARD)
        btn_bar.grid(row=4, column=0, columnspan=5, sticky=tk.W,
                     padx=12, pady=(0, 8))
        self.run_btn = ttk.Button(btn_bar, text="▶  Start Test",
                                  style="Accent.TButton",
                                  command=self.start_run)
        self.run_btn.pack(side=tk.LEFT, padx=(0, 8))
        self.stop_btn = ttk.Button(btn_bar, text="■  Stop",
                                   style="Danger.TButton",
                                   command=self.stop_run, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT)
        self.run_status = ttk.Label(btn_bar, text="Ready", style="DimCard.TLabel")
        self.run_status.pack(side=tk.LEFT, padx=16)

        # ── NN progress bar (shown during train/predict, no log spam) ──
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

        # ── Bottom: config parameter panel on left + run log on right ──
        paned = ttk.Panedwindow(self, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
        paned.after_idle(lambda: paned.sashpos(
            0, int(paned.winfo_width() * 0.56)))

        cfg_card = ttk.LabelFrame(paned, text=" config.py Experiment Parameters ")
        paned.add(cfg_card, weight=3)
        self.config_panel = ConfigPanel(cfg_card, app)
        self.config_panel.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        # ── Log card ─────────────────────────────────────────────────
        log_card = ttk.LabelFrame(paned, text=" Run Log ")
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
            "Hint: choose run mode and step, then click Start Test.\n"
            "Online mode requires the M8190A and oscilloscope; use Virtual Channel mode when no hardware is connected.\n"
            "In Offline/Virtual mode you may fill in the last 6 digits of the STEP2 / STEP4 waveform IDs,\n"
            "or paste a full run-id / filename to auto-detect the stage and strip the extension.\n"
            "Data will auto-refresh and switch to the latest experiment after the test completes.\n", "head")

    # ── Run control ───────────────────────────────────────────────

    def build_command(self):
        mode = self.mode_var.get()
        args = self.MODES[mode][1]
        step = self.step_combo.get().split()[0] or "all"
        cmd = [sys.executable, str(MAIN_PY)]
        for k, v in args.items():
            cmd += [k, v]
        cmd += ["--use-nn", "1" if self.use_nn_var.get() else "0",
                "--step", step]
        # Append positioning args in offline/virtual mode
        if mode in ("offline", "virtual"):
            qpsk_suffix = self.qpsk_suffix_var.get().strip()
            bpl_suffix = self.bpl_suffix_var.get().strip()
            if qpsk_suffix and step in ("all", "step2"):
                cmd += ["--run-suffix-qpsk", qpsk_suffix]
            if bpl_suffix and step in ("all", "step4"):
                cmd += ["--run-suffix-bpl", bpl_suffix]
        return cmd

    def _on_full_run_id_change(self, *args):
        """Auto-detect stage from a full run-id or filename, strip extension, and fill the matching box."""
        full = self.full_run_id_var.get().strip()
        if not full:
            return
        # Strip common file extensions
        for ext in (".txt", ".json", ".npz", ".png", ".xlsx", ".mat"):
            if full.lower().endswith(ext):
                full = full[:-len(ext)]
                break
        # Use the last segment as suffix
        suffix = full.split("_")[-1] if "_" in full else full
        if not suffix or len(suffix) < 4:
            return
        # Detect stage prefix
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
            messagebox.showerror("main.py not found", f"Could not find {MAIN_PY}")
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
                    "Detected OFFLINE_FLAG=0",
                    "OFFLINE_FLAG=0 in config.py: step1/step3 will still download "
                    "waveforms to the AWG during the run.\n\nSet OFFLINE_FLAG to 1 before running?\n"
                    "(Choose No to cancel this run)"):
                self.config_panel.set_value("OFFLINE_FLAG", "1")
                self.config_panel.save(silent=True)
            else:
                return
        elif mode == "online" and off_flag not in (0, None):
            if not messagebox.askyesno(
                    "Detected OFFLINE_FLAG=1",
                    "Online mode but OFFLINE_FLAG=1 in config.py: step1/step3 will not "
                    "download waveforms to the AWG, so the oscilloscope may capture no signal.\n\nContinue anyway?"):
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
                        "Confirm Overwrite",
                        f"Images for this run-id already exist in {plot_dir}.\n\n"
                        f"Overwrite?"):
                    return
        cmd += ["--run-id", run_id]

        mode_label = self.MODES[self.mode_var.get()][0]
        if self.mode_var.get() == "online":
            if not messagebox.askyesno(
                    "Confirm Online Mode",
                    "Online mode will control the M8190A and oscilloscope. Confirm hardware is connected?"):
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
            messagebox.showerror("Launch Failed", str(exc))
            self.proc = None
            return
        self._append_log(f"\n$ {' '.join(cmd)}\n", "head")
        self._append_log(f"Mode: {mode_label}\n", "head")
        self.run_btn.configure(state=tk.DISABLED)
        self.stop_btn.configure(state=tk.NORMAL)
        self.run_status.configure(text="Running…")
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
            self.nn_status.configure(text="NN predicting…")
            self.progress_frame.grid()
        elif phase == "done":
            self.progress_var.set(100.0)
            self.nn_status.configure(text="NN equalization done")
            self.progress_frame.grid()

    def _on_done(self, code):
        self.proc = None
        self.run_btn.configure(state=tk.NORMAL)
        self.stop_btn.configure(state=tk.DISABLED)
        self.app.set_running(False)
        if code == 0:
            self.run_status.configure(text="Test completed ✔")
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
        ttk.Label(hdr, text="Keithley 2400 SourceMeter Control",
                  style="Title.TLabel").pack(side=tk.LEFT)
        self.status_lbl = ttk.Label(hdr, text="Disconnected", style="Pill.TLabel")
        self.status_lbl.pack(side=tk.RIGHT)

        # Connection card
        card = make_card(scrollable)
        card.pack(fill=tk.X, padx=16, pady=6)
        inner = tk.Frame(card, bg=COLOR_CARD)
        inner.pack(fill=tk.X, padx=12, pady=12)

        ttk.Label(inner, text="COM Port", style="Section.TLabel").grid(
            row=0, column=0, sticky=tk.W, padx=(0, 8), pady=4)
        self.port_combo = ttk.Combobox(inner, textvariable=self.port_var,
                                       values=[], width=40, state="readonly")
        self.port_combo.grid(row=0, column=1, sticky=tk.W, padx=(0, 8), pady=4)
        ttk.Button(inner, text="⟳ Refresh", command=self._refresh_ports).grid(
            row=0, column=2, padx=(0, 8), pady=4)

        ttk.Label(inner, text="Baud", style="Section.TLabel").grid(
            row=1, column=0, sticky=tk.W, padx=(0, 8), pady=4)
        self.baud_combo = ttk.Combobox(inner, values=[9600, 19200, 38400, 57600, 115200],
                                       width=12, state="readonly")
        self.baud_combo.set(str(config.K2400_BAUDRATE))
        self.baud_combo.grid(row=1, column=1, sticky=tk.W, padx=(0, 8), pady=4)

        self.conn_btn = ttk.Button(inner, text="Connect", command=self._toggle_connect)
        self.conn_btn.grid(row=1, column=2, padx=(0, 8), pady=4)

        # Source settings card
        card2 = make_card(scrollable)
        card2.pack(fill=tk.X, padx=16, pady=6)
        inner2 = tk.Frame(card2, bg=COLOR_CARD)
        inner2.pack(fill=tk.X, padx=12, pady=12)

        ttk.Label(inner2, text="Source Mode", style="Section.TLabel").grid(
            row=0, column=0, sticky=tk.W, padx=(0, 8), pady=4)
        self.mode_combo = ttk.Combobox(inner2, textvariable=self.source_mode_var,
                                       values=self.SOURCE_MODES, state="readonly", width=14)
        self.mode_combo.bind("<<ComboboxSelected>>", self._on_mode_change)
        self.mode_combo.grid(row=0, column=1, sticky=tk.W, padx=(0, 8), pady=4)

        ttk.Label(inner2, text="Level", style="Section.TLabel").grid(
            row=1, column=0, sticky=tk.W, padx=(0, 8), pady=4)
        self.level_entry = ttk.Entry(inner2, textvariable=self.level_var, width=14)
        self.level_entry.grid(row=1, column=1, sticky=tk.W, padx=(0, 8), pady=4)
        self.level_unit_lbl = ttk.Label(inner2, text="V", style="Section.TLabel")
        self.level_unit_lbl.grid(row=1, column=2, sticky=tk.W, pady=4)

        ttk.Label(inner2, text="Compliance", style="Section.TLabel").grid(
            row=2, column=0, sticky=tk.W, padx=(0, 8), pady=4)
        self.comp_entry = ttk.Entry(inner2, textvariable=self.compliance_var, width=14)
        self.comp_entry.grid(row=2, column=1, sticky=tk.W, padx=(0, 8), pady=4)
        self.comp_unit_lbl = ttk.Label(inner2, text="A", style="Section.TLabel")
        self.comp_unit_lbl.grid(row=2, column=2, sticky=tk.W, pady=4)

        ttk.Label(inner2, text="NPLC", style="Section.TLabel").grid(
            row=3, column=0, sticky=tk.W, padx=(0, 8), pady=4)
        self.nplc_entry = ttk.Entry(inner2, textvariable=self.nplc_var, width=14)
        self.nplc_entry.grid(row=3, column=1, sticky=tk.W, padx=(0, 8), pady=4)

        ttk.Checkbutton(inner2, text="Auto range", variable=self.auto_range_var).grid(
            row=4, column=1, sticky=tk.W, padx=(0, 8), pady=4)

        ttk.Button(inner2, text="Apply Settings", command=self._apply_settings).grid(
            row=5, column=1, sticky=tk.W, padx=(0, 8), pady=(12, 0))

        # Output / measure card
        card3 = make_card(scrollable)
        card3.pack(fill=tk.X, padx=16, pady=6)
        inner3 = tk.Frame(card3, bg=COLOR_CARD)
        inner3.pack(fill=tk.X, padx=12, pady=12)

        self.out_btn = ttk.Button(inner3, text="Output ON", command=self._toggle_output)
        self.out_btn.grid(row=0, column=0, padx=(0, 8), pady=4)
        ttk.Button(inner3, text="Measure", command=self._measure).grid(
            row=0, column=1, padx=(0, 8), pady=4)
        ttk.Button(inner3, text="Reset Instrument", command=self._reset).grid(
            row=0, column=2, padx=(0, 8), pady=4)

        self.last_measure_lbl = ttk.Label(
            inner3, text="Last measure: --", style="Section.TLabel")
        self.last_measure_lbl.grid(row=1, column=0, columnspan=4, sticky=tk.W, pady=(12, 0))

        # Log card
        card4 = make_card(scrollable)
        card4.pack(fill=tk.BOTH, expand=True, padx=16, pady=6)
        inner4 = tk.Frame(card4, bg=COLOR_CARD)
        inner4.pack(fill=tk.BOTH, expand=True, padx=12, pady=12)
        ttk.Label(inner4, text="Communication Log", style="Section.TLabel").pack(anchor=tk.W)
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
            messagebox.showwarning("No Port", "Please select a COM port first.")
            return
        baud = int(self.baud_combo.get() or config.K2400_BAUDRATE)
        try:
            self.instrument = Keithley2400(port=port, baudrate=baud,
                                           timeout=config.K2400_TIMEOUT)
            self.instrument.connect()
            self.conn_btn.configure(text="Disconnect")
            self.status_lbl.configure(text=f"Connected ({port})")
            self._log(f"Connected to {port} at {baud} baud")
            self._apply_settings()
        except K2400ConnectionError as exc:
            messagebox.showerror("Connection Failed", str(exc))
            self._log(f"Connection failed: {exc}", "err")
            self.instrument = None
        except Exception as exc:
            messagebox.showerror("Connection Error", str(exc))
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
        self.conn_btn.configure(text="Connect")
        self.status_lbl.configure(text="Disconnected")
        self.output_var.set(False)
        self.out_btn.configure(text="Output ON")
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
            messagebox.showerror("Invalid Value", "Level, compliance and NPLC must be numbers.")
        except K2400ConfigError as exc:
            messagebox.showerror("Configuration Error", str(exc))
        except Exception as exc:
            messagebox.showerror("Apply Settings Failed", str(exc))
            self._log(f"Apply settings failed: {exc}", "err")

    def _toggle_output(self):
        if self.instrument is None or not self.instrument.connected:
            self._log("Not connected.")
            return
        try:
            if self.output_var.get():
                self.instrument.output_off()
                self.output_var.set(False)
                self.out_btn.configure(text="Output ON")
                self._log("Output OFF")
            else:
                self.instrument.output_on()
                self.output_var.set(True)
                self.out_btn.configure(text="Output OFF")
                self._log("Output ON")
        except Exception as exc:
            messagebox.showerror("Output Control Failed", str(exc))
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
            messagebox.showerror("Measurement Failed", str(exc))
            self._log(f"Measurement failed: {exc}", "err")
        except Exception as exc:
            messagebox.showerror("Measurement Error", str(exc))
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
            messagebox.showerror("Reset Failed", str(exc))
            self._log(f"Reset failed: {exc}", "err")

    def on_close(self):
        """Call when the application exits to safely turn off output."""
        self._disconnect()


# ═══════════════════════════════════════════════════════════════════════════════
# Tab 6: Grid Scan (bias vs Vpp)
# ═══════════════════════════════════════════════════════════════════════════════

class GridScanPanel(ttk.Frame):
    """GUI panel for automated 2-D parameter sweeps (Keithley bias vs AWG Vpp)."""

    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        self.scanner: Optional[GridScanner] = None
        self.scan_thread: Optional[threading.Thread] = None
        self.stop_requested = False

        # Configuration variables
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

    # --- UI construction ----------------------------------------------------
    def _build_ui(self):
        paned = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=12, pady=12)

        # Left: configuration
        left = tk.Frame(paned, bg=COLOR_BG)
        paned.add(left, weight=1)

        hdr = tk.Frame(left, bg=COLOR_BG)
        hdr.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(hdr, text="Grid Scan Configuration", style="Title.TLabel").pack(anchor=tk.W)

        card = make_card(left)
        card.pack(fill=tk.X, pady=6)
        inner = tk.Frame(card, bg=COLOR_CARD)
        inner.pack(fill=tk.X, padx=12, pady=12)

        # Parameter 1 (Keithley)
        ttk.Label(inner, text="Parameter 1 (Keithley)", style="Section.TLabel").grid(
            row=0, column=0, columnspan=4, sticky=tk.W, pady=(0, 6))

        ttk.Label(inner, text="Name").grid(row=1, column=0, sticky=tk.W, padx=(0, 4))
        ttk.Entry(inner, textvariable=self.p1_name_var, width=14).grid(
            row=1, column=1, sticky=tk.W, padx=(0, 8), pady=2)
        ttk.Label(inner, text="Mode").grid(row=1, column=2, sticky=tk.W, padx=(0, 4))
        ttk.Combobox(inner, textvariable=self.p1_mode_var,
                     values=["voltage", "current"], state="readonly", width=10).grid(
            row=1, column=3, sticky=tk.W, pady=2)

        ttk.Label(inner, text="Start").grid(row=2, column=0, sticky=tk.W, padx=(0, 4))
        ttk.Entry(inner, textvariable=self.p1_start_var, width=10).grid(
            row=2, column=1, sticky=tk.W, padx=(0, 8), pady=2)
        ttk.Label(inner, text="Stop").grid(row=2, column=2, sticky=tk.W, padx=(0, 4))
        ttk.Entry(inner, textvariable=self.p1_stop_var, width=10).grid(
            row=2, column=3, sticky=tk.W, pady=2)

        ttk.Label(inner, text="Step").grid(row=3, column=0, sticky=tk.W, padx=(0, 4))
        ttk.Entry(inner, textvariable=self.p1_step_var, width=10).grid(
            row=3, column=1, sticky=tk.W, padx=(0, 8), pady=2)

        # Parameter 2 (AWG Vpp)
        ttk.Label(inner, text="Parameter 2 (AWG Vpp)", style="Section.TLabel").grid(
            row=4, column=0, columnspan=4, sticky=tk.W, pady=(12, 6))

        ttk.Label(inner, text="Start").grid(row=5, column=0, sticky=tk.W, padx=(0, 4))
        ttk.Entry(inner, textvariable=self.vpp_start_var, width=10).grid(
            row=5, column=1, sticky=tk.W, padx=(0, 8), pady=2)
        ttk.Label(inner, text="Stop").grid(row=5, column=2, sticky=tk.W, padx=(0, 4))
        ttk.Entry(inner, textvariable=self.vpp_stop_var, width=10).grid(
            row=5, column=3, sticky=tk.W, pady=2)

        ttk.Label(inner, text="Step").grid(row=6, column=0, sticky=tk.W, padx=(0, 4))
        ttk.Entry(inner, textvariable=self.vpp_step_var, width=10).grid(
            row=6, column=1, sticky=tk.W, padx=(0, 8), pady=2)

        # Pipeline options
        ttk.Label(inner, text="Pipeline", style="Section.TLabel").grid(
            row=7, column=0, columnspan=4, sticky=tk.W, pady=(12, 6))

        ttk.Label(inner, text="Run mode").grid(row=8, column=0, sticky=tk.W, padx=(0, 4))
        ttk.Combobox(inner, textvariable=self.run_mode_var,
                     values=["step1-4", "step1-2"], state="readonly", width=12).grid(
            row=8, column=1, sticky=tk.W, padx=(0, 8), pady=2)
        ttk.Label(inner, text="Repeats").grid(row=8, column=2, sticky=tk.W, padx=(0, 4))
        ttk.Entry(inner, textvariable=self.repeats_var, width=10).grid(
            row=8, column=3, sticky=tk.W, pady=2)

        ttk.Checkbutton(inner, text="Offline mode", variable=self.offline_var).grid(
            row=9, column=0, columnspan=2, sticky=tk.W, pady=2)
        ttk.Checkbutton(inner, text="Virtual channel", variable=self.use_virtual_channel_var).grid(
            row=9, column=2, columnspan=2, sticky=tk.W, pady=2)
        ttk.Checkbutton(inner, text="Use NN post-equalizer", variable=self.use_nn_var).grid(
            row=10, column=0, columnspan=2, sticky=tk.W, pady=2)

        # Keithley connection
        ttk.Label(inner, text="Keithley Connection", style="Section.TLabel").grid(
            row=11, column=0, columnspan=4, sticky=tk.W, pady=(12, 6))

        ttk.Label(inner, text="COM Port").grid(row=12, column=0, sticky=tk.W, padx=(0, 4))
        self.port_combo = ttk.Combobox(inner, textvariable=self.port_var,
                                       values=[], width=18, state="readonly")
        self.port_combo.grid(row=12, column=1, sticky=tk.W, padx=(0, 8), pady=2)
        ttk.Button(inner, text="⟳ Refresh", command=self._refresh_ports).grid(
            row=12, column=2, columnspan=2, sticky=tk.W, pady=2)

        ttk.Label(inner, text="Baud").grid(row=13, column=0, sticky=tk.W, padx=(0, 4))
        ttk.Entry(inner, textvariable=self.baud_var, width=10).grid(
            row=13, column=1, sticky=tk.W, padx=(0, 8), pady=2)
        ttk.Label(inner, text="Compliance").grid(row=13, column=2, sticky=tk.W, padx=(0, 4))
        ttk.Entry(inner, textvariable=self.compliance_var, width=10).grid(
            row=13, column=3, sticky=tk.W, pady=2)

        ttk.Label(inner, text="NPLC").grid(row=14, column=0, sticky=tk.W, padx=(0, 4))
        ttk.Entry(inner, textvariable=self.nplc_var, width=10).grid(
            row=14, column=1, sticky=tk.W, padx=(0, 8), pady=2)

        # Run controls
        ctrl = tk.Frame(left, bg=COLOR_CARD)
        ctrl.pack(fill=tk.X, pady=(12, 0), padx=2)
        self.run_btn = ttk.Button(ctrl, text="▶ Start Grid Scan", command=self._start_scan)
        self.run_btn.pack(side=tk.LEFT, padx=(8, 8), pady=8)
        self.stop_btn = ttk.Button(ctrl, text="⏹ Stop", command=self._stop_scan, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=(0, 8), pady=8)

        self.progress_var = tk.DoubleVar(value=0.0)
        self.progress_lbl = ttk.Label(left, text="Ready")
        self.progress_lbl.pack(anchor=tk.W, pady=(8, 0))
        self.progress = ttk.Progressbar(left, variable=self.progress_var, maximum=1.0)
        self.progress.pack(fill=tk.X, pady=(4, 0))

        # Right: results browser
        right = tk.Frame(paned, bg=COLOR_BG)
        paned.add(right, weight=2)

        ttk.Label(right, text="Past Grid Scans", style="Title.TLabel").pack(anchor=tk.W, pady=(0, 8))

        # Scan selector
        sel = tk.Frame(right, bg=COLOR_BG)
        sel.pack(fill=tk.X, pady=(0, 6))
        self.scan_var = tk.StringVar()
        self.scan_combo = ttk.Combobox(sel, textvariable=self.scan_var,
                                       values=[], state="readonly", width=40)
        self.scan_combo.pack(side=tk.LEFT, padx=(0, 8))
        self.scan_combo.bind("<<ComboboxSelected>>", self._on_scan_selected)
        ttk.Button(sel, text="⟳ Refresh", command=self._refresh_scan_list).pack(side=tk.LEFT)

        # Summary table
        self.tree = ttk.Treeview(right, show="headings", height=8)
        self.tree.pack(fill=tk.X, pady=(0, 8))
        self.tree["columns"] = ("point", "param1", "vpp", "ber", "ser", "snr_db", "rate_gbps")
        for col in self.tree["columns"]:
            self.tree.heading(col, text=col.replace("_", " ").title())
            self.tree.column(col, width=80, anchor=tk.CENTER)
        self.tree.column("param1", width=100)
        self.tree.column("vpp", width=80)

        # Contour plot
        plot_card = make_card(right)
        plot_card.pack(fill=tk.BOTH, expand=True, pady=(6, 0))
        plot_inner = tk.Frame(plot_card, bg=COLOR_CARD)
        plot_inner.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        self.plot_metric_var = tk.StringVar(value="snr_db")
        ttk.Label(plot_inner, text="Contour metric").pack(anchor=tk.W)
        metric_combo = ttk.Combobox(plot_inner, textvariable=self.plot_metric_var,
                                    values=["snr_db", "ber", "ser"], state="readonly", width=12)
        metric_combo.pack(anchor=tk.W, pady=(0, 6))
        metric_combo.bind("<<ComboboxSelected>>", self._on_metric_changed)

        self.fig = Figure(figsize=(6, 4), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, plot_inner)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.ax.set_title("Select a grid scan to view contour plot")
        self.canvas.draw()

    # --- Helpers ------------------------------------------------------------
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
            self.ax.set_title("No data")
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
            self.ax.set_title("Invalid data")
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
        self.ax.set_ylabel("AWG Vpp (V)")
        self.ax.set_title(f"{metric.upper()} Grid Scan ({scan_id})")
        self.fig.tight_layout()
        self.canvas.draw()

    # --- Scan control -------------------------------------------------------
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
            messagebox.showerror("Invalid Configuration", str(exc))
            return

        self.stop_requested = False
        self.run_btn.configure(state=tk.DISABLED)
        self.stop_btn.configure(state=tk.NORMAL)
        self.progress_var.set(0.0)
        self.progress_lbl.configure(text=f"Scan {cfg.scan_id} starting...")

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
        self.progress_lbl.configure(text="Stop requested")

    def _scan_done(self, csv_path: Path, scan_id: str):
        self.run_btn.configure(state=tk.NORMAL)
        self.stop_btn.configure(state=tk.DISABLED)
        self.progress_lbl.configure(text=f"Done: {csv_path}")
        self.progress_var.set(1.0)
        self._refresh_scan_list()
        self.scan_var.set(scan_id)
        self._on_scan_selected()
        messagebox.showinfo("Grid Scan Complete", f"Summary saved to:\n{csv_path}")

    def _scan_error(self, exc: Exception):
        self.run_btn.configure(state=tk.NORMAL)
        self.stop_btn.configure(state=tk.DISABLED)
        self.progress_lbl.configure(text=f"Error: {exc}")
        messagebox.showerror("Grid Scan Failed", str(exc))


# ═══════════════════════════════════════════════════════════════════════════════
# Main window
# ═══════════════════════════════════════════════════════════════════════════════

class DmtGuiApp(tk.Tk):
    def __init__(self):
        super().__init__()

        # Windows taskbar icon needs an explicit AppUserModelID
        _set_windows_taskbar_icon()

        # DPI scaling
        dpi = self.winfo_fpixels("1i")
        self.font_scale = max(dpi / 96.0, 1.0)
        self.tk.call("tk", "scaling", dpi / 72.0)

        self.title(f"{APP_EMOJI} DMT Communication System Experiment Platform")
        self._icon, self._icon_ico = _create_emoji_icon(APP_EMOJI)
        # Prefer ICO on Windows to avoid the taskbar feather; other platforms use photo
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

        # Clean up historical empty plot directories
        _cleanup_empty_plot_dirs()

        self.runs = []
        self.records = []
        self.current_run = None
        self._record_by_run = {}
        self._running = False

        # ── Top header bar ───────────────────────────────────────────
        header = tk.Frame(self, bg=COLOR_BG)
        header.pack(fill=tk.X, padx=16, pady=(14, 6))
        ttk.Label(header, text=f"{APP_EMOJI} DMT Communication System Experiment Platform",
                  style="Title.TLabel").pack(side=tk.LEFT)
        self.pill_runs = ttk.Label(header, style="Pill.TLabel")
        self.pill_runs.pack(side=tk.RIGHT, padx=(8, 0))
        self.pill_records = ttk.Label(header, style="Pill.TLabel")
        self.pill_records.pack(side=tk.RIGHT)

        # ── Experiment selector bar ──────────────────────────────────
        sel = make_card(self)
        sel.pack(fill=tk.X, padx=16, pady=(4, 8))
        inner = tk.Frame(sel, bg=COLOR_CARD)
        inner.pack(fill=tk.X, padx=12, pady=10)
        f = self.font_scale
        ttk.Label(inner, text="Experiment ID (run_id)", style="Section.TLabel"
                  ).pack(side=tk.LEFT)
        self.run_var = tk.StringVar()
        self.run_combo = ttk.Combobox(
            inner, textvariable=self.run_var, state="readonly",
            width=30, font=(FONT_MONO, 10))
        self.run_combo.pack(side=tk.LEFT, padx=(8, 8))
        self.run_combo.bind("<<ComboboxSelected>>",
                            lambda _e: self.select_run(self.run_var.get()))
        ttk.Button(inner, text="⟳ Refresh Data", command=self.reload_data
                   ).pack(side=tk.LEFT)
        self.metrics_var = tk.StringVar(value="")
        ttk.Label(inner, textvariable=self.metrics_var, style="Metrics.TLabel"
                  ).pack(side=tk.LEFT, padx=20)

        # ── Six tabs ────────────────────────────────────────────────
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=16, pady=(0, 8))

        tab1 = ttk.Frame(self.notebook)
        self.notebook.add(tab1, text="  📈 Waveform & Spectrum  ")
        self.panel_wave = PlotPanel(tab1, self, TAB_WAVEFORM)
        self.panel_wave.pack(fill=tk.BOTH, expand=True, padx=2, pady=6)

        tab2 = ttk.Frame(self.notebook)
        self.notebook.add(tab2, text="  🎛 DMT Modulation  ")
        self.panel_mod = PlotPanel(tab2, self, TAB_MODULATION)
        self.panel_mod.pack(fill=tk.BOTH, expand=True, padx=2, pady=6)

        tab3 = ResultsPanel(self.notebook, self)
        self.notebook.add(tab3, text="  📊 Transmission Results  ")
        self.results_panel = tab3

        tab4 = RunPanel(self.notebook, self)
        self.notebook.add(tab4, text="  ▶ Run Test  ")
        self.panel_run = tab4

        tab5 = Keithley2400Panel(self.notebook, self)
        self.notebook.add(tab5, text="  ⚡ Keithley 2400  ")
        self.panel_k2400 = tab5

        tab6 = GridScanPanel(self.notebook, self)
        self.notebook.add(tab6, text="  🔲 Grid Scan  ")
        self.panel_grid = tab6

        # Default to the "Run Test" page (4th tab, index 3)
        self.notebook.select(3)
        
        # ── Status bar ───────────────────────────────────────────────
        self.status_var = tk.StringVar()
        status = tk.Label(self, textvariable=self.status_var, anchor=tk.W,
                          bg=COLOR_BG, fg=COLOR_TEXT_DIM, bd=0,
                          font=(FONT_FAMILY, 9))
        status.pack(fill=tk.X, side=tk.BOTTOM, padx=16, pady=(0, 8))

        # Ensure the Keithley output is turned off when the GUI closes
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        self.reload_data()

    # ── Data ───────────────────────────────────────────────────────

    def reload_data(self, select_latest=False):
        self.records = list_records()
        self._record_by_run = {r.get("run_id"): r for r in self.records}
        # De-duplicate from records and show newest first so single-step runs are visible too
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
        self.status_var.set(f"Data directory: {ASSETS_DIR}; Records directory: {RECORDS_DIR}")

    def select_run(self, run_id, source=None):
        if not run_id:
            return
        self.current_run = run_id
        if self.run_var.get() != run_id:
            self.run_var.set(run_id)
        rec = self._record_by_run.get(run_id)
        if rec:
            self.metrics_var.set(
                f"Rate {rec.get('final_rate_gbps', 0):.2f} Gbps | "
                f"BER {rec.get('final_ber', 0):.3e} | "
                f"SER {rec.get('final_ser', 0):.3e} | "
                f"Avg SNR {rec.get('mean_recovered_snr_db', 0):.2f} dB | "
                f"Pilot {rec.get('pilot_pattern', '-')}")
        else:
            self.metrics_var.set("(No record file for this experiment)")
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
