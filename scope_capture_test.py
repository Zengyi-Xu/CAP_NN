#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Standalone oscilloscope capture test.

This script mirrors MATLAB oscrunQPSK.m / oscrunDMT.m to verify that the
oscilloscope can capture a real waveform correctly. It supports both USB-B
(USBTMC) and TCPIP (port 5025) connections.

Usage:
    python scope_capture_test.py
    python scope_capture_test.py --resource "TCPIP0::192.168.1.10::5025::SOCKET" --channel CHAN2
    python scope_capture_test.py --usb --channel CHAN2 --out rxdata/scope_test.txt
"""

import argparse
import sys
from pathlib import Path

import numpy as np

import config
from oscilloscope import KeysightScopeUSB
from utils import save_txt, resample_signal


def plot_and_analyze(data: np.ndarray,
                     fs_osc: float,
                     fs_awg: float,
                     title: str = "Scope Capture") -> None:
    """Plot time domain, spectrum, and print basic statistics."""
    import matplotlib.pyplot as plt

    n = len(data)
    t = np.arange(n) / fs_osc
    freqs = np.fft.fftshift(np.fft.fftfreq(n, d=1.0 / fs_osc))
    # MATLAB: 10*log10(abs(fft(...)))
    spec = 10 * np.log10(np.abs(np.fft.fftshift(np.fft.fft(data))) + 1e-12)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    ax = axes[0]
    ax.plot(t * 1e6, data, "b-")
    ax.set_title(f"{title} - Time Domain")
    ax.set_xlabel("Time (us)")
    ax.set_ylabel("Voltage (V)")
    ax.grid(True)

    ax = axes[1]
    ax.plot(freqs / 1e9, spec, "b-")
    ax.set_title(f"{title} - Spectrum")
    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel("Magnitude (dB)")
    ax.grid(True)

    fig.tight_layout()
    plt.show()

    print("\n========== Waveform Statistics ==========")
    print(f"  samples       = {n}")
    print(f"  duration      = {n/fs_osc*1e6:.2f} us")
    print(f"  min           = {np.min(data):.6f} V")
    print(f"  max           = {np.max(data):.6f} V")
    print(f"  mean          = {np.mean(data):.6f} V")
    print(f"  std           = {np.std(data):.6f} V")
    print(f"  peak-to-peak  = {np.ptp(data):.6f} V")
    print(f"  RMS           = {np.sqrt(np.mean(data**2)):.6f} V")
    print(f"  spectrum max  = {np.max(spec):.2f} dB")
    print(f"  spectrum min  = {np.min(spec):.2f} dB")

    # Simple energy check: if std is close to 0 or spectrum max is very low,
    # the captured waveform likely has no signal.
    if np.std(data) < 1e-6:
        print("\n[WARNING] Waveform std is extremely low. No signal detected!")
    else:
        print("\n[INFO] Waveform has non-zero energy. Signal likely present.")


def main():
    parser = argparse.ArgumentParser(description="Standalone scope capture test")
    parser.add_argument("--resource", "-r", type=str, default=None,
                        help="VISA resource string or 'TCPIP0::ip::5025::SOCKET'")
    parser.add_argument("--usb", action="store_true",
                        help="Use first available USB instrument (overrides --resource)")
    parser.add_argument("--channel", "-c", type=str, default="CHAN2",
                        help="Scope channel (MATLAB oscrun uses CHAN2)")
    parser.add_argument("--sample-rate", type=float, default=config.OSC_SAMPLE_RATE,
                        help="Scope sample rate in Hz")
    parser.add_argument("--timebase", type=float, default=80e-6,
                        help="Timebase scale in s/div")
    parser.add_argument("--resample", action="store_true", default=True,
                        help="Resample captured waveform to AWG sample rate")
    parser.add_argument("--out", "-o", type=str, default=None,
                        help="Output file path (default: rxdata/scope_test_{count}.txt)")
    parser.add_argument("--no-plot", action="store_true",
                        help="Do not display plots")
    args = parser.parse_args()

    # Determine resource
    resource = args.resource
    if args.usb:
        resource = None  # auto-select first USB
    elif resource is None:
        # Default: use config, but if config is still USB and user wants TCPIP,
        # they should pass --resource explicitly
        resource = config.OSC_VISA_ADDR

    print(f"[TEST] Resource = {resource or 'auto-USB'}")
    print(f"[TEST] Channel  = {args.channel}")
    print(f"[TEST] fs_scope = {args.sample_rate/1e9:.2f} GSa/s")
    print(f"[TEST] timebase = {args.timebase*1e6:.1f} us/div")

    try:
        with KeysightScopeUSB(resource=resource) as scope:
            print("\nAvailable resources:")
            for r in scope.list_resources():
                print(f"  {r}")

            data, preamble = scope.capture(
                channel=args.channel,
                sample_rate=args.sample_rate,
                timebase_scale=args.timebase,
                resample_to_awg=False  # keep raw scope samples for diagnosis
            )
    except Exception as e:
        print(f"\n[ERROR] Scope capture failed: {e}")
        raise

    print("\n========== Scope Preamble ==========")
    for k, v in preamble.items():
        print(f"  {k:15s} = {v}")

    # Optional resample
    if args.resample:
        data_resampled = resample_signal(data,
                                         fs_target=config.AWG_SAMPLE_RATE,
                                         fs_source=args.sample_rate)
        print(f"\n[INFO] Resampled to {config.AWG_SAMPLE_RATE/1e9:.2f} GSa/s, "
              f"length {len(data_resampled)}")
    else:
        data_resampled = data

    # Save
    if args.out is None:
        from utils import load_txt
        count = 0
        if config.COUNT_FILE.exists():
            try:
                count = int(load_txt(config.COUNT_FILE))
            except Exception:
                count = 0
        out_path = config.RXDATA_DIR / f"scope_test_{count}.txt"
        save_txt(config.COUNT_FILE, np.array([count + 1]), fmt="%d")
    else:
        out_path = Path(args.out)
    save_txt(out_path, data_resampled)
    print(f"\n[INFO] Saved waveform to {out_path}")

    # Plot raw scope data
    if not args.no_plot:
        plot_and_analyze(data, args.sample_rate, config.AWG_SAMPLE_RATE,
                         title=f"Scope {args.channel} @ {args.sample_rate/1e9:.2f} GSa/s")

        if args.resample:
            plot_and_analyze(data_resampled, config.AWG_SAMPLE_RATE, config.AWG_SAMPLE_RATE,
                             title=f"Resampled to AWG rate")


if __name__ == "__main__":
    main()
