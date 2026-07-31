#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
Standalone M8190A arbitrary waveform download test.

This script is a line-by-line Python equivalent of the MATLAB code in
    D:\BaiduSyncdisk\SyncWorkplace\Individual\Code\20260429_900G_testing\AWGM8190A_Auto
especially:
    - AWGM8190A_Auto.m      (top-level auto download)
    - AWG_transmit.m        (basic configuration & run)
    - download.m            (argument parsing & channel mapping)
    - download_M8190A.m     (M8190A-specific download)
    - xfprintf.m / xquery.m / xbinblockwrite.m  (raw SCPI over TCP)

It uses Python's standard socket library to talk to the AWG firmware raw TCP
port (default 5025), exactly as the MATLAB reference does with tcpclient().

Usage:
    python awg_download.py
    python awg_download.py --host 192.168.1.10 --port 5025 -f txdata/SNRest_QPSK.txt
    python awg_download.py --route AC --amplitude 0.3 --ch1 --ch2
"""

import argparse
import socket
import struct
import sys
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Default parameters (matching the original MATLAB reference)
# ---------------------------------------------------------------------------
DEFAULT_HOST = "localhost"          # MATLAB: device_name = "DESKTOP-CION5EQ"
DEFAULT_PORT = 5025                 # MATLAB RX_CHANNEL_SENSING_CONFIG.txt(9)
DEFAULT_SAMPLE_RATE = 8e9           # Hz
DEFAULT_AMPLITUDE = 0.5             # Vpp
DEFAULT_WAVEFORM = r"C:\Users\Lab VLC\Desktop\PSCAP\txdata\datatx_pre_4QAM.txt"
DEFAULT_CONFIG_TXT = "RX_CHANNEL_SENSING_CONFIG.txt"

# M8190A_12bit mode constants (from loadArbConfig.m)
SEGMENT_GRANULARITY = 64
MIN_SEGMENT_SIZE = 5 * SEGMENT_GRANULARITY
MAX_SEGMENT_SIZE = 3 * 512 * 1024 * 1024
MAX_SEGMENT_NUMBER = 512 * 1024

# Output routing options (see M8190A user manual / gen_arb_M8190A.m)
#   "DC"  -> :OUTP1:ROUT DC  + :DC1:VOLT:AMPL  (DC-coupled amplified output)
#   "AC"  -> :OUTP1:ROUT AC  + :AC1:VOLT:AMPL  (AC-coupled amplified output)
#   "DAC" -> :OUTP1:ROUT DAC + :VOLT1:AMPL     (direct DAC output, smallest amp)
OUTPUT_ROUTES = ("DC", "AC", "DAC")


# =============================================================================
# Low-level SCPI helpers (raw socket, MATLAB xfprintf/xquery equivalents)
# =============================================================================

class ScpiSocket:
    """Raw TCP socket wrapper that behaves like MATLAB tcpclient/visadev.SOCKET."""

    def __init__(self, host: str, port: int, timeout: float = 30.0):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.sock: socket.socket | None = None
        self._buf = b""

    def connect(self):
        print(f"[SCPI] Connecting to {self.host}:{self.port} ...")
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.settimeout(self.timeout)
        self.sock.connect((self.host, self.port))
        print(f"[SCPI] Connected.")
        return self

    def close(self):
        if self.sock is not None:
            try:
                self.sock.close()
            except Exception as e:
                print(f"[SCPI] Warning during close: {e}")
            self.sock = None

    def __enter__(self):
        return self.connect()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def write_line(self, data: bytes | str):
        """Send bytes/string terminated by LF (MATLAB writeline)."""
        if self.sock is None:
            raise RuntimeError("SCPI socket not connected")
        if isinstance(data, str):
            data = data.encode("ascii")
        self.sock.sendall(data + b"\n")

    def read_line(self) -> str:
        """Read one LF-terminated line (MATLAB readline)."""
        if self.sock is None:
            raise RuntimeError("SCPI socket not connected")
        while b"\n" not in self._buf:
            chunk = self.sock.recv(65536)
            if not chunk:
                break
            self._buf += chunk
        if b"\n" not in self._buf:
            raise TimeoutError("SCPI read_line timeout")
        line, self._buf = self._buf.split(b"\n", 1)
        return line.decode("ascii", errors="replace").strip()


def xfprintf(f: ScpiSocket, cmd: str, ignore_error: bool = False) -> int:
    """Send a SCPI command and check the error queue (MATLAB xfprintf)."""
    f.write_line(cmd)
    for _ in range(50):
        try:
            err = xquery(f, ":SYST:ERR?")
        except Exception:
            err = ""
        if not err:
            print(f"[SCPI] Warning: no response to :SYST:ERR? after {cmd}")
            return -1
        code = 0
        try:
            code = int(err.split(",")[0])
        except ValueError:
            pass
        if code == 0:
            return 0
        if not ignore_error:
            print(f"[SCPI] Error from AWG after '{cmd}': {err}")
            return -1
    return -1


def xquery(f: ScpiSocket, cmd: str) -> str:
    """Send a SCPI query and return the response string (MATLAB xquery)."""
    f.write_line(cmd)
    return f.read_line()


def xbinblockwrite(f: ScpiSocket, data: np.ndarray, fmt: str, cmd: str):
    """Send IEEE 488.2 binary block data (MATLAB xbinblockwrite for tcpclient).

    Args:
        f:    SCPI socket.
        data: 1-D numpy array; must match `fmt`.
        fmt:  one of 'int8','uint8','int16','uint16','int32','uint32'.
        cmd:  SCPI header, e.g. ":TRACe1:DATA 1,0,".
    """
    dtype_map = {
        "int8": np.int8,
        "uint8": np.uint8,
        "int16": np.int16,
        "uint16": np.uint16,
        "int32": np.int32,
        "uint32": np.uint32,
    }
    if fmt not in dtype_map:
        raise ValueError(f"xbinblockwrite: unsupported format {fmt}")
    data = np.asarray(data, dtype=dtype_map[fmt]).ravel()
    data_bytes = data.tobytes()  # little-endian on x86 / M8190A firmware
    n = len(data_bytes)
    n_str = str(n)
    header = f"{cmd}#{len(n_str)}{n_str}"
    payload = header.encode("ascii") + data_bytes + b"\n"
    f.sock.sendall(payload)


# =============================================================================
# ArbConfig and waveform loading (MATLAB makeArbConfig / readFile / loadfile)
# =============================================================================

def make_arb_config(visa_addr: str, amplitude: float, sample_rate: float, port: int):
    """Build an arbConfig dictionary matching makeArbConfig.m + loadArbConfig.m."""
    arb = {
        "model": "M8190A_12bit",
        "connectionType": "tcpip",
        "visaAddr": visa_addr,
        "ip_address": "192.168.0.100",
        "port": port,
        "LOIPAddr": "xxx.xxx.xxx.xxx",
        "SD1ModuleIndex": 0,
        "M8070ModuleID": "M2",
        "defaultFc": 0,
        "tooltips": 1,
        "DACRange": 1,
        "amplScale": 2.83,
        "amplScaleMode": "Leave Unchanged",
        "filterSettings": "None",
        "userSamplerate": sample_rate,
        "triggerMode": "Continuous",
        "amplitude": amplitude,
        "offset": 0,
        "outputType": "Single Ended",
        "clockSource": "ExtRef",       # 'Unchanged','IntRef','AxieRef','ExtRef','ExtClk'
        "clockFreq": 10_000_000,       # 10 MHz external reference
        "peaking": None,
        "interleaving": 0,
        "sampleMarker": "Sample rate / 4",
        "useM8192A": 0,
        "visaAddrM8192A": "TCPIP0::localhost::hislip0::INSTR",
        "timeout": 30,
    }
    # loadArbConfig.m: M8190A_12bit parameters
    arb["numChannels"] = 2
    arb["channelMask"] = np.ones(arb["numChannels"], dtype=int)
    arb["fixedSampleRate"] = 0
    arb["defaultSampleRate"] = 12e9
    arb["maximumSampleRate"] = 12e9
    arb["minimumSampleRate"] = 125e6
    arb["minimumSegmentSize"] = MIN_SEGMENT_SIZE
    arb["maximumSegmentSize"] = MAX_SEGMENT_SIZE
    arb["segmentGranularity"] = SEGMENT_GRANULARITY
    arb["maxSegmentNumber"] = MAX_SEGMENT_NUMBER
    arb["maximumModules"] = 4
    # user-defined default sample rate overrides everything
    arb["defaultSampleRate"] = sample_rate
    return arb


def read_file(filename: str | Path, sample_rate: float = DEFAULT_SAMPLE_RATE):
    """Read a single-column ASCII waveform and prepare it for the AWG.

    This is a simplified Python version of readFile.m + loadfile.m for the
    'Oscilloscope (.txt)' case. It returns the same outputs as the MATLAB
    reference.
    """
    filename = Path(filename)
    if not filename.exists():
        raise FileNotFoundError(f"Waveform file not found: {filename}")

    print(f"[LOAD] Reading waveform from {filename}")
    lines = filename.read_text(encoding="utf-8", errors="ignore").splitlines()

    # Skip header lines until a numeric line is found (same logic as loadfile.m)
    start_idx = 0
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        if line.upper() in ("Y", "DATA") or line.upper().startswith("Y1"):
            start_idx = i + 1
            continue
        # If the line starts with a number, data begins here
        try:
            float(line.split(",")[0].split("\t")[0])
            start_idx = i
            break
        except ValueError:
            continue
    else:
        raise ValueError("No numeric data found in waveform file")

    # Read numeric data (one column)
    values = []
    for line in lines[start_idx:]:
        line = line.strip()
        if not line:
            continue
        try:
            values.append(float(line.split(",")[0].split("\t")[0]))
        except ValueError:
            break
    iqdata = np.asarray(values, dtype=float).reshape(-1, 1)

    # scaleMinMax = [-1, 1, 1]  -> symmetric scale to [-1, +1]
    scale_min, scale_max, symm = -1.0, 1.0, True
    if symm:
        max_val = np.max(np.abs(iqdata))
        if max_val > 0:
            scale = min(abs(scale_max / max_val), abs(scale_min / -max_val))
            iqdata = iqdata * scale

    # Compute repetition count to satisfy segment granularity & minimum size
    # (loadfile.m end section)
    seg_gran = SEGMENT_GRANULARITY
    seg_min = MIN_SEGMENT_SIZE
    length_orig = iqdata.shape[0]
    rpt = int(np.lcm(length_orig, seg_gran) // length_orig)
    while rpt * length_orig < seg_min:
        rpt += 1

    # channelMapping for a single-column real signal on a 2-channel M8190A
    channel_mapping = np.array([[1, 0], [0, 1]], dtype=int)

    # Default marker: first half high, second half low (download.m default)
    marker = np.concatenate([
        np.full(length_orig // 2, 15, dtype=np.uint16),
        np.zeros(length_orig - length_orig // 2, dtype=np.uint16)
    ])

    print(f"[LOAD] {length_orig} samples, fs={sample_rate/1e9:.3f} GHz, "
          f"rpt={rpt} -> segment length={length_orig*rpt}")
    return iqdata, sample_rate, marker, rpt, channel_mapping


# =============================================================================
# Download logic (MATLAB download.m / download_M8190A.m / gen_arb_M8190A.m)
# =============================================================================

def _fixlength(x, length):
    """MATLAB fixlength helper: tile or truncate vector to `length` elements."""
    x = np.asarray(x).ravel()
    if x.size == 0:
        return np.zeros(length)
    reps = int(np.ceil(length / x.size))
    return np.tile(x, reps)[:length]


def _do_run(f: ScpiSocket, channel_mapping: np.ndarray):
    """Start AWG output (MATLAB doRun, no M8192A sync)."""
    active_ch = np.where(channel_mapping.sum(axis=1) > 0)[0] + 1
    if len(active_ch) > 1:
        xfprintf(f, ":INST:COUP:STATe ON")
    for ch in active_ch:
        xfprintf(f, f":INIT:IMM{ch}")
    print(f"[RUN] Started channels: {list(active_ch)}")


def _gen_arb_m8190a(f: ScpiSocket, arb_config: dict, chan: int,
                    data: np.ndarray, marker: np.ndarray,
                    segm_num: int, run: int,
                    segment_length: int, segment_offset: int,
                    route: str):
    """Download one real waveform to one channel/segment (MATLAB gen_arb_M8190A)."""
    if not chan:
        return
    segm_len = data.size
    if segm_len > 0:
        # Delete old segment (ignore error if absent), then define new one
        if run >= 0 and segment_offset == 0:
            xfprintf(f, f":TRACe{chan}:DELete {segm_num}", ignore_error=True)
            xfprintf(f, f":TRACe{chan}:DEFine {segm_num},{segment_length}")

        # Scale to 12-bit DAC: int16(round(8191 * data) * 4)
        dac_data = np.int16(np.round(8191.0 * data) * 4)

        # Add marker low 2 bits
        if marker is not None and marker.size:
            if marker.size != dac_data.size:
                raise ValueError("marker length must equal data length")
            dac_data = dac_data + np.int16(np.bitwise_and(marker.astype(np.uint16), 3))

        # Download in chunks of 523200 int16 samples
        chunk = 523200
        offset = 0
        while offset < segm_len:
            block = dac_data[offset:offset + chunk]
            cmd = f":TRACe{chan}:DATA {segm_num},{offset + segment_offset},"
            xbinblockwrite(f, block, "int16", cmd)
            offset += chunk

        xquery(f, "*OPC?")

        if run >= 0 and (segment_offset + segm_len >= segment_length):
            xfprintf(f, f":TRACe{chan}:SELect {segm_num}")

    if segment_offset + segm_len >= segment_length:
        # Output routing & amplitude
        if route == "DAC":
            xfprintf(f, f":OUTPut{chan}:ROUTe DAC")
            amp_cmd = f":VOLTage{chan}:AMPLitude"
        else:
            xfprintf(f, f":OUTPut{chan}:ROUTe {route}")
            amp_cmd = f":{route}{chan}:VOLT:AMPL"
        xfprintf(f, f"{amp_cmd} {arb_config['amplitude']}")
        xfprintf(f, f":VOLTage{chan}:OFFSet {arb_config['offset']}")

        if run >= 0:
            xfprintf(f, f":FUNCtion{chan}:MODE ARBitrary")
        xfprintf(f, f":OUTPut{chan} ON")


def download_m8190a(f: ScpiSocket, arb_config: dict, fs: float,
                    data: np.ndarray, marker1: np.ndarray, marker2: np.ndarray,
                    segm_num: int, keep_open: bool,
                    channel_mapping: np.ndarray, run: int,
                    segment_length: int | None, segment_offset: int | None,
                    route: str):
    """M8190A-specific download (MATLAB download_M8190A.m, 2-channel direct mode)."""
    if segment_length is None:
        segment_length = data.shape[0]
    if segment_offset is None:
        segment_offset = 0

    # Limit to first 2 channels (first module)
    ch_map = np.asarray(channel_mapping, dtype=int)
    if ch_map.shape[0] > 2:
        ch_map = ch_map[:2, :]

    # Query options to determine one-channel vs two-channel
    opts = xquery(f, "*opt?")
    if "001" in opts:
        num_channels = 1
        ch_map = np.vstack([ch_map[0:1, :], np.zeros((1, ch_map.shape[1]), dtype=int)])
    else:
        num_channels = 2

    # Optional *RST omitted here (same as AWG_transmit.m which does not send *RST)

    # Stop output before reconfiguration
    if segment_offset == 0:
        for i in range(1, num_channels + 1):
            if ch_map[i - 1, :].sum() > 0:
                xfprintf(f, f":ABORt{i}")

    # Set sample rate, 12-bit WSP mode, clock source
    dwid = "WSPeed"  # M8190A_12bit
    clock_source = arb_config.get("clockSource", "ExtRef")
    for i in range(1, num_channels + 1):
        if ch_map[i - 1, :].sum() == 0:
            continue
        if clock_source == "ExtRef":
            cmd = (f":ROSC:SOURce EXT; :ROSC:FREQuency {arb_config['clockFreq']:.15g}; "
                   f":FREQuency:RASTer:SOURce{i} INTernal; :FREQuency:RASTer {fs:.15g};")
        else:
            cmd = f":FREQuency:RASTer {fs:.15g};"
        if num_channels == 1:
            cmd += f" :TRACe1:DWIDth {dwid};"
        else:
            cmd += f" :TRACe1:DWIDth {dwid}; :TRACe2:DWIDth {dwid};"
        if xfprintf(f, cmd) != 0:
            print("[ERROR] Failed to set sample rate / mode. Aborting.")
            return

    # Trigger/continuous mode
    trigger_mode = arb_config.get("triggerMode", "Continuous")
    cont_mode = 1 if trigger_mode == "Continuous" else 0
    gate_mode = 1 if trigger_mode == "Gated" else 0
    for i in range(1, num_channels + 1):
        if ch_map[i - 1, :].sum() > 0:
            xfprintf(f, f":INIT:CONTinuous{i} {cont_mode}; GATE{i} {gate_mode}")

    # Direct mode waveform download (real data)
    for col in range(ch_map.shape[1] // 2):
        for ch in np.where(ch_map[:, 2 * col] > 0)[0] + 1:
            _gen_arb_m8190a(f, arb_config, int(ch), np.real(data[:, col]).ravel(),
                            marker1, segm_num, run, segment_length, segment_offset, route)
        for ch in np.where(ch_map[:, 2 * col + 1] > 0)[0] + 1:
            _gen_arb_m8190a(f, arb_config, int(ch), np.imag(data[:, col]).ravel(),
                            marker2, segm_num, run, segment_length, segment_offset, route)

    # Start if full segment downloaded
    if segment_offset + data.shape[0] >= segment_length:
        _do_run(f, ch_map)

    if not keep_open:
        f.close()


def download(iqdata: np.ndarray, fs: float,
             channel_mapping: np.ndarray | None = None,
             segment_number: int = 1,
             marker: np.ndarray | None = None,
             arb_config: dict | None = None,
             port: int = DEFAULT_PORT,
             run: bool = True,
             route: str = "DC"):
    """Top-level download entry (MATLAB download.m simplified for M8190A_12bit)."""
    if arb_config is None:
        raise ValueError("arb_config is required")

    # Ensure column vector
    if iqdata.shape[0] < iqdata.shape[1]:
        iqdata = iqdata.T

    # Default channel mapping for a single-column real signal on 2 channels
    if channel_mapping is None:
        channel_mapping = np.array([[1, 0], [0, 1]], dtype=int)
    channel_mapping = np.asarray(channel_mapping, dtype=int)

    # Pad channel mapping width to 2 * number of data columns
    target_width = 2 * iqdata.shape[1]
    if channel_mapping.shape[1] < target_width:
        pad = np.zeros((channel_mapping.shape[0], target_width - channel_mapping.shape[1]), dtype=int)
        channel_mapping = np.hstack([channel_mapping, pad])

    # Normalize if needed (already normalized by read_file, keep for safety)
    scale = np.max(np.abs(iqdata))
    if scale > 1.0:
        iqdata = iqdata / scale

    # Markers: default square wave if not provided
    n = iqdata.shape[0]
    if marker is None:
        marker = np.concatenate([
            np.full(n // 2, 15, dtype=np.uint16),
            np.zeros(n - n // 2, dtype=np.uint16)
        ])
    marker = np.asarray(marker).ravel()
    marker1 = np.bitwise_and(marker.astype(np.uint16), 3)
    marker2 = np.bitwise_and(np.right_shift(marker.astype(np.uint16), 2), 3)

    # Check granularity
    seg_len = n
    if seg_len % SEGMENT_GRANULARITY != 0:
        raise ValueError(f"Segment size {seg_len} must be multiple of {SEGMENT_GRANULARITY}")
    if seg_len < MIN_SEGMENT_SIZE:
        raise ValueError(f"Segment size {seg_len} must be >= {MIN_SEGMENT_SIZE}")
    if seg_len > MAX_SEGMENT_SIZE:
        raise ValueError(f"Segment size {seg_len} must be <= {MAX_SEGMENT_SIZE}")

    host = "localhost"  # download_M8190A.m uses local host
    with ScpiSocket(host, port) as f:
        download_m8190a(f, arb_config, fs, iqdata, marker1, marker2,
                        segment_number, keep_open=False,
                        channel_mapping=channel_mapping, run=1 if run else -1,
                        segment_length=seg_len, segment_offset=0,
                        route=route)


# =============================================================================
# Top-level: AWG_transmit style configuration then download
# =============================================================================

def awg_transmit(iqdata: np.ndarray, fs: float, vpp: float,
                 host: str, port: int, route: str = "DC",
                 channel_mapping: np.ndarray | None = None):
    """Mirror of MATLAB AWG_transmit.m: configure, then download & run."""
    visa_addr = f"TCPIP0::{host}::{port}::SOCKET"
    arb_config = make_arb_config(visa_addr, vpp, fs, port)

    with ScpiSocket(host, port) as f:
        # Self-test (MATLAB AWG_transmit.m)
        pon = xquery(f, ":TEST:PON?")
        print(f"[INFO] :TEST:PON? -> {pon}")
        if '"Selftest passed"' not in pon and "Selftest passed" not in pon:
            print("[WARN] AWG self-test did not report 'Selftest passed'")

        # Reference clock & sample rate
        xfprintf(f, ":ROSC:FREQ 1e7")
        xfprintf(f, ":ROSC:SOUR EXT")
        xfprintf(f, f":FREQ:RAST {fs:.15g}")

        # Configure both channels (same as AWG_transmit.m)
        for ch in (1, 2):
            xfprintf(f, f":TRACe{ch}:DWIDth WSP")          # 12-bit wideband
            if route in ("DC", "AC"):
                xfprintf(f, f":{route}{ch}:FORM NRZ")       # NRZ format for amplified output
            xfprintf(f, f":{route}{ch}:VOLT:AMPL {vpp:.15g}")
            xfprintf(f, f":OUTP{ch}:ROUT {route}")
            xfprintf(f, f":OUTP{ch}:NORM ON")
            xfprintf(f, f":OUTP{ch}:COMP ON")
            xfprintf(f, f":TRAC{ch}:SEL 1")

        print("[INFO] begin to load sequence...")

    # Re-open socket inside download (download_M8190A.m creates its own tcpclient)
    download(iqdata, fs,
             channel_mapping=channel_mapping,
             segment_number=1,
             marker=None,
             arb_config=arb_config,
             port=port,
             run=True,
             route=route)

    time.sleep(3)
    print("[DONE] AWG transmit complete.")


# =============================================================================
# CLI / standalone execution
# =============================================================================

def _parse_args():
    p = argparse.ArgumentParser(description="Standalone M8190A waveform download test")
    p.add_argument("--host", default=DEFAULT_HOST, help="AWG host/IP")
    p.add_argument("--port", type=int, default=None, help="AWG raw TCP port (default 5025)")
    p.add_argument("-f", "--waveform", default=DEFAULT_WAVEFORM, help="Input .txt waveform")
    p.add_argument("--sample-rate", type=float, default=DEFAULT_SAMPLE_RATE, help="Sample rate Hz")
    p.add_argument("--amplitude", type=float, default=DEFAULT_AMPLITUDE, help="Output Vpp")
    p.add_argument("--route", default="DC", choices=OUTPUT_ROUTES,
                   help="Output path: DC (DC-coupled amp), AC (AC-coupled amp), DAC (direct DAC)")
    p.add_argument("--config-txt", default=DEFAULT_CONFIG_TXT,
                   help="RX_CHANNEL_SENSING_CONFIG.txt used by MATLAB to read port")
    p.add_argument("--no-run", action="store_true", help="Download only, do not start output")
    return p.parse_args()


def _read_port_from_config(path: str | Path) -> int:
    """Read line 9 (1-indexed) of RX_CHANNEL_SENSING_CONFIG.txt."""
    path = Path(path)
    if not path.exists():
        return DEFAULT_PORT
    nums = [float(x) for x in path.read_text().split()]
    if len(nums) >= 9:
        return int(nums[8])  # MATLAB indexing: 9th element -> Python index 8
    return DEFAULT_PORT


def main():
    args = _parse_args()
    port = args.port if args.port is not None else _read_port_from_config(args.config_txt)

    # Read waveform (same as MATLAB readFile)
    iqdata, fs, marker, rpt, ch_map = read_file(args.waveform, args.sample_rate)

    # Repeat to meet segment constraints (same as MATLAB AWGM8190A_Auto)
    iqdata = np.tile(iqdata, (rpt, 1))
    marker = np.tile(marker, rpt)

    print(f"[MAIN] host={args.host}, port={port}, fs={fs/1e9:.3f} GHz, "
          f"Vpp={args.amplitude}, route={args.route}, waveform_len={iqdata.shape[0]}")

    # Run the same flow as MATLAB AWG_transmit + download
    awg_transmit(iqdata, fs, args.amplitude, args.host, port,
                 route=args.route, channel_mapping=ch_map)


if __name__ == "__main__":
    main()
