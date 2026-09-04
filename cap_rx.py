"""Carrierless Amplitude Phase (CAP) demodulation core.

Ports MATLAB functions:
  - CAPmatch_filter.m
  - mdb_match_filter.m
"""
from typing import Optional, Tuple

import numpy as np
from scipy.signal import convolve, upfirdn

from constellation import average_power, demodulate, load_constellation


def capmatch_filter(
    rx_signal: np.ndarray,
    gt: np.ndarray,
    t: np.ndarray,
    fc: float,
    taps: int,
    upsampleno: int,
    offsetsample: int = 0,
) -> np.ndarray:
    """Single-band CAP matched filter / down-converter (ports CAPmatch_filter.m).

    Parameters
    ----------
    rx_signal : np.ndarray
        Real received multi-band or single-band CAP waveform.
    gt : np.ndarray
        Baseband shaping filter, length == taps.
    t : np.ndarray
        Time vector corresponding to filter taps.
    fc : float
        Centre frequency of the desired CAP band.
    taps : int
        Filter length (odd).
    upsampleno : int
        Upsampling factor.
    offsetsample : int
        Downsample phase offset.

    Returns
    -------
    complex_data : np.ndarray
        Complex symbols at symbol rate after matched filtering.
    """
    rx_signal = np.asarray(rx_signal).flatten()
    gtI = gt * np.cos(2 * np.pi * fc * t)
    gtQ = gt * np.sin(2 * np.pi * fc * t)

    half = (taps - 1) // 2
    data_ext = np.concatenate([rx_signal[-half:], rx_signal, rx_signal[:half]])

    I = convolve(data_ext, gtI, mode="full")
    Q = convolve(data_ext, gtQ, mode="full")

    DataCap = I[taps - 1 : -(taps - 1)] + 1j * Q[taps - 1 : -(taps - 1)]

    # Downsample to symbol rate
    received = DataCap[offsetsample::upsampleno]
    received = received - np.mean(received)
    received = received / np.sqrt(np.mean(np.abs(received) ** 2))
    return received


def mdb_match_filter(
    rx_signal: np.ndarray,
    gt: np.ndarray,
    t: np.ndarray,
    fc: float,
    upsampleno: int,
    offsetsample: int = 0,
) -> np.ndarray:
    """Duo-binary CAP matched filter (ports mdb_match_filter.m).

    Mixes down first, then applies baseband filter.
    """
    rx_signal = np.asarray(rx_signal).flatten()
    down_data = rx_signal * np.cos(2 * np.pi * fc * t) - 1j * rx_signal * np.sin(2 * np.pi * fc * t)
    band_data = convolve(down_data, gt, mode="same")
    band_data = band_data[offsetsample::upsampleno]
    band_data = band_data - np.mean(band_data)
    band_data = band_data / np.sqrt(np.mean(np.abs(band_data) ** 2))
    return band_data


def demodulate_with_ber(
    rx_symbols: np.ndarray,
    order: int,
    constellation: str,
    origin_data: np.ndarray,
    skip_head: int = 0,
    skip_tail: int = 0,
) -> Tuple[np.ndarray, float, float]:
    """Demodulate received symbols and compute SER/BER.

    Parameters
    ----------
    rx_symbols : np.ndarray
        Received complex symbols.
    order : int
        Constellation order.
    constellation : str
        "APSK" or "QAM".
    origin_data : np.ndarray
        Original transmitted decimal symbols.
    skip_head, skip_tail : int
        Number of symbols to discard at the beginning/end (filter transients).

    Returns
    -------
    decisions : np.ndarray
        Demodulated decimal symbols.
    ser : float
        Symbol error rate.
    ber : float
        Bit error rate (assumes log2(M) bits per symbol).
    """
    rx_symbols = np.asarray(rx_symbols).flatten()
    origin_data = np.asarray(origin_data).flatten()
    cons = load_constellation(order, constellation)
    avp = float(np.sqrt(np.mean(np.abs(cons) ** 2)))
    rx_symbols = rx_symbols / np.sqrt(np.mean(np.abs(rx_symbols) ** 2)) * avp

    decisions = demodulate(rx_symbols, order, constellation)
    valid = slice(skip_head, len(decisions) - skip_tail if skip_tail else None)
    dec_valid = decisions[valid]
    tx_valid = origin_data[valid]

    ser = float(np.mean(dec_valid != tx_valid))
    bits_per_sym = int(np.log2(order))
    ber = float(np.sum(dec_valid != tx_valid) * bits_per_sym / (len(tx_valid) * bits_per_sym))
    return decisions, ser, ber
