"""Carrierless Amplitude Phase (CAP) modulation core.

Ports MATLAB functions:
  - CAPmod.m
  - cap_gen.m
  - shaping_fildes.m
  - Pulse_shaping_ZY.m / Gen_CAP_filters_ZY.m

Implements single-band and multi-band CAP transmitters.
"""
from typing import List, Optional, Tuple, Union

import numpy as np
from scipy.signal import convolve, resample_poly, upfirdn


def srrc_filter(rolloff: float, span: int, sps: int) -> np.ndarray:
    """Square-root raised cosine filter.

    Equivalent to MATLAB ``rcosdesign(rolloff, span, sps, 'sqrt')``.
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
    # Normalise energy to 1
    h = h / np.sqrt(np.sum(h ** 2))
    return h


def srrc_filter_full(
    rolloff: float,
    upsamplesymbol: int,
    upsampleno: int,
) -> np.ndarray:
    """Generate full-length SRRC at the sample rate and truncate if needed.

    Matches the MATLAB gtr computation in cap_gen.m / Pulse_shaping_ZY.m.
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

    # Special case: centre tap
    centre_idx = upsamplesymbol // 2
    gtr[centre_idx] = 1 + rolloff * (4 / np.pi - 1)

    # Special case: x = ±1 (|t_norm * r| = 1 / (4*rolloff))
    special_mask = np.isclose(np.abs(x), 1.0) & (np.arange(upsamplesymbol) != centre_idx)
    gtr[special_mask] = (rolloff / np.sqrt(2)) * (
        (1 + 2 / np.pi) * np.sin(np.pi / (4 * rolloff))
        + (1 - 2 / np.pi) * np.cos(np.pi / (4 * rolloff))
    )
    return gtr


def shaping_filter(rolloff: float, span: int, sps: int, shape: str = "srrc") -> np.ndarray:
    """Generate pulse shaping filter.

    Parameters
    ----------
    rolloff : float
        Roll-off factor.
    span : int
        Filter span in symbols.
    sps : int
        Samples per symbol.
    shape : str
        "rc", "srrc", or "btn".

    Returns
    -------
    h : np.ndarray
        1-D real filter coefficients.
    """
    shape = shape.lower()
    if shape.startswith("srrc"):
        return srrc_filter(rolloff, span, sps)
    if shape.startswith("rc"):
        # MATLAB rcosdesign(..., 'normal')
        h = srrc_filter(rolloff, span, sps)
        # RC is the convolution of two SRRC filters; approximate via scipy
        h_rc = np.convolve(h, h)
        return h_rc / np.sqrt(np.sum(h_rc ** 2))
    if shape.startswith("btn"):
        # BTN filter from Paul Haigh's CAP paper
        delay = span * sps // 2
        t = (np.arange(-delay, delay + 1)) / sps
        h = (
            np.sinc(t)
            * (2 * np.pi * rolloff * t / np.log(2) * np.sin(np.pi * rolloff * t) + 2 * np.cos(np.pi * rolloff * t) - 1)
            / ((np.pi * rolloff * t / np.log(2)) ** 2 + 1)
        )
        return h / np.sqrt(np.sum(h ** 2))
    raise ValueError(f"Unknown shape: {shape}")


def capmod(
    complex_sym: np.ndarray,
    gt: np.ndarray,
    t: np.ndarray,
    fc: float,
    taps: int,
    upsampleno: int,
) -> np.ndarray:
    """Single-band CAP modulator (ports CAPmod.m).

    Parameters
    ----------
    complex_sym : np.ndarray
        1-D complex symbol sequence.
    gt : np.ndarray
        Real baseband shaping filter (SRRC), length == taps.
    t : np.ndarray
        Time vector corresponding to filter taps, centred at 0.
    fc : float
        Carrier frequency for this CAP band (Hz).
    taps : int
        Filter length (odd).
    upsampleno : int
        Upsampling factor.

    Returns
    -------
    cap_signal : np.ndarray
        Real passband CAP waveform, power-normalised to 1.
    """
    complex_sym = np.asarray(complex_sym).flatten()
    gtI = gt * np.cos(2 * np.pi * fc * t)
    gtQ = gt * np.sin(2 * np.pi * fc * t)

    Idata = np.zeros(len(complex_sym) * upsampleno, dtype=float)
    Qdata = np.zeros_like(Idata)
    Idata[::upsampleno] = complex_sym.real
    Qdata[::upsampleno] = complex_sym.imag

    # Circular extension for filter transient
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
    """CAP modulator variant used in duo-binary CAP (ports mdb_mod.m).

    Applies pulse shaping first, then heterodynes real/imag parts separately.
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
    """Compute multi-band CAP parameters (ports main_CAP_3band_totalB.m setup).

    Parameters
    ----------
    Rs : float
        Aggregate baud rate (symbols/s).
    m : int
        Number of bands.
    rolloff : float
        Roll-off factor.
    cf : float
        Compression factor (0 < cf < 1). Smaller -> more spectral overlap.
    fs : float
        Sampling rate (Hz).

    Returns
    -------
    fc : np.ndarray
        Centre frequencies of each band (Hz), ordered from high to low.
    upsampleno : int
        Total upsampling factor = round(m * fs / Rs).
    taps : int
        Filter length = span * upsampleno + 1.
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
    """Generate multi-band CAP transmit waveform.

    Parameters
    ----------
    symbols_per_band : list of np.ndarray
        Complex symbol sequence for each band. All must have the same length.
    Rs : float
        Aggregate baud rate (symbols/s).
    fs : float
        Sampling rate (Hz).
    rolloff : float
        Roll-off factor.
    cf : float
        Compression factor.
    span : int
        Filter span in symbols.
    shape : str
        Pulse shape: "srrc", "rc", "btn".

    Returns
    -------
    tx_signal : np.ndarray
        Power-normalised multi-band CAP waveform.
    fc : np.ndarray
        Centre frequencies used for each band.
    gt : np.ndarray
        Baseband shaping filter.
    t : np.ndarray
        Filter time vector.
    band_signals : np.ndarray
        2-D array (num_bands, N) of individual band signals before summation.
    """
    m = len(symbols_per_band)
    if m == 0:
        raise ValueError("At least one band required")
    numofsymbols = len(symbols_per_band[0])
    if any(len(s) != numofsymbols for s in symbols_per_band):
        raise ValueError("All bands must have the same number of symbols")

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
    # Truncate to taps, centred
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
    """Generate single-band CAP transmit waveform (ports oldcapAPSKTxRx20220406.m).

    Parameters
    ----------
    symbols : np.ndarray
        1-D complex symbol sequence.
    fs : float
        Sampling rate (Hz).
    Rs : float
        Symbol rate (symbols/s).
    rolloff : float
        Roll-off factor.
    subcar : float
        Normalised carrier placement parameter (0.5 centres the band).
    start_freq : float
        Additional low-frequency offset in normalised units.
    taps : int
        Filter length (odd).
    shape : str
        Pulse shape.

    Returns
    -------
    tx_signal : np.ndarray
        Power-normalised CAP waveform.
    filter_I : np.ndarray
        I-path shaping filter.
    filter_Q : np.ndarray
        Q-path shaping filter.
    t : np.ndarray
        Full-length filter time vector.
    """
    symbols = np.asarray(symbols).flatten()
    upsampleno = int(round(fs / Rs))
    upsamplesymbol = len(symbols) * upsampleno
    # MATLAB: linspace(1,upsamplesymbol,upsamplesymbol)-1-upsamplesymbol/2
    t_norm = np.arange(upsamplesymbol, dtype=float) - upsamplesymbol / 2.0
    t = t_norm / fs

    # Normalised symbol rate = 1/upsampleno
    r = 1.0 / upsampleno
    gtr = srrc_filter_full(rolloff, upsamplesymbol, upsampleno)

    BW = 1 + rolloff
    subcar1 = BW * subcar + start_freq

    filter_I_full = gtr * np.cos(2 * np.pi * subcar1 * t_norm * r)
    filter_Q_full = gtr * np.sin(2 * np.pi * subcar1 * t_norm * r)

    # Truncate to requested taps, centred
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
