"""LMS and LMS+Volterra equalizers.

Ports MATLAB functions:
  - LMS_1DownS_Testnan.m
  - LMS_volterra_1DownS_Testnan.m
"""
from typing import Optional, Tuple

import numpy as np


def lms_equalizer(
    rxdata: np.ndarray,
    txdata: np.ndarray,
    taps_lms: int,
    mu_lms: float,
    numof_ts: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Symbol-rate LMS linear equalizer.

    Parameters
    ----------
    rxdata : np.ndarray
        Received complex symbols (1-D).
    txdata : np.ndarray
        Transmitted complex symbols (1-D), aligned with rxdata.
    taps_lms : int
        Number of LMS taps (odd).
    mu_lms : float
        LMS step size.
    numof_ts : int
        Number of training symbols.

    Returns
    -------
    k : np.ndarray
        Equalised output, same length as rxdata (head/tail padded).
    y : np.ndarray
        Training-stage outputs.
    E : np.ndarray
        Training-stage errors.
    W : np.ndarray
        Converged LMS tap weights.
    """
    rxdata = np.asarray(rxdata).flatten()
    txdata = np.asarray(txdata).flatten()

    rxdata = rxdata / np.sqrt(np.mean(np.abs(rxdata) ** 2))
    txdata = txdata / np.sqrt(np.mean(np.abs(txdata) ** 2))

    x = rxdata[:numof_ts]
    d = txdata[:numof_ts]

    compensation = taps_lms
    half = (taps_lms - 1) // 2
    W = np.zeros(taps_lms, dtype=complex)

    ntr = len(x)
    y = np.zeros(ntr, dtype=complex)
    E = np.zeros(ntr, dtype=complex)

    nn = 0
    n = compensation - 1
    while n < ntr:
        # MATLAB indexing is 1-based; central tap aligned to n-compensation/2+1/2
        idx = n - half
        X_lms = x[idx - half : idx + half + 1][::-1]
        y[nn] = np.dot(W, X_lms)
        e = d[idx] - y[nn]
        W = W + mu_lms * e * np.conj(X_lms)
        E[nn] = e
        n += 1
        nn += 1

    # Apply to whole sequence
    k = np.zeros(len(rxdata), dtype=complex)
    n = compensation - 1
    mm = 0
    while n < len(rxdata):
        idx = n - half
        X_lms = rxdata[idx - half : idx + half + 1][::-1]
        k[mm] = np.dot(W, X_lms)
        n += 1
        mm += 1

    # Pad head/tail to preserve length
    head = (compensation - 1) // 2
    tail = len(rxdata) - mm
    k = np.concatenate([rxdata[:head], k[:mm], rxdata[-tail:]]) if tail > 0 else np.concatenate([rxdata[:head], k[:mm]])
    k = k / np.sqrt(np.mean(np.abs(k) ** 2))
    return k, y, E, W


def lms_volterra_equalizer(
    rxdata: np.ndarray,
    txdata: np.ndarray,
    taps_lms: int,
    mu_lms: float,
    taps_volterra: int,
    mu_volterra: float,
    numof_ts: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Symbol-rate LMS linear + 2nd-order Volterra equalizer.

    Parameters
    ----------
    rxdata, txdata : np.ndarray
        Received and transmitted complex symbols.
    taps_lms : int
        Linear LMS taps.
    mu_lms : float
        Linear step size.
    taps_volterra : int
        Volterra memory length.
    mu_volterra : float
        Volterra step size.
    numof_ts : int
        Training length.

    Returns
    -------
    k : np.ndarray
        Equalised output.
    y, E, W : as in lms_equalizer.
    V : np.ndarray
        Converged Volterra kernel (upper-triangular matrix).
    X_v : np.ndarray
        Last Volterra input matrix (diagnostic).
    """
    rxdata = np.asarray(rxdata).flatten()
    txdata = np.asarray(txdata).flatten()

    rxdata = rxdata / np.sqrt(np.mean(np.abs(rxdata) ** 2))
    txdata = txdata / np.sqrt(np.mean(np.abs(txdata) ** 2))

    x = rxdata[:numof_ts]
    d = txdata[:numof_ts]

    compensation = max(taps_lms, taps_volterra)
    half_lms = (taps_lms - 1) // 2
    half_vol = (taps_volterra - 1) // 2

    W = np.zeros(taps_lms, dtype=complex)
    delta = 0.0000001
    V = delta * np.eye(taps_volterra, dtype=complex)

    ntr = len(x)
    y = np.zeros(ntr, dtype=complex)
    E = np.zeros(ntr, dtype=complex)

    nn = 0
    n = compensation - 1
    X_v = np.zeros((taps_volterra, taps_volterra), dtype=complex)
    while n < ntr:
        idx_lms = n - half_lms
        X_lms = x[idx_lms - half_lms : idx_lms + half_lms + 1][::-1]

        idx_vol = n - half_vol
        X_vol = x[idx_vol - half_vol : idx_vol + half_vol + 1][::-1]
        X_v = np.triu(np.outer(X_vol, np.conj(X_vol)))
        VV = np.sum(V * X_v)

        y[nn] = np.dot(W, X_lms) + VV
        e = d[idx_lms] - y[nn]
        W = W + mu_lms * e * np.conj(X_lms)
        V = V + mu_volterra * e * np.conj(X_v)
        E[nn] = e
        n += 1
        nn += 1

    # Apply to whole sequence
    k = np.zeros(len(rxdata), dtype=complex)
    n = compensation - 1
    mm = 0
    while n < len(rxdata):
        idx_lms = n - half_lms
        X_lms = rxdata[idx_lms - half_lms : idx_lms + half_lms + 1][::-1]

        idx_vol = n - half_vol
        X_vol = rxdata[idx_vol - half_vol : idx_vol + half_vol + 1][::-1]
        R_v = np.triu(np.outer(X_vol, np.conj(X_vol)))
        R_vv = np.sum(V * R_v)

        k[mm] = np.dot(W, X_lms) + R_vv
        n += 1
        mm += 1

    head = (compensation - 1) // 2
    tail = len(rxdata) - mm
    if tail > 0:
        k = np.concatenate([rxdata[:head], k[:mm], rxdata[-tail:]])
    else:
        k = np.concatenate([rxdata[:head], k[:mm]])
    k = k / np.sqrt(np.mean(np.abs(k) ** 2))
    return k, y, E, W, V, X_v
