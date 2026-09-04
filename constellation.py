"""APSK/QAM constellation mapping/demapping for UW_APSK CAP.

Ports MATLAB functions:
  - GS_CCSDSmodulation_cons.m
  - GS_CCSDSdemodulation_cons.m

Constellation tables are plain text files with two columns (I, Q) ordered by
symbol index 0..M-1, e.g. CCSDS32APSK.txt, CCSDS64QAM.txt.
"""
from pathlib import Path
from typing import Union

import numpy as np

from config import DATA_DIR


def generate_standard_qam(order: int) -> np.ndarray:
    """Generate standard square QAM constellation normalised to unit average power.

    Matches MATLAB ``qammod(0:order-1, order)``.
    """
    if int(np.log2(order)) % 2 != 0:
        # Cross QAM: use a rectangular grid with the right number of points
        cols = int(2 ** np.ceil(np.log2(order) / 2))
        rows = int(2 ** np.floor(np.log2(order) / 2))
        x = np.arange(cols) - (cols - 1) / 2
        y = np.arange(rows) - (rows - 1) / 2
        xv, yv = np.meshgrid(x, y)
        cons = (xv + 1j * yv).flatten()
        cons = cons[:order]
    else:
        m = int(np.sqrt(order))
        x = np.arange(m) - (m - 1) / 2
        xv, yv = np.meshgrid(x, x)
        # MATLAB qammod orders symbols along columns (Gray code not required for Tx mapping)
        cons = (xv + 1j * yv).flatten()
    # Normalize to unit average power, matching MATLAB qammod
    cons = cons / np.sqrt(np.mean(np.abs(cons) ** 2))
    return cons


def load_constellation(order: int, constellation: str = "APSK", data_dir: Union[str, Path] = DATA_DIR) -> np.ndarray:
    """Load CCSDS{order}{constellation}.txt as a complex array of length `order`.

    If the file does not exist, fall back to standard QAM (for QAM) or raise an
    error (for APSK, because APSK ring geometries are project-specific).

    Parameters
    ----------
    order : int
        Constellation order, e.g. 32, 64, 128.
    constellation : str
        "APSK" or "QAM".
    data_dir : path-like
        Directory containing the constellation text files.

    Returns
    -------
    cons : np.ndarray
        1-D complex array, cons[symbol_index] = I + 1j*Q.
    """
    data_dir = Path(data_dir)
    filename = data_dir / f"CCSDS{order}{constellation}.txt"
    if not filename.exists():
        if constellation.upper() == "QAM":
            return generate_standard_qam(order)
        raise FileNotFoundError(
            f"Constellation file {filename} not found and no standard generator for {constellation}"
        )
    table = np.loadtxt(filename)
    if table.ndim != 2 or table.shape[1] != 2:
        raise ValueError(f"Constellation file {filename} must have two columns")
    if table.shape[0] != order:
        raise ValueError(f"Constellation file {filename} has {table.shape[0]} points, expected {order}")
    return table[:, 0] + 1j * table[:, 1]


def modulate(decimal_symbols: np.ndarray, order: int, constellation: str = "APSK", data_dir: Union[str, Path] = DATA_DIR) -> np.ndarray:
    """Map decimal symbols to complex constellation points.

    Parameters
    ----------
    decimal_symbols : np.ndarray
        Integer symbols in [0, order-1].
    order : int
        Constellation order.
    constellation : str
        "APSK" or "QAM".

    Returns
    -------
    qamdata : np.ndarray
        Complex constellation points, same shape as input.
    """
    decimal_symbols = np.asarray(decimal_symbols)
    cons = load_constellation(order, constellation, data_dir)
    if np.any(decimal_symbols < 0) or np.any(decimal_symbols >= order):
        raise ValueError("decimal_symbols out of range")
    return cons[decimal_symbols]


def demodulate(rx_symbols: np.ndarray, order: int, constellation: str = "APSK", data_dir: Union[str, Path] = DATA_DIR) -> np.ndarray:
    """Minimum-distance demodulation of complex symbols to decimal decisions.

    Parameters
    ----------
    rx_symbols : np.ndarray
        Received complex symbols.
    order : int
        Constellation order.
    constellation : str
        "APSK" or "QAM".

    Returns
    -------
    decisions : np.ndarray
        Integer symbols in [0, order-1], same shape as input.
    """
    rx_symbols = np.asarray(rx_symbols)
    cons = load_constellation(order, constellation, data_dir)
    # (N, M) distance matrix
    distances = np.abs(rx_symbols[..., None] - cons[None, :])
    return np.argmin(distances, axis=-1)


def average_power(order: int, constellation: str = "APSK", data_dir: Union[str, Path] = DATA_DIR) -> float:
    """Return average power of the specified constellation."""
    cons = load_constellation(order, constellation, data_dir)
    return float(np.sqrt(np.mean(np.abs(cons) ** 2)))
