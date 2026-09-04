"""UW_APSK CAP 的 APSK/QAM 星座映射/解映射。

移植自 MATLAB 函数：
  - GS_CCSDSmodulation_cons.m
  - GS_CCSDSdemodulation_cons.m

星座表为两列（I、Q）的纯文本文件，按符号序号 0..M-1 排列，
例如 CCSDS32APSK.txt、CCSDS64QAM.txt。
"""
from pathlib import Path
from typing import Union

import numpy as np

from config import DATA_DIR


def generate_standard_qam(order: int) -> np.ndarray:
    """生成归一化到单位平均功率的标准方形 QAM 星座。

    与 MATLAB ``qammod(0:order-1, order)`` 一致。
    """
    if int(np.log2(order)) % 2 != 0:
        # 十字 QAM：使用点数正确的矩形网格
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
        # MATLAB qammod 按列排序符号（发射端映射无需格雷码）
        cons = (xv + 1j * yv).flatten()
    # 归一化到单位平均功率，与 MATLAB qammod 一致
    cons = cons / np.sqrt(np.mean(np.abs(cons) ** 2))
    return cons


def load_constellation(order: int, constellation: str = "APSK", data_dir: Union[str, Path] = DATA_DIR) -> np.ndarray:
    """加载 CCSDS{order}{constellation}.txt，返回长度为 `order` 的复数数组。

    若文件不存在，QAM 回退到标准 QAM 生成，APSK 则抛出异常
    （因为 APSK 的环几何结构是项目相关的）。

    参数
    ----------
    order : int
        星座阶数，例如 32、64、128。
    constellation : str
        "APSK" 或 "QAM"。
    data_dir : path-like
        存放星座文本文件的目录。

    返回
    -------
    cons : np.ndarray
        一维复数数组，cons[symbol_index] = I + 1j*Q。
    """
    data_dir = Path(data_dir)
    filename = data_dir / f"CCSDS{order}{constellation}.txt"
    if not filename.exists():
        if constellation.upper() == "QAM":
            return generate_standard_qam(order)
        raise FileNotFoundError(
            f"星座文件 {filename} 不存在，且 {constellation} 没有可用的标准生成器"
        )
    table = np.loadtxt(filename)
    if table.ndim != 2 or table.shape[1] != 2:
        raise ValueError(f"星座文件 {filename} 必须包含两列")
    if table.shape[0] != order:
        raise ValueError(f"星座文件 {filename} 含 {table.shape[0]} 个点，期望为 {order}")
    return table[:, 0] + 1j * table[:, 1]


def modulate(decimal_symbols: np.ndarray, order: int, constellation: str = "APSK", data_dir: Union[str, Path] = DATA_DIR) -> np.ndarray:
    """将十进制符号映射为复数星座点。

    参数
    ----------
    decimal_symbols : np.ndarray
        [0, order-1] 范围内的整数符号。
    order : int
        星座阶数。
    constellation : str
        "APSK" 或 "QAM"。

    返回
    -------
    qamdata : np.ndarray
        复数星座点，形状与输入相同。
    """
    decimal_symbols = np.asarray(decimal_symbols)
    cons = load_constellation(order, constellation, data_dir)
    if np.any(decimal_symbols < 0) or np.any(decimal_symbols >= order):
        raise ValueError("decimal_symbols 超出范围")
    return cons[decimal_symbols]


def demodulate(rx_symbols: np.ndarray, order: int, constellation: str = "APSK", data_dir: Union[str, Path] = DATA_DIR) -> np.ndarray:
    """将复数符号按最小距离准则解调为十进制判决结果。

    参数
    ----------
    rx_symbols : np.ndarray
        接收到的复数符号。
    order : int
        星座阶数。
    constellation : str
        "APSK" 或 "QAM"。

    返回
    -------
    decisions : np.ndarray
        [0, order-1] 范围内的整数符号，形状与输入相同。
    """
    rx_symbols = np.asarray(rx_symbols)
    cons = load_constellation(order, constellation, data_dir)
    # (N, M) 距离矩阵
    distances = np.abs(rx_symbols[..., None] - cons[None, :])
    return np.argmin(distances, axis=-1)


def average_power(order: int, constellation: str = "APSK", data_dir: Union[str, Path] = DATA_DIR) -> float:
    """返回指定星座的平均功率。"""
    cons = load_constellation(order, constellation, data_dir)
    return float(np.sqrt(np.mean(np.abs(cons) ** 2)))
