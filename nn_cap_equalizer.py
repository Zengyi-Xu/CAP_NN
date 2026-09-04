"""多频带 CAP 的 NN 后均衡器封装。

以子进程方式调用 data/nn/CAP_multiband_NN.py，并读取输出符号。
"""
import os
import subprocess
import sys
import threading
from pathlib import Path
from typing import Optional

import numpy as np

import config_cap as cfg
from utils import load_txt, save_txt


class CAPNNEqualizer:
    """CAP_multiband_NN.py 的封装。"""

    def __init__(
        self,
        nn_dir: Path = cfg.NN_DIR,
        script_name: str = "CAP_multiband_NN.py",
        python_exe: Optional[str] = None,
    ):
        self.nn_dir = Path(nn_dir)
        self.script = self.nn_dir / script_name
        self.python_exe = python_exe or sys.executable

    def run(
        self,
        tx_symbols: np.ndarray,
        rx_symbols: np.ndarray,
        output_name: str = "Rxdata_afterNN1.txt",
    ) -> np.ndarray:
        """运行 NN 均衡。

        Parameters
        ----------
        tx_symbols : np.ndarray
            发射符号，形状 (N, 6)，各频带实部/虚部交错排列。
        rx_symbols : np.ndarray
            接收（匹配滤波后）符号，形状 (N, 6)。
        output_name : str
            用于读取预测结果的输出文件名。

        Returns
        -------
        np.ndarray
            NN 均衡后的符号，形状 (M, 6)。
        """
        if not self.script.exists():
            raise FileNotFoundError(f"未找到 NN 脚本: {self.script}")

        save_txt(self.nn_dir / "Txdata_NN.txt", tx_symbols)
        save_txt(self.nn_dir / "Rxdata_NN1.txt", rx_symbols)

        env = os.environ.copy()
        env["MPLBACKEND"] = "Agg"
        env["DISABLE_TQDM"] = "1"

        cmd = [self.python_exe, str(self.script)]
        print(f"正在运行 CAP NN 均衡器: {' '.join(cmd)}")
        proc = subprocess.Popen(
            cmd,
            cwd=str(self.nn_dir),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )

        def _stream():
            for line in proc.stdout:
                print(line, end="")

        reader = threading.Thread(target=_stream, daemon=True)
        reader.start()
        code = proc.wait()
        reader.join(timeout=2)
        if code != 0:
            raise RuntimeError(f"NN 脚本运行失败，返回码 {code}")

        output_file = self.nn_dir / output_name
        if not output_file.exists():
            raise FileNotFoundError(f"未找到 NN 输出: {output_file}")
        return load_txt(output_file)


def run_cap_nn_equalizer(
    tx_symbols: np.ndarray,
    rx_symbols: np.ndarray,
    nn_dir: Path = cfg.NN_DIR,
) -> np.ndarray:
    """便捷函数。"""
    eq = CAPNNEqualizer(nn_dir=nn_dir)
    return eq.run(tx_symbols, rx_symbols)
