"""NN 后均衡器封装。

直接调用现有的 ZY_BiGRU_GPU.py：
1. 将 Tx/Rx 波形写入 ZY_BiGRU_GPU.py 在 NN 目录下期望的文件名
2. 在 NN 目录中以子进程方式运行 ZY_BiGRU_GPU.py
3. 读取生成的 Rxdata_afterNN1.txt 并返回
"""
import numpy as np
import subprocess
import sys
import shutil
import os
import threading
from pathlib import Path
from typing import Optional

import config
from utils import save_txt, load_txt


class NNEqualizer:
    """ZY_BiGRU_GPU 封装。"""

    def __init__(self,
                 nn_dir: Path = config.NN_DIR,
                 script_name: str = "ZY_BiGRU_GPU.py",
                 python_exe: Optional[str] = None):
        self.nn_dir = Path(nn_dir)
        self.script = self.nn_dir / script_name
        self.python_exe = python_exe or sys.executable

    def run(self,
            tx_waveform: np.ndarray,
            rx_waveform: np.ndarray,
            output_name: str = "Rxdata_afterNN1.txt",
            epochs: Optional[int] = None,
            use_pretrained: bool = True) -> np.ndarray:
        """运行 NN 均衡。

        Args:
            tx_waveform: 发射参考波形
            rx_waveform: 接收波形
            output_name: NN 输出文件名
            epochs: 训练轮数（若提供，则临时修改脚本中的 epochs）
            use_pretrained: 是否优先使用已训练的 trained_model_temp.pth

        Returns:
            NN 均衡后的波形（一维 numpy 数组）
        """
        if not self.script.exists():
            raise FileNotFoundError(f"未找到 NN 脚本: {self.script}")

        # 归一化后写入 NN 目录（与原始脚本一致）
        tx_norm = tx_waveform / np.max(np.abs(tx_waveform))
        rx_norm = rx_waveform / np.max(np.abs(rx_waveform))

        save_txt(self.nn_dir / "Txdata_NN.txt", tx_norm)
        save_txt(self.nn_dir / "Rxdata_NN1.txt", rx_norm)

        # 构建环境变量
        env = os.environ.copy()
        env["MPLBACKEND"] = "Agg"  # 避免 plt.show() 阻塞
        env["DISABLE_TQDM"] = "1"  # 禁用 NN 训练进度条

        # 以子进程方式运行 NN 脚本（实时流式输出，供 GUI 进度显示）
        cmd = [self.python_exe, str(self.script)]
        print(f"正在运行 NN 均衡器: {' '.join(cmd)}")
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
                try:
                    print(line, end="")
                except UnicodeEncodeError:
                    print(line.encode("utf-8", errors="replace")
                          .decode("gbk", errors="replace"), end="")

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


def run_nn_equalizer(tx_waveform: np.ndarray,
                     rx_waveform: np.ndarray,
                     nn_dir: Path = config.NN_DIR) -> np.ndarray:
    """便捷函数。"""
    eq = NNEqualizer(nn_dir=nn_dir)
    return eq.run(tx_waveform, rx_waveform)
