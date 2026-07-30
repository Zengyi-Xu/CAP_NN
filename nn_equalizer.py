"""NN 后均衡器包装器.

直接调用现有的 ZY_BiGRU_GPU.py：
1. 把 Tx/Rx 波形写入 NN 目录下 ZY_BiGRU_GPU.py 期望的文件名
2. 在 NN 目录中以子进程方式运行 ZY_BiGRU_GPU.py
3. 读取生成的 Rxdata_afterNN1.txt 并返回
"""
import numpy as np
import subprocess
import sys
import shutil
import os
from pathlib import Path
from typing import Optional

import config
from utils import save_txt, load_txt


class NNEqualizer:
    """ZY_BiGRU_GPU 包装器."""

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
        """运行 NN 均衡.

        Args:
            tx_waveform: 发送端参考波形
            rx_waveform: 接收端波形
            output_name: NN 输出文件名
            epochs: 训练轮数（若提供则临时修改脚本中的 epochs）
            use_pretrained: 是否优先使用已训练的 trained_model_temp.pth

        Returns:
            NN 均衡后的波形 (1-D numpy 数组)
        """
        if not self.script.exists():
            raise FileNotFoundError(f"NN script not found: {self.script}")

        # 归一化并写入 NN 目录（与原始脚本保持一致）
        tx_norm = tx_waveform / np.max(np.abs(tx_waveform))
        rx_norm = rx_waveform / np.max(np.abs(rx_waveform))

        save_txt(self.nn_dir / "Txdata_NN.txt", tx_norm)
        save_txt(self.nn_dir / "Rxdata_NN1.txt", rx_norm)

        # 构造环境变量
        env = os.environ.copy()
        env["MPLBACKEND"] = "Agg"  # 避免 plt.show() 阻塞

        # 子进程运行 NN 脚本
        cmd = [self.python_exe, str(self.script)]
        print(f"Running NN equalizer: {' '.join(cmd)}")
        result = subprocess.run(
            cmd,
            cwd=str(self.nn_dir),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace"
        )
        try:
            print(result.stdout)
        except UnicodeEncodeError:
            print(result.stdout.encode("utf-8", errors="replace").decode("gbk", errors="replace"))
        if result.returncode != 0:
            raise RuntimeError(f"NN script failed with return code {result.returncode}")

        output_file = self.nn_dir / output_name
        if not output_file.exists():
            raise FileNotFoundError(f"NN output not found: {output_file}")

        return load_txt(output_file)


def run_nn_equalizer(tx_waveform: np.ndarray,
                     rx_waveform: np.ndarray,
                     nn_dir: Path = config.NN_DIR) -> np.ndarray:
    """便捷函数."""
    eq = NNEqualizer(nn_dir=nn_dir)
    return eq.run(tx_waveform, rx_waveform)
