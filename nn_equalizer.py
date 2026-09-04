"""NN post-equalizer wrapper.

Directly calls the existing ZY_BiGRU_GPU.py:
1. Write Tx/Rx waveforms to the filenames expected by ZY_BiGRU_GPU.py in the NN directory
2. Run ZY_BiGRU_GPU.py as a subprocess in the NN directory
3. Read the generated Rxdata_afterNN1.txt and return it
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
    """ZY_BiGRU_GPU wrapper."""

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
        """Run NN equalization.

        Args:
            tx_waveform: transmitted reference waveform
            rx_waveform: received waveform
            output_name: NN output filename
            epochs: number of training epochs (if provided, temporarily modify the epochs in the script)
            use_pretrained: whether to prefer the already-trained trained_model_temp.pth

        Returns:
            NN-equalized waveform (1-D numpy array)
        """
        if not self.script.exists():
            raise FileNotFoundError(f"NN script not found: {self.script}")

        # Normalize and write to NN directory (consistent with original script)
        tx_norm = tx_waveform / np.max(np.abs(tx_waveform))
        rx_norm = rx_waveform / np.max(np.abs(rx_waveform))

        save_txt(self.nn_dir / "Txdata_NN.txt", tx_norm)
        save_txt(self.nn_dir / "Rxdata_NN1.txt", rx_norm)

        # Build environment variables
        env = os.environ.copy()
        env["MPLBACKEND"] = "Agg"  # Avoid plt.show() blocking
        env["DISABLE_TQDM"] = "1"  # Disable NN training progress bar

        # Run NN script in a subprocess (stream output in real time for GUI progress display)
        cmd = [self.python_exe, str(self.script)]
        print(f"Running NN equalizer: {' '.join(cmd)}")
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
            raise RuntimeError(f"NN script failed with return code {code}")

        output_file = self.nn_dir / output_name
        if not output_file.exists():
            raise FileNotFoundError(f"NN output not found: {output_file}")

        return load_txt(output_file)


def run_nn_equalizer(tx_waveform: np.ndarray,
                     rx_waveform: np.ndarray,
                     nn_dir: Path = config.NN_DIR) -> np.ndarray:
    """Convenience function."""
    eq = NNEqualizer(nn_dir=nn_dir)
    return eq.run(tx_waveform, rx_waveform)
