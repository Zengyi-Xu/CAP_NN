"""NN post-equalizer wrapper for multi-band CAP.

Calls data/nn/CAP_multiband_NN.py in a subprocess and reads the output symbols.
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
    """Wrapper for CAP_multiband_NN.py."""

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
        """Run NN equalization.

        Parameters
        ----------
        tx_symbols : np.ndarray
            Transmitted symbols, shape (N, 6) with real/imag interleaved per band.
        rx_symbols : np.ndarray
            Received (matched-filter) symbols, shape (N, 6).
        output_name : str
            Output filename to read predictions from.

        Returns
        -------
        np.ndarray
            NN-equalised symbols, shape (M, 6).
        """
        if not self.script.exists():
            raise FileNotFoundError(f"NN script not found: {self.script}")

        save_txt(self.nn_dir / "Txdata_NN.txt", tx_symbols)
        save_txt(self.nn_dir / "Rxdata_NN1.txt", rx_symbols)

        env = os.environ.copy()
        env["MPLBACKEND"] = "Agg"
        env["DISABLE_TQDM"] = "1"

        cmd = [self.python_exe, str(self.script)]
        print(f"Running CAP NN equalizer: {' '.join(cmd)}")
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
            raise RuntimeError(f"NN script failed with return code {code}")

        output_file = self.nn_dir / output_name
        if not output_file.exists():
            raise FileNotFoundError(f"NN output not found: {output_file}")
        return load_txt(output_file)


def run_cap_nn_equalizer(
    tx_symbols: np.ndarray,
    rx_symbols: np.ndarray,
    nn_dir: Path = cfg.NN_DIR,
) -> np.ndarray:
    """Convenience function."""
    eq = CAPNNEqualizer(nn_dir=nn_dir)
    return eq.run(tx_symbols, rx_symbols)
