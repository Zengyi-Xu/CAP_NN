# -*- coding: utf-8 -*-
"""Parameter grid scan for the DMT-NN experiment platform.

Automates 2-D sweeps over a Keithley-controlled bias (voltage or current) and
the AWG output amplitude (Vpp). For each grid point the DMT pipeline is run,
Step 2 / Step 4 can be repeated and averaged, and the results are saved as a
CSV summary plus CodePlot-compatible contour plots.

Data layout for a scan:

    data/grid_scans/<scan_id>/
        config.json          scan parameters
        summary.csv          one row per grid point (averaged metrics)
        contour_*.png        rendered contour plots
        codeplot_assets/
            data/            npz files for contour plots
            scripts/         CodePlot v5 loadable .py scripts

Individual DMT runs are still recorded under their own run_ids in
`data/records/` and `data/codeplot_assets/<point_run_id>/`, so the original
waveforms are preserved but are not shown in the grid-scan GUI by default.
"""
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.interpolate import griddata

import config
from keithley2400_controller import Keithley2400
from record import generate_run_id


GRID_SCAN_DIR = config.DATA_DIR / "grid_scans"
GRID_SCAN_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def _setup_scan_logger(scan_id: str) -> logging.Logger:
    logger = logging.getLogger(f"grid_scan.{scan_id}")
    logger.setLevel(logging.DEBUG)
    if logger.handlers:
        return logger

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    log_dir = GRID_SCAN_DIR / scan_id
    log_dir.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(log_dir / "grid_scan.log", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class GridScanConfig:
    """Parameters for a 2-D grid scan."""

    scan_id: str = ""
    # Parameter 1: Keithley-controlled bias
    param1_name: str = "bias_voltage"   # displayed name in CSV / plots
    param1_mode: str = "voltage"        # "voltage" or "current"
    param1_start: float = 0.0
    param1_stop: float = 1.0
    param1_step: float = 0.1
    # Parameter 2: AWG Vpp
    vpp_start: float = 0.1
    vpp_stop: float = 1.0
    vpp_step: float = 0.1
    # Pipeline options
    run_mode: str = "step1-4"           # "step1-4" or "step1-2"
    step4_repeats: int = 1              # repeats of the final measurement step
    use_awg: bool = False
    use_nn: bool = False
    use_virtual_channel: bool = False
    offline: bool = True
    # Keithley connection
    keithley_port: str = "COM1"
    keithley_baudrate: int = 9600
    keithley_timeout: float = 5.0
    keithley_compliance: float = 0.1
    keithley_nplc: float = 1.0

    def __post_init__(self):
        if not self.scan_id:
            self.scan_id = generate_run_id()
        if self.param1_mode not in ("voltage", "current"):
            raise ValueError("param1_mode must be 'voltage' or 'current'")
        if self.run_mode not in ("step1-4", "step1-2"):
            raise ValueError("run_mode must be 'step1-4' or 'step1-2'")
        if self.step4_repeats < 1:
            raise ValueError("step4_repeats must be >= 1")


# ---------------------------------------------------------------------------
# Grid utilities
# ---------------------------------------------------------------------------
def _make_grid(start: float, stop: float, step: float) -> np.ndarray:
    """Return a 1-D grid inclusive of stop when step divides exactly."""
    n = int(np.round((stop - start) / step)) + 1
    vals = start + np.arange(n) * step
    # Clamp the last point to stop to avoid FP drift
    if len(vals) > 0:
        vals[-1] = min(vals[-1], stop)
    return vals


def _safe_log10(x: float) -> float:
    return float(10 * np.log10(x)) if x > 0 else np.nan


# ---------------------------------------------------------------------------
# Contour plot helpers
# ---------------------------------------------------------------------------
_CONTOUR_TEMPLATE = """import numpy as np
from pathlib import Path
from scipy.interpolate import griddata

data = np.load(Path(__file__).parent.parent / "data" / "$data_name")
bias = data["bias"]
vpp = data["vpp"]
z = data["z"]

xi = np.linspace(bias.min(), bias.max(), 100)
yi = np.linspace(vpp.min(), vpp.max(), 100)
Xi, Yi = np.meshgrid(xi, yi)
Zi = griddata((bias, vpp), z, (Xi, Yi), method="cubic")

ax = fig.add_subplot(111)
levels = np.linspace(np.nanmin(Zi), np.nanmax(Zi), 20) if np.any(np.isfinite(Zi)) else 10
im = ax.contourf(Xi, Yi, Zi, levels=levels, cmap="viridis", extend="both")
ax.set_xlabel("$xlabel")
ax.set_ylabel("$ylabel")
ax.set_title("$title")
fig.colorbar(im, ax=ax, label="$zlabel")
fig.tight_layout()
"""


def _save_contour_plot(scan_dir: Path, scan_id: str, name: str,
                       bias: np.ndarray, vpp: np.ndarray, z: np.ndarray,
                       title: str, xlabel: str, ylabel: str, zlabel: str,
                       use_log: bool = False):
    """Render a contour plot and write a CodePlot-compatible script."""
    import matplotlib.pyplot as plt

    data_dir = scan_dir / "codeplot_assets" / "data"
    script_dir = scan_dir / "codeplot_assets" / "scripts"
    data_dir.mkdir(parents=True, exist_ok=True)
    script_dir.mkdir(parents=True, exist_ok=True)

    data_name = f"{name}.npz"
    np.savez(data_dir / data_name, bias=bias, vpp=vpp, z=z)

    script_code = _CONTOUR_TEMPLATE.replace("$data_name", data_name)
    script_code = script_code.replace("$title", title)
    script_code = script_code.replace("$xlabel", xlabel)
    script_code = script_code.replace("$ylabel", ylabel)
    script_code = script_code.replace("$zlabel", zlabel)

    script_path = script_dir / f"{name}.py"
    script_path.write_text(script_code, encoding="utf-8")

    fig = plt.figure(dpi=config.PLOT_DPI)
    ns = {"fig": fig, "np": np, "plt": plt, "Path": Path, "__file__": str(script_path)}
    exec(script_code, ns)

    png_path = scan_dir / f"{name}.png"
    fig.savefig(png_path, dpi=config.PLOT_DPI, bbox_inches="tight")
    plt.close(fig)

    return png_path, script_path


def _generate_contour_plots(scan_dir: Path, cfg: GridScanConfig,
                            rows: List[Dict]) -> Dict[str, Path]:
    """Generate contour plots for BER, SER, and SNR if available."""
    bias = np.array([r["param1"] for r in rows])
    vpp = np.array([r["vpp"] for r in rows])
    out = {}

    def _plot(name: str, z: np.ndarray, title: str, zlabel: str, use_log: bool = False):
        if np.all(~np.isfinite(z)):
            return None
        png, script = _save_contour_plot(
            scan_dir, cfg.scan_id, name, bias, vpp, z,
            title=title,
            xlabel=cfg.param1_name,
            ylabel="AWG Vpp (V)",
            zlabel=zlabel,
            use_log=use_log,
        )
        return png

    ber = np.array([r.get("ber", np.nan) for r in rows])
    ser = np.array([r.get("ser", np.nan) for r in rows])
    snr = np.array([r.get("snr_db", np.nan) for r in rows])

    if cfg.run_mode == "step1-4":
        out["contour_ber"] = _plot("contour_ber", ber,
                                   f"BER Grid Scan ({cfg.scan_id})", "BER", use_log=True)
        out["contour_ser"] = _plot("contour_ser", ser,
                                   f"SER Grid Scan ({cfg.scan_id})", "SER", use_log=True)
    out["contour_snr"] = _plot("contour_snr", snr,
                               f"SNR Grid Scan ({cfg.scan_id})", "SNR (dB)")
    return out


# ---------------------------------------------------------------------------
# CSV summary
# ---------------------------------------------------------------------------
CSV_COLUMNS = [
    "point_idx", "param1", "param1_mode", "vpp", "run_id",
    "ber", "ser", "snr_db", "rate_gbps", "repeats",
]


def _save_summary_csv(scan_dir: Path, rows: List[Dict]):
    """Write the averaged grid scan results to summary.csv."""
    path = scan_dir / "summary.csv"
    header = ",".join(CSV_COLUMNS) + "\n"
    lines = [header]
    for r in rows:
        line = ",".join(str(r.get(c, "")) for c in CSV_COLUMNS) + "\n"
        lines.append(line)
    path.write_text("".join(lines), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Grid scanner
# ---------------------------------------------------------------------------
class GridScanner:
    """Run a 2-D parameter grid scan with optional Keithley bias control."""

    def __init__(self, cfg: GridScanConfig, logger: Optional[logging.Logger] = None):
        self.cfg = cfg
        self.scan_dir = GRID_SCAN_DIR / cfg.scan_id
        self.scan_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger or _setup_scan_logger(cfg.scan_id)
        self.keithley: Optional[Keithley2400] = None
        self.rows: List[Dict] = []
        self.original_vpp = config.AWG_VPP
        self._stop_flag = False

    def request_stop(self):
        """Signal the scan loop to stop after the current point."""
        self._stop_flag = True

    def _stop_requested(self) -> bool:
        return self._stop_flag

    # --- Keithley helpers ---------------------------------------------------
    def _connect_keithley(self):
        if not self.cfg.keithley_port:
            self.logger.info("No Keithley port configured; skipping source control")
            return
        self.keithley = Keithley2400(
            port=self.cfg.keithley_port,
            baudrate=self.cfg.keithley_baudrate,
            timeout=self.cfg.keithley_timeout,
            logger=self.logger,
        )
        self.keithley.connect()
        self.keithley.set_source_mode(self.cfg.param1_mode)
        self.keithley.set_compliance(self.cfg.keithley_compliance)
        self.keithley.set_nplc(self.cfg.keithley_nplc)
        self.keithley.set_range(auto=True)

    def _set_bias(self, value: float):
        if self.keithley is None:
            self.logger.info(f"[no Keithley] bias would be {value}")
            return
        self.keithley.set_output_level(value)
        self.keithley.output_on()
        self.logger.info(f"Keithley bias set to {value} ({self.cfg.param1_mode})")

    def _disconnect_keithley(self):
        if self.keithley is not None:
            try:
                self.keithley.disconnect()
            except Exception as exc:
                self.logger.warning(f"Keithley disconnect error: {exc}")
            finally:
                self.keithley = None

    # --- Pipeline helpers ---------------------------------------------------
    def _run_point_step1_2(self, point_run_id: str):
        """Run step1 + step2, repeat step2 N times, return averaged metrics."""
        # Lazy import so the GUI can open even if PyVISA is not installed
        from main import step1_generate_qpsk_tx, step2_receive_qpsk

        tx_qpsk = step1_generate_qpsk_tx(use_awg=self.cfg.use_awg, run_id=point_run_id)

        snrs_list = []
        for r in range(self.cfg.step4_repeats):
            repeat_run_id = f"{point_run_id}_r{r}"
            snrs, _ = step2_receive_qpsk(
                tx_qpsk,
                offline=self.cfg.offline,
                use_virtual_channel=self.cfg.use_virtual_channel,
                run_id=repeat_run_id,
            )
            snrs_list.append(snrs)

        avg_snr = float(np.nanmean(np.stack(snrs_list))) if snrs_list else np.nan
        return {
            "ber": np.nan,
            "ser": np.nan,
            "snr_db": _safe_log10(avg_snr) if avg_snr > 0 else np.nan,
            "rate_gbps": np.nan,
        }

    def _run_point_step1_4(self, point_run_id: str):
        """Run step1-4, repeat step4 N times, return averaged metrics."""
        # Lazy import so the GUI can open even if PyVISA is not installed
        from main import (
            step1_generate_qpsk_tx,
            step2_receive_qpsk,
            step3_generate_bitloading_tx,
            step4_receive_bitloading,
        )

        tx_qpsk = step1_generate_qpsk_tx(use_awg=self.cfg.use_awg, run_id=point_run_id)
        snrs, _ = step2_receive_qpsk(
            tx_qpsk,
            offline=self.cfg.offline,
            use_virtual_channel=self.cfg.use_virtual_channel,
            run_id=point_run_id,
        )
        tx_bpl = step3_generate_bitloading_tx(
            snrs,
            constellation=config.CONSTELLATION_QAM,
            use_awg=self.cfg.use_awg,
            run_id=point_run_id,
        )

        results = []
        for r in range(self.cfg.step4_repeats):
            repeat_run_id = f"{point_run_id}_r{r}"
            res = step4_receive_bitloading(
                tx_bpl,
                offline=self.cfg.offline,
                use_nn=self.cfg.use_nn,
                use_virtual_channel=self.cfg.use_virtual_channel,
                run_id=repeat_run_id,
            )
            results.append(res)

        ber = float(np.mean([r["ber"] for r in results]))
        ser = float(np.mean([r["ser"] for r in results]))
        mean_snr = float(np.mean([np.nanmean(r["SNR_R"]) for r in results]))
        rate_gbps = float(tx_bpl.get("datarate_gbps", 0.0))
        return {
            "ber": ber,
            "ser": ser,
            "snr_db": _safe_log10(mean_snr) if mean_snr > 0 else np.nan,
            "rate_gbps": rate_gbps,
        }

    def _run_point(self, param1: float, vpp: float, point_idx: int) -> Dict:
        """Run a single grid point and return averaged metrics."""
        point_run_id = f"{self.cfg.scan_id}_p{point_idx:04d}"
        self.logger.info(f"=== Point {point_idx}: {self.cfg.param1_name}={param1}, Vpp={vpp} ===")

        # Set hardware
        self._set_bias(param1)
        config.AWG_VPP = float(vpp)

        # Run pipeline
        try:
            if self.cfg.run_mode == "step1-2":
                metrics = self._run_point_step1_2(point_run_id)
            else:
                metrics = self._run_point_step1_4(point_run_id)
        except Exception as exc:
            self.logger.error(f"Point {point_idx} failed: {exc}")
            metrics = {"ber": np.nan, "ser": np.nan, "snr_db": np.nan, "rate_gbps": np.nan}

        row = {
            "point_idx": point_idx,
            "param1": param1,
            "param1_mode": self.cfg.param1_mode,
            "vpp": vpp,
            "run_id": point_run_id,
            "repeats": self.cfg.step4_repeats,
        }
        row.update(metrics)
        self.rows.append(row)
        self.logger.info(f"Point {point_idx} result: {metrics}")
        return row

    # --- Main entry ---------------------------------------------------------
    def run(self, progress_callback=None) -> Path:
        """Execute the full grid scan and return the summary CSV path."""
        self.logger.info(f"Starting grid scan {self.cfg.scan_id}")
        self.logger.info(f"Config: {asdict(self.cfg)}")

        # Save scan config
        (self.scan_dir / "config.json").write_text(
            json.dumps(asdict(self.cfg), indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )

        # Build grid
        p1_grid = _make_grid(self.cfg.param1_start, self.cfg.param1_stop, self.cfg.param1_step)
        vpp_grid = _make_grid(self.cfg.vpp_start, self.cfg.vpp_stop, self.cfg.vpp_step)
        total = len(p1_grid) * len(vpp_grid)
        self.logger.info(f"Grid size: {len(p1_grid)} x {len(vpp_grid)} = {total} points")

        # Temporarily suppress interactive plots during scan, but save assets
        original_plot_show = config.PLOT_SHOW
        original_plot_save = config.PLOT_SAVE
        original_offline_flag = config.OFFLINE_FLAG
        config.PLOT_SHOW = False
        config.PLOT_SAVE = True
        config.OFFLINE_FLAG = 0 if not self.cfg.offline else 1

        try:
            self._connect_keithley()
            point_idx = 0
            for p1 in p1_grid:
                for vpp in vpp_grid:
                    if self._stop_requested():
                        self.logger.info("Grid scan stopped by user")
                        break
                    self._run_point(p1, vpp, point_idx)
                    point_idx += 1
                    if progress_callback:
                        progress_callback(point_idx / total)
                if self._stop_requested():
                    break
        finally:
            self._disconnect_keithley()
            config.AWG_VPP = self.original_vpp
            config.PLOT_SHOW = original_plot_show
            config.PLOT_SAVE = original_plot_save
            config.OFFLINE_FLAG = original_offline_flag

        # Save summary and plots
        csv_path = _save_summary_csv(self.scan_dir, self.rows)
        self.logger.info(f"Saved summary CSV to {csv_path}")

        try:
            plot_paths = _generate_contour_plots(self.scan_dir, self.cfg, self.rows)
            self.logger.info(f"Generated contour plots: {plot_paths}")
        except Exception as exc:
            self.logger.error(f"Contour plot generation failed: {exc}")

        return csv_path


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------
def list_grid_scans() -> List[Dict]:
    """Return metadata for all saved grid scans."""
    scans = []
    if not GRID_SCAN_DIR.is_dir():
        return scans
    for p in sorted(GRID_SCAN_DIR.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
        if p.is_dir():
            cfg_path = p / "config.json"
            csv_path = p / "summary.csv"
            info = {"scan_id": p.name, "path": str(p)}
            if cfg_path.exists():
                try:
                    info["config"] = json.loads(cfg_path.read_text(encoding="utf-8"))
                except Exception:
                    pass
            if csv_path.exists():
                try:
                    lines = csv_path.read_text(encoding="utf-8").strip().splitlines()
                    info["points"] = len(lines) - 1
                except Exception:
                    pass
            scans.append(info)
    return scans


def load_summary(scan_id: str) -> Tuple[Optional[List[str]], Optional[List[List[str]]]]:
    """Load CSV summary as (header, rows)."""
    csv_path = GRID_SCAN_DIR / scan_id / "summary.csv"
    if not csv_path.exists():
        return None, None
    lines = csv_path.read_text(encoding="utf-8").strip().splitlines()
    if not lines:
        return None, None
    return lines[0].split(","), [line.split(",") for line in lines[1:]]
