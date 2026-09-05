# -*- coding: utf-8 -*-
"""UW_APSK_CAP_PY 实验平台的参数网格扫描。

在 Keithley 控制的偏置（电压或电流）与第二个 CAP 收发机参数（虚拟信道信噪比
或 AWG 输出峰峰值 Vpp）两个维度上自动进行二维扫描。每个网格点运行一次 CAP
收发机，测量可重复进行并取平均，结果保存为 CSV 汇总表以及等高线图。

一次扫描的数据目录结构：

    data/grid_scans/<scan_id>/
        config.json          扫描参数
        summary.csv          每个网格点一行（平均后的指标）
        contour_*.png        渲染出的等高线图
        codeplot_assets/
            data/            等高线图所用的 npz 文件
            scripts/         可被 CodePlot v5 加载的 .py 脚本

各次单独的 CAP 运行仍以其自身的 run_id 记录在
`data/records/` 中，因此原始波形得以保留。
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

import config_cap as config
from keithley2400_controller import Keithley2400
from record import generate_run_id


GRID_SCAN_DIR = config.GRID_SCAN_DIR
GRID_SCAN_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# 日志
# ---------------------------------------------------------------------------
def _setup_scan_logger(scan_id: str) -> logging.Logger:
    logger = logging.getLogger(f"grid_scan_cap.{scan_id}")
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
# 配置
# ---------------------------------------------------------------------------
@dataclass
class GridScanConfig:
    """CAP 二维网格扫描的参数。"""

    scan_id: str = ""
    # 参数 1：Keithley 控制的偏置
    param1_name: str = "bias_voltage"   # 在 CSV / 图中显示的名称
    param1_mode: str = "voltage"        # 取 "voltage" 或 "current"
    param1_start: float = 0.0
    param1_stop: float = 1.0
    param1_step: float = 0.1
    # 参数 2：CAP 第二参数
    param2_name: str = "snr_db"         # 取 "snr_db" 或 "vpp"
    param2_start: float = 20.0
    param2_stop: float = 35.0
    param2_step: float = 1.0
    # 流水线选项
    run_mode: str = "singleband"        # 取 "singleband" 或 "multiband"
    step_repeats: int = 1               # 测量步骤的重复次数
    order: int = 64
    constellation: str = "QAM"
    use_nn: bool = False
    use_virtual_channel: bool = True
    # Keithley 连接
    keithley_port: str = "COM1"
    keithley_baudrate: int = 9600
    keithley_timeout: float = 5.0
    keithley_compliance: float = 0.1
    keithley_nplc: float = 1.0
    keithley_interface: str = "rs232"   # "rs232" 或 "gpib"

    def __post_init__(self):
        if not self.scan_id:
            self.scan_id = generate_run_id()
        if self.param1_mode not in ("voltage", "current"):
            raise ValueError("param1_mode 必须为 'voltage' 或 'current'")
        if self.run_mode not in ("singleband", "multiband"):
            raise ValueError("run_mode 必须为 'singleband' 或 'multiband'")
        if self.param2_name not in ("snr_db", "vpp"):
            raise ValueError("param2_name 必须为 'snr_db' 或 'vpp'")
        if self.step_repeats < 1:
            raise ValueError("step_repeats 必须 >= 1")
        if self.keithley_interface.lower() not in ("rs232", "serial", "usb", "gpib",
                                                   "gpio", "ieee488"):
            raise ValueError("keithley_interface 必须为 'rs232' 或 'gpib'")


# ---------------------------------------------------------------------------
# 网格工具
# ---------------------------------------------------------------------------
def _make_grid(start: float, stop: float, step: float) -> np.ndarray:
    """返回一维网格；当步长恰好整除时包含终点 stop。"""
    n = int(np.round((stop - start) / step)) + 1
    vals = start + np.arange(n) * step
    if len(vals) > 0:
        vals[-1] = min(vals[-1], stop)
    return vals


def _safe_log10(x: float) -> float:
    return float(10 * np.log10(x)) if x > 0 else float("nan")


# ---------------------------------------------------------------------------
# 等高线图辅助
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


def _save_contour_plot(scan_dir: Path, cfg: GridScanConfig, name: str,
                       bias: np.ndarray, vpp: np.ndarray, z: np.ndarray,
                       title: str, xlabel: str, ylabel: str, zlabel: str,
                       use_log: bool = False):
    """渲染等高线图，并写出兼容 CodePlot 的脚本。"""
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
    """在数据可用时生成误码率和符号误码率的等高线图。"""
    bias = np.array([r["param1"] for r in rows])
    vpp = np.array([r["vpp"] for r in rows])
    out = {}

    def _plot(name: str, z: np.ndarray, title: str, zlabel: str, use_log: bool = False):
        if np.all(~np.isfinite(z)):
            return None
        ylabel = "虚拟信道信噪比 (dB)" if cfg.param2_name == "snr_db" else "AWG 输出幅度 Vpp (V)"
        png, script = _save_contour_plot(
            scan_dir, cfg, name, bias, vpp, z,
            title=title,
            xlabel=cfg.param1_name,
            ylabel=ylabel,
            zlabel=zlabel,
            use_log=use_log,
        )
        return png

    ber = np.array([r.get("ber", np.nan) for r in rows])
    ser = np.array([r.get("ser", np.nan) for r in rows])

    out["contour_ber"] = _plot("contour_ber", ber,
                               f"误码率网格扫描 ({cfg.scan_id})", "误码率", use_log=True)
    out["contour_ser"] = _plot("contour_ser", ser,
                               f"符号误码率网格扫描 ({cfg.scan_id})", "符号误码率", use_log=True)
    return out


# ---------------------------------------------------------------------------
# CSV 汇总
# ---------------------------------------------------------------------------
CSV_COLUMNS = [
    "point_idx", "param1", "param1_mode", "vpp", "run_id",
    "ber", "ser", "snr_db", "rate_gbps", "repeats",
]


def _save_summary_csv(scan_dir: Path, rows: List[Dict]):
    """将平均后的网格扫描结果写入 summary.csv。"""
    path = scan_dir / "summary.csv"
    header = ",".join(CSV_COLUMNS) + "\n"
    lines = [header]
    for r in rows:
        line = ",".join(str(r.get(c, "")) for c in CSV_COLUMNS) + "\n"
        lines.append(line)
    path.write_text("".join(lines), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# 网格扫描器
# ---------------------------------------------------------------------------
class GridScanner:
    """运行二维参数网格扫描，可选 Keithley 偏置控制。"""

    def __init__(self, cfg: GridScanConfig, logger: Optional[logging.Logger] = None):
        self.cfg = cfg
        self.scan_dir = GRID_SCAN_DIR / cfg.scan_id
        self.scan_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger or _setup_scan_logger(cfg.scan_id)
        self.keithley: Optional[Keithley2400] = None
        self.rows: List[Dict] = []
        self._stop_flag = False

    def request_stop(self):
        """通知扫描循环在当前点结束后停止。"""
        self._stop_flag = True

    def _stop_requested(self) -> bool:
        return self._stop_flag

    # --- Keithley helpers ---------------------------------------------------
    def _connect_keithley(self):
        if not self.cfg.keithley_port:
            self.logger.info("未配置 Keithley 端口，跳过源控制")
            return
        self.keithley = Keithley2400(
            port=self.cfg.keithley_port,
            baudrate=self.cfg.keithley_baudrate,
            timeout=self.cfg.keithley_timeout,
            logger=self.logger,
            interface=self.cfg.keithley_interface,
        )
        self.keithley.connect()
        self.keithley.set_source_mode(self.cfg.param1_mode)
        self.keithley.set_compliance(self.cfg.keithley_compliance)
        self.keithley.set_nplc(self.cfg.keithley_nplc)
        self.keithley.set_range(auto=True)

    def _set_bias(self, value: float):
        if self.keithley is None:
            self.logger.info(f"[无 Keithley] 偏置将设为 {value}")
            return
        self.keithley.set_output_level(value)
        self.keithley.output_on()
        self.logger.info(f"Keithley 偏置已设置为 {value} ({self.cfg.param1_mode})")

    def _disconnect_keithley(self):
        if self.keithley is not None:
            try:
                self.keithley.disconnect()
            except Exception as exc:
                self.logger.warning(f"Keithley 断开连接出错：{exc}")
            finally:
                self.keithley = None

    # --- Pipeline helpers ---------------------------------------------------
    def _run_point_singleband(self, point_run_id: str, snr_db: float) -> Dict:
        """运行单带 CAP 并返回平均指标。"""
        # 延迟导入，以便未安装 torch 时 GUI 仍能打开
        import main_cap

        results = []
        for r in range(self.cfg.step_repeats):
            record = main_cap.run_singleband(
                numofsymbols=config.SB_NUMOFSYMBOLS,
                order=self.cfg.order,
                constellation=self.cfg.constellation,
                snr_db=snr_db,
                seed=100 + r,
            )
            results.append(record)

        ber = float(np.mean([r["ber"] for r in results]))
        ser = float(np.mean([r["ser"] for r in results]))
        return {
            "ber": ber,
            "ser": ser,
            "snr_db": snr_db,
            "rate_gbps": np.nan,
        }

    def _run_point_multiband(self, point_run_id: str, snr_db: float) -> Dict:
        """运行多带 CAP 并返回平均指标。"""
        import main_cap

        results = []
        for r in range(self.cfg.step_repeats):
            record = main_cap.run_multiband(
                numofsymbols=config.MB_NUMOFSYMBOLS,
                order=self.cfg.order,
                constellation=self.cfg.constellation,
                snr_db=snr_db,
                seed=1 + r,
                use_lms=True,
                use_nn=self.cfg.use_nn,
            )
            results.append(record)

        # BER 优先取 NN > LMS > 原始平均
        def _pick_ber(r):
            return r.get("nn_ber_avg") or r.get("eq_ber_avg") or r.get("raw_ber_avg", np.nan)

        def _pick_ser(r):
            return r.get("nn_ser") or r.get("eq_ser") or r.get("raw_ser") or [np.nan]

        ber = float(np.mean([_pick_ber(r) for r in results]))
        ser = float(np.mean([np.mean(_pick_ser(r)) for r in results]))
        return {
            "ber": ber,
            "ser": ser,
            "snr_db": snr_db,
            "rate_gbps": np.nan,
        }

    def _run_point(self, param1: float, param2: float, point_idx: int) -> Dict:
        """运行单个网格点并返回平均指标。"""
        point_run_id = f"{self.cfg.scan_id}_p{point_idx:04d}"
        param2_label = "SNR" if self.cfg.param2_name == "snr_db" else "Vpp"
        self.logger.info(
            f"=== 点 {point_idx}：{self.cfg.param1_name}={param1}，{param2_label}={param2} ==="
        )

        # 设置硬件 / 仿真参数
        self._set_bias(param1)
        if self.cfg.param2_name == "snr_db":
            config.SB_SNR_DB = float(param2)
            config.MB_SNR_DB = float(param2)
            snr_db = param2
        else:
            config.AWG_VPP = float(param2)
            snr_db = config.SB_SNR_DB

        # 保存原始信道标志，在扫描信噪比时强制使用虚拟信道
        original_use_vc = config.USE_VIRTUAL_CHANNEL
        if self.cfg.param2_name == "snr_db":
            config.USE_VIRTUAL_CHANNEL = 1 if self.cfg.use_virtual_channel else 0

        # 运行流水线
        try:
            if self.cfg.run_mode == "singleband":
                metrics = self._run_point_singleband(point_run_id, snr_db)
            else:
                metrics = self._run_point_multiband(point_run_id, snr_db)
        except Exception as exc:
            self.logger.error(f"点 {point_idx} 失败：{exc}")
            metrics = {"ber": np.nan, "ser": np.nan, "snr_db": param2, "rate_gbps": np.nan}
        finally:
            config.USE_VIRTUAL_CHANNEL = original_use_vc

        row = {
            "point_idx": point_idx,
            "param1": param1,
            "param1_mode": self.cfg.param1_mode,
            "vpp": param2,
            "run_id": point_run_id,
            "repeats": self.cfg.step_repeats,
        }
        row.update(metrics)
        self.rows.append(row)
        self.logger.info(f"点 {point_idx} 结果：{metrics}")
        return row

    # --- Main entry ---------------------------------------------------------
    def run(self, progress_callback=None) -> Path:
        """执行完整的网格扫描并返回汇总 CSV 路径。"""
        self.logger.info(f"开始 CAP 网格扫描 {self.cfg.scan_id}")
        self.logger.info(f"配置：{asdict(self.cfg)}")

        # 保存扫描配置
        (self.scan_dir / "config.json").write_text(
            json.dumps(asdict(self.cfg), indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )

        # 构建网格
        p1_grid = _make_grid(self.cfg.param1_start, self.cfg.param1_stop, self.cfg.param1_step)
        p2_grid = _make_grid(self.cfg.param2_start, self.cfg.param2_stop, self.cfg.param2_step)
        total = len(p1_grid) * len(p2_grid)
        self.logger.info(f"网格规模：{len(p1_grid)} x {len(p2_grid)} = 共 {total} 个点")

        # 扫描期间临时关闭交互式绘图，但仍保存图片资源
        original_plot_show = config.PLOT_SHOW
        original_plot_save = config.PLOT_SAVE
        config.PLOT_SHOW = False
        config.PLOT_SAVE = True

        try:
            self._connect_keithley()
            point_idx = 0
            for p1 in p1_grid:
                for p2 in p2_grid:
                    if self._stop_requested():
                        self.logger.info("网格扫描已被用户停止")
                        break
                    self._run_point(p1, p2, point_idx)
                    point_idx += 1
                    if progress_callback:
                        progress_callback(point_idx / total)
                if self._stop_requested():
                    break
        finally:
            self._disconnect_keithley()
            config.PLOT_SHOW = original_plot_show
            config.PLOT_SAVE = original_plot_save

        # 保存汇总结果和图
        csv_path = _save_summary_csv(self.scan_dir, self.rows)
        self.logger.info(f"汇总 CSV 已保存至 {csv_path}")

        try:
            plot_paths = _generate_contour_plots(self.scan_dir, self.cfg, self.rows)
            self.logger.info(f"已生成等高线图：{plot_paths}")
        except Exception as exc:
            self.logger.error(f"等高线图生成失败：{exc}")

        return csv_path


# ---------------------------------------------------------------------------
# 公开辅助函数
# ---------------------------------------------------------------------------
def list_grid_scans() -> List[Dict]:
    """返回所有已保存网格扫描的元数据。"""
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
    """加载 CSV 汇总，返回 (表头, 数据行)。"""
    csv_path = GRID_SCAN_DIR / scan_id / "summary.csv"
    if not csv_path.exists():
        return None, None
    lines = csv_path.read_text(encoding="utf-8").strip().splitlines()
    if not lines:
        return None, None
    return lines[0].split(","), [line.split(",") for line in lines[1:]]
