"""CodePlot v5 绘图适配器.

把每次测试的绘图数据和可编辑的 CodePlot v5 脚本保存下来，
同时在 Spyder 中直接显示图像。

每个完整测试会在 data/codeplot_assets/<run_id>/ 下生成：
    data/      : 每张图的 *.npz 数据
    scripts/   : 每张图的 *.py 脚本（可用 codeplot_v5.py 打开编辑）
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from string import Template

import config


CODEPLOT_DIR = config.DATA_DIR / "codeplot_assets"
CODEPLOT_DIR.mkdir(parents=True, exist_ok=True)


def _prepare_dirs(run_id: str):
    """返回本次测试的 base、data、scripts 目录."""
    base = CODEPLOT_DIR / run_id
    data_dir = base / "data"
    script_dir = base / "scripts"
    data_dir.mkdir(parents=True, exist_ok=True)
    script_dir.mkdir(parents=True, exist_ok=True)
    return base, data_dir, script_dir


def _save_data(data_dir: Path, name: str, **arrays):
    """把数组保存为 NPZ."""
    np.savez(data_dir / f"{name}.npz", **arrays)


def _write_script(script_dir: Path, name: str, code: str):
    """写入 CodePlot v5 可加载的脚本."""
    (script_dir / f"{name}.py").write_text(code, encoding="utf-8")


def _display(script_code: str, script_path: Path):
    """在 Spyder 中显示图像（脚本里使用外部 fig 变量）."""
    fig = plt.figure(dpi=config.PLOT_DPI)
    ns = {"fig": fig, "np": np, "plt": plt, "Path": Path,
          "__file__": str(script_path)}
    exec(script_code, ns)
    plt.show()


def _save_and_display(run_id: str, name: str, arrays: dict, script_template: Template,
                      script_vars: dict):
    """通用：保存数据、写脚本、显示."""
    base, data_dir, script_dir = _prepare_dirs(run_id)
    _save_data(data_dir, name, **arrays)
    script_vars = dict(script_vars)
    script_vars["data_name"] = f"{name}.npz"
    script_code = script_template.substitute(script_vars)
    script_path = script_dir / f"{name}.py"
    _write_script(script_dir, name, script_code)
    _display(script_code, script_path)
    return script_path


# ═══════════════════════════════════════════════════════════════════════════════
# 脚本模板
# ═══════════════════════════════════════════════════════════════════════════════

_TIME_TEMPLATE = Template("""import numpy as np
from pathlib import Path

data = np.load(Path(__file__).parent.parent / "data" / "$data_name")
t = data["t"]
sig = data["sig"]

ax = fig.add_subplot(111)
ax.plot(t, sig, "b.-", linewidth=1, markersize=2)
ax.set_title("$title")
ax.set_xlabel("Sample")
ax.set_ylabel("Amplitude")
ax.grid(True, alpha=0.3)
fig.tight_layout()
""")


_SPECTRUM_TEMPLATE = Template("""import numpy as np
from pathlib import Path

data = np.load(Path(__file__).parent.parent / "data" / "$data_name")
sig = data["sig"]
fs = float(data["fs"])

n = len(sig)
freqs = np.fft.fftshift(np.fft.fftfreq(n, d=1.0 / fs))
# 与 MATLAB 保持一致：10*log10(abs(fft(sig)))
spec = 10 * np.log10(np.abs(np.fft.fftshift(np.fft.fft(sig))) + 1e-12)

ax = fig.add_subplot(111)
ax.plot(freqs / 1e9, spec, "b-", linewidth=1)
ax.set_title("$title")
ax.set_xlabel("Frequency (GHz)")
ax.set_ylabel("Magnitude (dB)")
ax.grid(True, alpha=0.3)
fig.tight_layout()
""")


_CONSTELLATION_TEMPLATE = Template("""import numpy as np
from pathlib import Path

data = np.load(Path(__file__).parent.parent / "data" / "$data_name")
iq = data["iq"]

ax = fig.add_subplot(111)
ax.plot(iq.real, iq.imag, "b.", alpha=0.3, markersize=3)
ax.set_title("$title")
ax.set_xlabel("I")
ax.set_ylabel("Q")
ax.grid(True, alpha=0.3)
ax.axis("equal")
fig.tight_layout()
""")


_SNR_TEMPLATE = Template("""import numpy as np
from pathlib import Path

data = np.load(Path(__file__).parent.parent / "data" / "$data_name")
est = 10 * np.log10(np.maximum(data["est"].astype(float), 1e-12))
real = 10 * np.log10(np.maximum(data["real"].astype(float), 1e-12))

ax = fig.add_subplot(111)
ax.plot(est, "b", label="Est-SNR", marker="o", markersize=3, linewidth=1)
ax.plot(real, "r", label="TestReal-SNR", marker="x", markersize=3, linewidth=1)
ax.set_title("$title")
ax.set_xlabel("Subcarrier")
ax.set_ylabel("SNR (dB)")
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
""")


_NONLINEARITY_TEMPLATE = Template("""import numpy as np
from pathlib import Path

data = np.load(Path(__file__).parent.parent / "data" / "$data_name")
tx = data["tx"]
rx = data["rx"]

ax = fig.add_subplot(111)
if len(tx) > 5000:
    hb = ax.hexbin(tx, rx, gridsize=80, cmap="GnBu", mincnt=1)
    fig.colorbar(hb, ax=ax, label="Density")
else:
    ax.plot(tx, rx, "b.", alpha=0.2, markersize=3)

if np.any(tx):
    gain = np.sum(tx * rx) / np.sum(tx ** 2)
    t = np.linspace(tx.min(), tx.max(), 100)
    ax.plot(t, gain * t, "g--", linewidth=2, label=f"Linear fit (gain={gain:.3f})")

ax.set_title("$title")
ax.set_xlabel("TX Amplitude")
ax.set_ylabel("RX Amplitude")
ax.legend()
ax.grid(True, alpha=0.3)
ax.axis("equal")
fig.tight_layout()
""")


_BITPOWER_TEMPLATE = Template("""import numpy as np
from pathlib import Path

data = np.load(Path(__file__).parent.parent / "data" / "$data_name")
subcarriers = data["subcarriers"]
snrs_db = data["snrs_db"]
RQ = data["RQ"]
S = data["S"]
ratio = int(data["ratio"])
rate_gbps = float(data["rate_gbps"])

ax1 = fig.add_subplot(211)
ax1_bits = ax1.twinx()
ax1.plot(subcarriers, snrs_db, "b-", linewidth=1.5, label="SNR (dB)")
ax1_bits.plot(subcarriers, RQ, "r-", linewidth=1.5, marker="x", markersize=3, label="Bit allocation")
ax1.set_ylabel("SNR (dB)", color="b")
ax1_bits.set_ylabel("Bits / symbol", color="r")
ax1.set_title(f"Bit-Power Loading (ratio={ratio}, rate={rate_gbps:.2f} Gbps)")
ax1.grid(True, alpha=0.3)
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax1_bits.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

ax2 = fig.add_subplot(212, sharex=ax1)
ax2.plot(subcarriers, S, "g-", linewidth=1.5, marker="o", markersize=2, label="Power allocation")
ax2.set_xlabel("Subcarrier")
ax2.set_ylabel("Power scaling")
ax2.legend()
ax2.grid(True, alpha=0.3)

fig.tight_layout()
""")


_SERBER_TEMPLATE = Template("""import numpy as np
from pathlib import Path

data = np.load(Path(__file__).parent.parent / "data" / "$data_name")
ser = data["ser"]
ber = data["ber"]
RQ = data["RQ"] if "RQ" in data else None
carrier_idx = np.arange(len(ser))

ax1 = fig.add_subplot(211)
ax1.plot(carrier_idx, ser, "r-", marker="o", markersize=3, linewidth=1)
ax1.set_ylabel("SER")
ax1.set_title("Symbol Error Rate per Subcarrier")
ax1.grid(True, alpha=0.3)

ax2 = fig.add_subplot(212)
ax2.semilogy(carrier_idx, np.where(ber > 0, ber, 1e-12), "b-", marker="x", markersize=3, linewidth=1)
ax2.set_xlabel("Subcarrier Index")
ax2.set_ylabel("BER")
ax2.set_title("Bit Error Rate per Subcarrier")
ax2.grid(True, which="both", ls="--", alpha=0.3)

if RQ is not None:
    ax2_twin = ax2.twinx()
    ax2_twin.plot(carrier_idx, RQ, "g--", alpha=0.5, label="Bit allocation")
    ax2_twin.set_ylabel("Bits / symbol", color="g")
    ax2_twin.legend(loc="upper right")

fig.tight_layout()
""")


_CONST_DENSITY_TEMPLATE = Template("""import numpy as np
from pathlib import Path

data = np.load(Path(__file__).parent.parent / "data" / "$data_name")
out2 = data["out2"]
in_ref = data["in_ref"]
RQ = data["RQ"]
pilot_mask = data["pilot_mask"]

orders = sorted({int(b) for b in RQ if b > 0})
ncols = min(3, len(orders))
nrows = int(np.ceil(len(orders) / ncols))

for idx, bits in enumerate(orders):
    ax = fig.add_subplot(nrows, ncols, idx + 1)
    carriers = np.where(RQ == bits)[0]
    pts = []
    ideal = []
    for n in carriers:
        valid = ~pilot_mask[n, :]
        if not np.any(valid):
            continue
        pts.append(out2[n, valid])
        ideal.append(in_ref[n, valid])
    if not pts:
        ax.set_visible(False)
        continue
    pts = np.concatenate(pts)
    ideal = np.concatenate(ideal)

    gridsize = max(30, 2 * int(2 ** (bits / 2)))
    hb = ax.hexbin(pts.real, pts.imag, gridsize=gridsize, cmap="GnBu", mincnt=1)
    fig.colorbar(hb, ax=ax, label="Density")

    ax.set_title(f"{2**bits}-QAM (bits={bits})")
    ax.set_xlabel("I")
    ax.set_ylabel("Q")
    ax.axis("equal")
    ax.grid(True, alpha=0.3)

fig.suptitle("Constellation Density by Modulation Order", y=1.02)
fig.tight_layout()
""")


_CONST_BY_ORDER_TEMPLATE = Template("""import numpy as np
from pathlib import Path

data = np.load(Path(__file__).parent.parent / "data" / "$data_name")
out2 = data["out2"]
RQ = data["RQ"]
pilot_mask = data["pilot_mask"]

orders = sorted({int(b) for b in RQ if b > 0})
ncols = min(3, len(orders))
nrows = int(np.ceil(len(orders) / ncols))

for idx, bits in enumerate(orders):
    ax = fig.add_subplot(nrows, ncols, idx + 1)
    carriers = np.where(RQ == bits)[0]
    pts = []
    for n in carriers:
        valid = ~pilot_mask[n, :]
        if not np.any(valid):
            continue
        pts.append(out2[n, valid])
    if not pts:
        ax.set_visible(False)
        continue
    pts = np.concatenate(pts)

    ax.plot(pts.real, pts.imag, "b.", alpha=0.3, markersize=3)

    ax.set_title(f"{2**bits}-QAM (bits={bits})")
    ax.set_xlabel("I")
    ax.set_ylabel("Q")
    ax.axis("equal")
    ax.grid(True, alpha=0.3)

fig.suptitle("RX Constellation by Modulation Order", y=1.02)
fig.tight_layout()
""")


# ═══════════════════════════════════════════════════════════════════════════════
# 对外接口
# ═══════════════════════════════════════════════════════════════════════════════

def plot_time_waveform(t, sig, title: str, run_id: str, name: str):
    _save_and_display(
        run_id, name,
        arrays={"t": np.asarray(t), "sig": np.asarray(sig)},
        script_template=_TIME_TEMPLATE,
        script_vars={"title": title}
    )


def plot_spectrum(sig, fs: float, title: str, run_id: str, name: str):
    _save_and_display(
        run_id, name,
        arrays={"sig": np.asarray(sig), "fs": float(fs)},
        script_template=_SPECTRUM_TEMPLATE,
        script_vars={"title": title}
    )


def plot_constellation(iq, title: str, run_id: str, name: str):
    _save_and_display(
        run_id, name,
        arrays={"iq": np.asarray(iq)},
        script_template=_CONSTELLATION_TEMPLATE,
        script_vars={"title": title}
    )


def plot_snrs(est, real, title: str, run_id: str, name: str):
    _save_and_display(
        run_id, name,
        arrays={"est": np.asarray(est), "real": np.asarray(real)},
        script_template=_SNR_TEMPLATE,
        script_vars={"title": title}
    )


def plot_tx_rx_nonlinearity(tx, rx, title: str, run_id: str, name: str):
    _save_and_display(
        run_id, name,
        arrays={"tx": np.asarray(tx), "rx": np.asarray(rx)},
        script_template=_NONLINEARITY_TEMPLATE,
        script_vars={"title": title}
    )


def plot_bit_power_loading(subcarriers, snrs_db, RQ, S, ratio, rate_gbps,
                           title: str, run_id: str, name: str):
    _save_and_display(
        run_id, name,
        arrays={
            "subcarriers": np.asarray(subcarriers),
            "snrs_db": np.asarray(snrs_db),
            "RQ": np.asarray(RQ),
            "S": np.asarray(S),
            "ratio": int(ratio),
            "rate_gbps": float(rate_gbps),
        },
        script_template=_BITPOWER_TEMPLATE,
        script_vars={"title": title}
    )


def plot_ser_ber_per_carrier(ser, ber, RQ, title: str, run_id: str, name: str):
    arrays = {"ser": np.asarray(ser), "ber": np.asarray(ber)}
    if RQ is not None:
        arrays["RQ"] = np.asarray(RQ)
    _save_and_display(
        run_id, name,
        arrays=arrays,
        script_template=_SERBER_TEMPLATE,
        script_vars={"title": title}
    )


def plot_constellation_density(out2, in_ref, RQ, pilot_mask, title: str,
                               run_id: str, name: str):
    _save_and_display(
        run_id, name,
        arrays={
            "out2": np.asarray(out2),
            "in_ref": np.asarray(in_ref),
            "RQ": np.asarray(RQ),
            "pilot_mask": np.asarray(pilot_mask),
        },
        script_template=_CONST_DENSITY_TEMPLATE,
        script_vars={"title": title}
    )


def plot_constellation_by_order(out2, in_ref, RQ, pilot_mask, title: str,
                                run_id: str, name: str):
    _save_and_display(
        run_id, name,
        arrays={
            "out2": np.asarray(out2),
            "in_ref": np.asarray(in_ref),
            "RQ": np.asarray(RQ),
            "pilot_mask": np.asarray(pilot_mask),
        },
        script_template=_CONST_BY_ORDER_TEMPLATE,
        script_vars={"title": title}
    )
