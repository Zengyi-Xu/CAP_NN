# DMT 通信系统 Python 重写

本项目将原始 MATLAB DMT 通信代码重写为 Python，保留完整通信流程，并支持通过 pyvisa 直接控制 M8190A AWG 与 Keysight 示波器（USB-B）。

新增功能：
- 多种导频图案（training_only / comb / mesh）
- 内置虚拟信道（一阶低通、AWGN、三阶非线性、延迟）
- NN 后均衡（默认关闭，可手动启用）
- 自动绘图并在 Spyder 中显示
- 同时生成 CodePlot v5 可编辑脚本 + NPZ 数据，便于后续精修图
- 每次测试自动生成唯一编号并保存传输记录

## 项目结构

```
dmt_python/
├── config.py              # DMT / 硬件 / 绘图 / NN 参数
├── dmt_core.py            # DMT 调制、解调、bitloading、SNR/BER/SER 估计
├── awg_m8190a.py          # M8190A pyvisa 控制
├── oscilloscope.py        # 示波器 USB-B pyvisa 读取
├── nn_equalizer.py        # ZY_BiGRU_GPU 包装器
├── record.py              # 传输记录与唯一编号
├── plot_adapter.py        # 生成 CodePlot v5 脚本与 NPZ 数据
├── codeplot_v5.py         # CodePlot v5 图集排版工具
├── main.py                # 完整流程入口
├── utils.py               # 文件 I/O、同步、重采样、绘图
├── virtual_channel.py     # 虚拟信道仿真
├── requirements.txt
├── data/                  # 数据文件
│   ├── txdata/
│   ├── rxdata/
│   ├── codeplot_assets/   # 每次测试的 CodePlot 脚本与数据（按 run_id 分目录）
│   ├── records/           # 每次测试的 JSON/txt 记录
│   └── nn/                # NN 脚本与模型
└── README.md
```

## 安装依赖

建议使用虚拟环境：

```bash
python -m venv .venv
.venv\Scripts\activate.bat   # Windows cmd
# 或 .venv\Scripts\Activate.ps1  # Windows PowerShell
# 或 source .venv/Scripts/activate  # Git Bash

pip install numpy scipy matplotlib pyvisa PyVISA-py tqdm
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Windows 下如果 `torch` 安装报错 "File name extension too long"，请启用长路径支持（组策略 `Computer Configuration > Administrative Templates > System > Filesystem > Enable Win32 long paths`），或使用更短的项目路径。

若使用 Keysight VISA，建议同时安装 Keysight IO Libraries Suite，并确认 USB-B 驱动已安装；否则 `PyVISA-py` 已可驱动 USBTMC。

## 快速开始

### 1. 离线运行完整流程（不连接硬件，默认不调用 NN）

```bash
python main.py --offline 1 --use-awg 0
```

默认不调用 NN，以缩短运行时间。需要启用 NN 后均衡时：

```bash
python main.py --offline 1 --use-awg 0 --use-nn 1
```

离线模式默认读取 `data/rxdata/` 下已有的示波器文件；当 `--use-virtual-channel 1` 时直接用虚拟信道生成 RX。

#### 通过后缀定位离线波形

在线采集的波形现在会以 `rawOSC_QPSK_SNRest_<run_id>.txt` 和 `rawOSC_DMT_<run_id>.txt` 保存（`run_id` 形如 `20260806_013459_dee786`）。离线 rerun 时，只需输入最后 6 位即可自动定位同一组实验的两个阶段：

```bash
python main.py --offline 1 --use-awg 0 --run-suffix dee786
```

这等价于同时指定：

```bash
python main.py --offline 1 --use-awg 0 \
  --qpsk-rx data/rxdata/rawOSC_QPSK_SNRest_20260806_013459_dee786.txt \
  --bpl-rx  data/rxdata/rawOSC_DMT_20260806_013459_dee786.txt
```

如果只想重新处理某一阶段，也可以配合 `--step`：

```bash
python main.py --offline 1 --step step2 --run-suffix dee786
python main.py --offline 1 --step step4 --run-suffix dee786
```

> 提示：`--run-suffix` 与 `--qpsk-rx` / `--bpl-rx` 同时存在时，显式路径优先。

### 2. 在线运行（连接 M8190A + 示波器）

修改 `config.py` 中的 VISA 地址：

```python
M8190A_VISA_ADDR = "TCPIP0::192.168.1.10::5025::SOCKET"  # 改成 M8190A 实际 IP
OSC_VISA_ADDR = ""   # 留空自动查找第一个 USB 仪器
```

然后运行：

```bash
python main.py --offline 0 --use-awg 1
```

如需在线模式下启用 NN，再加 `--use-nn 1`。

### 3. 单步运行

```bash
python main.py --step step1  # 只生成 QPSK TX
python main.py --step step2  # 只接收 QPSK 并估计 SNR
python main.py --step step3  # 只生成 Bitloading TX
python main.py --step step4  # 只接收 Bitloading 并解调
```

### 4. 使用虚拟信道（无仪器调试）

在 `config.py` 中设置：

```python
USE_VIRTUAL_CHANNEL = 1
VIRTUAL_CHANNEL_FC = 2.0e9          # 一阶低通截止频率
VIRTUAL_CHANNEL_SNR_DB = 30         # 接收机 SNR
VIRTUAL_CHANNEL_NONLINEARITY = 0.02 # 三阶非线性系数
VIRTUAL_CHANNEL_DELAY = 5           # 样点延迟
VIRTUAL_CHANNEL_ATTENUATION = 0.9   # 线性衰减
```

然后运行：

```bash
python main.py --offline 1 --use-virtual-channel 1 --use-awg 0
```

虚拟信道会替代示波器读取，直接由 TX 生成 RX，便于无硬件时调试与维护。

### 5. 切换导频图案

在 `config.py` 中设置：

```python
PILOT_PATTERN = "comb"   # "training_only" / "comb" / "mesh"
PILOT_COMB_SPACING = 8   # 梳状导频子载波间隔
PILOT_MESH_FREQ_SPACING = 8
PILOT_MESH_TIME_SPACING = 5
```

- `training_only`：仅使用前几列 training symbol（与原 MATLAB 一致），所有子载波都传数据。
- `comb`：固定若干子载波全部传已知导频，适合快速时变信道；bitloading 时会自动把这些导频子载波的比特数置 0。
- `mesh`：时频二维稀疏导频，开销更小；插值精度受导频密度影响，建议先用虚拟信道验证后再上硬件。

> 提示：comb 的 SNR 估计会跳过纯导频子载波，并用最近邻数据子载波的 SNR 填充；`mean SNR` 已改用 `nanmean` 避免导频位置把均值撑大。

## 自动绘图与 CodePlot v5 脚本

默认情况下，图像会在 **Spyder 的 Plots 窗口** 中直接显示（通过 `plt.show()`）。

同时，每次测试都会在 `data/codeplot_assets/<run_id>/` 下保存：

```
data/codeplot_assets/<run_id>/
├── data/                # 每张图的 *.npz 数据
│   ├── SNRest_QPSK_time.npz
│   ├── SNRest_QPSK_spec.npz
│   ├── SNRest_QPSK_constellation.npz
│   ├── SNRest_QPSK_nonlinearity.npz
│   ├── SNR_QPSK.npz
│   ├── bit_power_loading.npz
│   ├── DMT_bitloading_Tx_time.npz
│   ├── DMT_bitloading_Tx_spec.npz
│   ├── DMT_bitloading_Rx_spec.npz
│   ├── DMT_bitloading_nonlinearity.npz
│   ├── SNR_compare.npz
│   ├── ser_ber_per_carrier.npz
│   ├── constellation_density.npz
│   └── constellation_by_order.npz
└── scripts/             # 每张图的可编辑 Python 脚本
    ├── SNRest_QPSK_time.py
    ├── SNRest_QPSK_spec.py
    ...
    └── constellation_by_order.py
```

要事后编辑某张图，打开 `codeplot_v5.py`，点击 **📂 加载脚本**，选择对应的 `scripts/<name>.py` 即可。脚本内使用 `fig` 变量，CodePlot v5 会自动把数据按相对路径从 `../data/<name>.npz` 加载进来。

### 当前绘制的图

| 名称 | 说明 |
|------|------|
| `SNRest_QPSK_time` / `SNRest_QPSK_spec` / `SNRest_QPSK_constellation` | QPSK TX 时域 / 频域 / 星座图 |
| `SNRest_QPSK_rx_spec` | QPSK RX 频谱 |
| `SNRest_QPSK_nonlinearity` | TX-RX 幅值非线性散点/密度图（蓝绿色 `GnBu`） |
| `SNR_QPSK` | QPSK 阶段估计的每载波 SNR（单位 dB） |
| `bit_power_loading` | 每子载波 SNR、bit loading、power loading，并标注 ratio |
| `DMT_bitloading_Tx_time` / `DMT_bitloading_Tx_spec` | Bitloading TX 时域 / 频谱 |
| `DMT_bitloading_Rx_spec` / `DMT_bitloading_nonlinearity` | Bitloading RX 频谱 / 非线性 |
| `SNR_compare` | QPSK 估计 SNR vs 最终恢复 SNR（单位 dB） |
| `constellation_density` | 按调制阶数分类的星座点密度热力图（蓝绿色 `GnBu`） |
| `constellation_by_order` | 按调制阶数分类的 RX 星座散点图（每阶数一张子图） |
| `ser_ber_per_carrier` | 每个子载波的 SER 与 BER |

> 注意：spectrogram（时频谱图）已移除，因为显示效果不佳。

> **Spyder 中看不到图的排查：**
> 1. 确认 Spyder 环境已安装 `matplotlib-inline`（`pip install matplotlib-inline`）。
> 2. 在 Spyder 菜单中选择 **Tools > Preferences > IPython console > Graphics**，把 **Backend** 设为 **Inline**。
> 3. 重启 IPython Console 后再运行 `main.py`。

在纯命令行/无显示环境中运行时，可设置：

```bash
set MPLBACKEND=Agg   # Windows cmd
# 或
$env:MPLBACKEND="Agg" # PowerShell
# 或
export MPLBACKEND=Agg # Git Bash
```

或直接在 `config.py` 中关闭显示：

```python
PLOT_SHOW = False
```

## 实验数据可视化 GUI

`dmt_gui.py` 提供一个桌面 GUI，用于浏览每次实验保存的数据并直接运行测试（无需额外依赖，tkinter + matplotlib，已适配高 DPI 屏幕）：

```bash
.venv\Scripts\python dmt_gui.py
```

包含四个子页面：

| 子页面 | 内容 |
|--------|------|
| 波形时频域 | QPSK 探测与 DMT Bitloading 的 TX/RX 时域波形、频谱 |
| DMT 符号调制 | Bit/Power Loading、QPSK 星座图、按调制阶数分类的 RX 星座图与密度热力图 |
| 传输实验结果 | 历次实验记录表（速率/BER/SER/SNR）、SNR 对比、每子载波 SER/BER、TX-RX 非线性、历次实验趋势 |
| 运行测试 | 选择模式（在线 / 离线 / 虚拟信道）与步骤后直接调用 `main.py`，日志实时显示，完成自动刷新 |

顶部通过 `run_id` 下拉框切换实验；在"传输实验结果"页点击记录表的行也可切换。数据来自 `data/codeplot_assets/<run_id>/data/*.npz` 与 `data/records/record_*.json`，运行新实验后点"刷新数据"即可。

## 传输记录

每次完整测试都会自动生成唯一编号 `run_id`（形如 `20260731_024911_164ad1`），并在 `data/records/` 下保存：

- `record_<run_id>.json`：完整参数与结果（导频、虚拟信道、速率、BER、SER、SNR、ratio、NN 等）
- `record_<run_id>.txt`：文本摘要

记录内容包括：
- 测试编号与时间戳
- 导频图案、是否使用虚拟信道/NN
- 虚拟信道参数（fc、SNR、非线性、延迟、衰减）
- 信道探测估算速率（estimated rate）
- 最终 bitloading 速率（final rate）
- 最终 BER / SER
- 恢复 SNR（dB）
- ratio

## 关键参数

在 `config.py` 中调整：

| 参数 | 含义 |
|------|------|
| `CARRIERNO` | 子载波总数 (1056) |
| `ZEROPAD1` | 每侧零填充 (16) |
| `UPSAMPLENO` | 上采样倍数 (2) |
| `DATANO_QPSK` | QPSK 符号数 (100) |
| `DATANO_BPL` | Bitloading 符号数 (200) |
| `AWG_SAMPLE_RATE` | AWG 采样率 8 GSa/s |
| `OSC_SAMPLE_RATE` | 示波器采样率 10 GSa/s |
| `M8190A_VISA_ADDR` | AWG VISA 地址 |
| `OSC_VISA_ADDR` | 示波器 USB VISA 地址 |
| `POSTEQ_FLAG` | 后均衡类型：0=无 NN, 1=RNN/GRU, 2=MLP, 3=Volterra；非 0 时 main.py 默认调用 NN |
| `USE_NN` | 由 `POSTEQ_FLAG` 自动决定，是否默认调用 NN 后均衡 |
| `PLOT_SHOW` / `PLOT_SAVE` | 是否显示 / 保存图像 |


## 注意事项

1. 示波器 USB-B 连接时，请确保 Keysight IO Libraries Suite 已识别到设备，且资源字符串以 `USB0::` 开头。
2. M8190A 通过 TCPIP socket 连接时，请确认防火墙放行 5025 端口。
3. 默认不启用 NN，以加快运行速度；需要后均衡时加 `--use-nn 1`，第一次运行会训练 8 个 epoch。
4. 若 `PyVISA` 找不到资源，可在 Python 中先测试：

```python
import pyvisa
rm = pyvisa.ResourceManager()
print(rm.list_resources())
```
