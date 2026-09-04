"""UW_APSK CAP Python 收发机的配置参数。

镜像自以下 MATLAB 参数集：
  - oldcapAPSKTxRx20220406.m   （单频带）
  - main_CAP_3band_totalB.m    （多频带）
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# 项目路径
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
TXDATA_DIR = DATA_DIR / "txdata"
RXDATA_DIR = DATA_DIR / "rxdata"
NN_DIR = DATA_DIR / "nn"
PLOT_DIR = DATA_DIR / "plots"
RECORD_DIR = DATA_DIR / "records"
GRID_SCAN_DIR = DATA_DIR / "grid_scans"

for _d in (DATA_DIR, TXDATA_DIR, RXDATA_DIR, NN_DIR, PLOT_DIR, RECORD_DIR, GRID_SCAN_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# 单频带 CAP 参数（oldcapAPSKTxRx20220406.m）
# ---------------------------------------------------------------------------
SB_NUMOFSYMBOLS = 51200
SB_QAMORDER = 64
SB_CONSTELLATION = "QAM"          # "APSK" 或 "QAM"
SB_UPSAMPLENO = 4
SB_ALPHA = 0.205
SB_SUBCAR = 0.5
SB_STARTFREQ = 1 / 100
SB_TAPS = 35
SB_AWG_SAMPLE_RATE = 2.0e9        # Hz

SB_SNR_DB = 27
SB_CHANNEL_FS = 100               # VLC 信道 Fs 参数
SB_CHANNEL_FACTOR = 40
SB_CHANNEL_NONLINEAR = False

SB_LMS_TAPS = 17
SB_LMS_MU = 0.005
SB_VOLD_TAPS = 11
SB_VOLD_MU = 0.0004
SB_NUMOF_TS = 8000

# ---------------------------------------------------------------------------
# 多频带 CAP 参数（main_CAP_3band_totalB.m）
# ---------------------------------------------------------------------------
MB_NUMOFSYMBOLS = 1024 * 32
MB_M = 16                         # 每个频带的调制阶数
MB_CONSTELLATION = "QAM"
MB_ROLLOFF = 0.2
MB_RS = 300e6                     # 聚合波特率
MB_FS = 1.2e9                     # 采样率
MB_CF = 0.11                      # 压缩因子
MB_SPAN = 8                       # 滤波器跨度（符号数）
MB_SHAPE = "srrc"
MB_NUM_BANDS = 3

MB_SNR_DB = 25
MB_CHANNEL_FS = 100
MB_CHANNEL_FACTOR = 15
MB_CHANNEL_NONLINEAR = False

# ---------------------------------------------------------------------------
# 运行模式
# ---------------------------------------------------------------------------
OFFLINE_FLAG = 1                  # 1 = 处理已保存的文件，0 = 连接在线硬件
USE_VIRTUAL_CHANNEL = 1           # 1 = 通过信道模型生成接收信号，0 = 读取文件

# ---------------------------------------------------------------------------
# 硬件 VISA 地址（复用自 DMT_PY_NN）
# ---------------------------------------------------------------------------
M8190A_VISA_ADDR = "TCPIP0::localhost::5025::SOCKET"
M8190A_PORT = 5025
AWG_OUTPUT_ROUTE = "DAC"

OSC_VISA_ADDR = "TCPIP0::192.168.193.176::5025::SOCKET"
OSC_CHANNEL = "CHAN1"
OSC_TIMEBASE_SCALE = 80e-6

# ---------------------------------------------------------------------------
# NN 后均衡器（复用自 DMT_PY_NN）
# ---------------------------------------------------------------------------
USE_NN = 0                        # 0 = 传统均衡器，1 = NN
POSTEQ_FLAG = 0                   # 0 = 无，1 = RNN/GRU，2 = MLP，3 = Volterra

NN_TX_FILE = NN_DIR / "Txdata_NN.txt"
NN_RX1_FILE = NN_DIR / "Rxdata_NN1.txt"
NN_OUTPUT1_FILE = NN_DIR / "Rxdata_afterNN1.txt"
NN_MODEL_TEMP = NN_DIR / "trained_model_temp.pth"

# ---------------------------------------------------------------------------
# 绘图 / 记录
# ---------------------------------------------------------------------------
PLOT_SHOW = False
PLOT_SAVE = True
PLOT_DPI = 300

# ---------------------------------------------------------------------------
# Keithley 2400 源表（RS-232 / USB 转 RS-232）
# ---------------------------------------------------------------------------
K2400_PORT = "COM1"               # 串口名（留空则自动检测）
K2400_BAUDRATE = 9600             # 2400 的默认 RS-232 波特率
K2400_TIMEOUT = 5.0               # 串口读取超时（秒）
K2400_SOURCE_MODE = "voltage"     # "voltage"（电压）或 "current"（电流）
K2400_LEVEL = 0.0                 # 源输出电平（电压模式为 V，电流模式为 A）
K2400_COMPLIANCE = 0.1            # 合规限值（电压模式为 A，电流模式为 V）
K2400_NPLC = 1.0                  # 测量积分时间


# ---------------------------------------------------------------------------
# 网格扫描参数（偏置电压 vs Vpp / 信噪比）
# ---------------------------------------------------------------------------
GRID_SCAN_PARAM1_NAME = "bias_voltage"  # 显示名称 / CSV 表头
GRID_SCAN_PARAM1_MODE = "voltage"       # "voltage" 或 "current"（Keithley 源模式）
GRID_SCAN_PARAM1_START = 0.0
GRID_SCAN_PARAM1_STOP = 1.0
GRID_SCAN_PARAM1_STEP = 0.2
GRID_SCAN_VPP_START = 0.1
GRID_SCAN_VPP_STOP = 0.5
GRID_SCAN_VPP_STEP = 0.1
GRID_SCAN_RUN_MODE = "singleband"       # "singleband"（单频带）或 "multiband"（多频带）
GRID_SCAN_REPEATS = 1                   # 每个网格点的重复次数


# ---------------------------------------------------------------------------
# 辅助常量
# ---------------------------------------------------------------------------
SB_SYMBOL_RATE = SB_AWG_SAMPLE_RATE / SB_UPSAMPLENO
