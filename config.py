"""DMT 通信系统配置参数.

参数尽量与原始 MATLAB 代码 (STEP1/STEP2/STEP3/STEP4) 保持一致。
"""
from pathlib import Path

# -----------------------------------------------------------------------------
# 项目路径
# -----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
TXDATA_DIR = DATA_DIR / "txdata"
RXDATA_DIR = DATA_DIR / "rxdata"
NN_DIR = DATA_DIR / "nn"

# 新增：绘图与记录目录
PLOT_DIR = DATA_DIR / "plots"
RECORD_DIR = DATA_DIR / "records"

for _d in (DATA_DIR, TXDATA_DIR, RXDATA_DIR, NN_DIR, PLOT_DIR, RECORD_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# DMT 信号参数
# -----------------------------------------------------------------------------
CARRIERNO = 1024 + 16 * 2          # 子载波总数 (1056)
ZEROPAD1 = 2 * 8                   # 每侧零填充数 (16)
UPSAMPLENO = 2                     # 上采样倍数
DATANO_QPSK = 100                  # QPSK 信道探测时符号数
DATANO_BPL = 200                   # bitloading 传输时符号数
TRAININGNO = 20                    # 用于信道估计的 training symbol 数
CP_RATIO = 1 / 32                  # 循环前缀比例

# 常用导出常量（由上面计算得到）
CARRIERNO1 = CARRIERNO // 2 - ZEROPAD1   # 有效子载波数 (512)
CP = int(CP_RATIO * CARRIERNO)           # CP 长度 (33)
EFF_CARNO = (CARRIERNO - ZEROPAD1 * 2) // 2

# -----------------------------------------------------------------------------
# 调制/归一化参数
# -----------------------------------------------------------------------------
NORMALIZE_FLAG = 1                 # 1=平均功率归一化, 0=最大幅值归一化
CONSTELLATION_APSK = "APSK"
CONSTELLATION_QAM = "QAM"

# -----------------------------------------------------------------------------
# 导频图案选项
# -----------------------------------------------------------------------------
# "training_only": 仅使用前几列 training symbol 做信道估计（与原 MATLAB 一致）
# "comb":          梳状导频，固定若干子载波全部传已知导频
# "mesh":          网状导频，在时频二维网格上插入导频
PILOT_PATTERN = "training_only"
PILOT_VALUE = 1.0 + 0.0j             # 导频符号值
PILOT_COMB_START = 4                 # 梳状导频起始子载波
PILOT_COMB_SPACING = 8               # 梳状导频子载波间隔
PILOT_MESH_START_FREQ = 4            # 网状导频起始子载波
PILOT_MESH_FREQ_SPACING = 8          # 网状导频频率间隔
PILOT_MESH_START_TIME = 2            # 网状导频起始符号
PILOT_MESH_TIME_SPACING = 5          # 网状导频时间间隔

# -----------------------------------------------------------------------------
# 预均衡选项 (与 MATLAB pre_equ_flag 对应)
# 0=no Pre; 1=Symbol Pre; 2=Wave NN Pre; 3=Wave Hardware Pre; 4=Wave NN+Hardware Pre
# -----------------------------------------------------------------------------
PRE_EQU_FLAG = 3

# -----------------------------------------------------------------------------
# 采样率/硬件参数
# -----------------------------------------------------------------------------
AWG_SAMPLE_RATE = 8e9              # AWG 采样率 (Hz)
OSC_SAMPLE_RATE = 10e9             # 示波器采样率 (Hz)
AWG_VPP = 0.5                      # AWG 输出幅度 (Vpp)

# -----------------------------------------------------------------------------
# 虚拟信道（用于无仪器时的调试/维护）
# -----------------------------------------------------------------------------
USE_VIRTUAL_CHANNEL = 1            # 1=离线模式时使用虚拟信道生成 RX，0=读取已有文件
VIRTUAL_CHANNEL_FC = 1.6e9         # 一阶低通截止频率 (Hz)，模拟发射端高频衰减
VIRTUAL_CHANNEL_SNR_DB = 30        # 接收机信噪比 (dB)，数值越高噪声越小
VIRTUAL_CHANNEL_NONLINEARITY = 0.02  # 接收机三阶非线性系数
VIRTUAL_CHANNEL_DELAY = 5          # 整数样点延迟（建议 <= CP）
VIRTUAL_CHANNEL_ATTENUATION = 0.9  # 线性幅度衰减

# -----------------------------------------------------------------------------
# 硬件 VISA 地址
# -----------------------------------------------------------------------------
# M8190A: 默认通过 TCPIP socket 连接；可改为其它 VISA 资源字符串
M8190A_VISA_ADDR = "TCPIP0::localhost::5025::SOCKET"
M8190A_PORT = 5025

# 示波器 USB-B (USBTMC) 资源字符串；留空则自动查找第一个 USB 仪器
OSC_VISA_ADDR = ""                 # e.g. "USB0::0x0957::0x17A6::MY12345678::INSTR"
OSC_CHANNEL = "CHAN2"              # 读取通道
OSC_TIMEBASE_SCALE = 60e-6         # 时基 (s/div)

# -----------------------------------------------------------------------------
# 文件路径
# -----------------------------------------------------------------------------
SNR_TABLE_APSK1 = DATA_DIR / "SNRtable_APSK1.txt"
SNR_TABLE_APSK3 = DATA_DIR / "SNRtable_APSK3.txt"
SNR_TABLE_APSK4 = DATA_DIR / "SNRtable_APSK4.txt"
SNR_TABLE_APSK5 = DATA_DIR / "SNRtable_APSK5.txt"
SNR_TABLE_APSK6 = DATA_DIR / "SNRtable_APSK6.txt"
SNR_TABLE_APSK7 = DATA_DIR / "SNRtable_APSK7.txt"
SNR_TABLE_QAM1 = DATA_DIR / "SNRtable_QAM1.txt"
SNR_TABLE_FEC1 = DATA_DIR / "SNRtable_FEC1.txt"
SNR_TABLE_FEC2 = DATA_DIR / "SNRtable_FEC2.txt"
SNR_TABLE_FEC3 = DATA_DIR / "SNRtable_FEC3.txt"
SNR_TABLE_FEC4 = DATA_DIR / "SNRtable_FEC4.txt"
SNR_TABLE_TARGET_34E3 = DATA_DIR / "SNRtableTarget3dot4E_3.txt"
SNR_TABLE_TARGET_38E3 = DATA_DIR / "SNRtableTarget3dot8E_3.txt"
SNR_TABLE_TARGET_26E3 = DATA_DIR / "SNRtableTarget2dot6E_3.txt"
SNR_TABLE_TARGET_24E3 = DATA_DIR / "SNRtableTarget2dot4E_3.txt"

QAMORDERALL_FILE = DATA_DIR / "QAMorderall.txt"
TH7_FILE = DATA_DIR / "th7.txt"
HARDWARE_PRE_FILE = DATA_DIR / "f_hardware_dB.txt"

ORIGIN_DEC_DATA_QPSK = DATA_DIR / "origin_dec_data_QPSK.txt"
ORIGIN_DEC_DATA_BPL = DATA_DIR / "origin_dec_data.txt"
ORIGIN_BINARY_FILE = DATA_DIR / "origindata_binary.txt"

DEMOD_FILE_QPSK = DATA_DIR / "demodulationfile_QPSK.mat"
BITPOWER_QPSK = DATA_DIR / "bitpowerInformation_QPSK.mat"
DEMOD_FILE_BPL = DATA_DIR / "demodulationfile.mat"
BITPOWER_BPL = DATA_DIR / "bitpowerInformation.mat"

FINAL_SNR_QPSK = DATA_DIR / "finalSNReveryCarrier_QPSK.txt"

TX_QPSK_FILE = TXDATA_DIR / "SNRest_QPSK.txt"
TX_QPSK_PRE_FILE = TXDATA_DIR / "pre_SNRest_QPSK.txt"
TX_BPL_FILE = TXDATA_DIR / "DMT_bitloading_Tx_QAM.txt"
TX_BPL_PRE_FILE = TXDATA_DIR / "pre_DMT_bitloading_Tx_QAM.txt"

WAVEFORM_DUMMY_LEN = DATA_DIR / "waveform_dummy_len.txt"
COUNT_FILE = DATA_DIR / "count.txt"

# NN 相关文件 (与 ZY_BiGRU_GPU.py 默认命名保持一致)
NN_TX_FILE = NN_DIR / "Txdata_NN.txt"
NN_RX1_FILE = NN_DIR / "Rxdata_NN1.txt"
NN_RX2_FILE = NN_DIR / "Rxdata_NN2.txt"
NN_OUTPUT1_FILE = NN_DIR / "Rxdata_afterNN1.txt"
NN_OUTPUT2_FILE = NN_DIR / "Rxdata_afterNN2.txt"
NN_PRETRAINED = NN_DIR / "pretrained_model.pth"
NN_MODEL_TEMP = NN_DIR / "trained_model_temp.pth"

# -----------------------------------------------------------------------------
# 绘图与后均衡选项
# -----------------------------------------------------------------------------
PLOT_SHOW = True                   # 是否在 Spyder 中 plt.show() 显示图像
PLOT_SAVE = False                  # 是否保存 PNG（主流程已改用 CodePlot v5 脚本）
PLOT_DPI = 150                     # 保存图像分辨率

USE_NN = 0                         # main.py 默认是否调用 ZY_BiGRU_GPU 做后均衡
POSTEQ_FLAG = 1                    # 0=无 NN, 1=RNN/GRU, 2=MLP, 3=Volterra

# -----------------------------------------------------------------------------
# 运行模式
# -----------------------------------------------------------------------------
OFFLINE_FLAG = 1                   # 1=离线处理已有 scope 文件, 0=在线连接示波器

# -----------------------------------------------------------------------------
# 随机种子 (保证可重复)
# -----------------------------------------------------------------------------
RANDOM_SEED = 110
