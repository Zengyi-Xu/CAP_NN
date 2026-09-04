"""DMT 通信系统配置参数。

参数与原始 MATLAB 代码（STEP1/STEP2/STEP3/STEP4）保持一致。
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
CARRIERNO = 1056    # 子载波总数（1056）
ZEROPAD1 = 16    # 每侧补零子载波数（16）
UPSAMPLENO = 2                     # 上采样因子
DATANO_QPSK = 100                  # QPSK 信道探测的符号数
DATANO_BPL = 200                   # 比特加载传输的符号数
TRAININGNO = 20                    # 信道估计的训练符号数
RATIO = 130                         # 比特加载中 raise_num 的减数，影响频谱效率
CP_RATIO = 1 / 32                  # 循环前缀比例

# 常用派生常量（由上述参数计算得到）
CARRIERNO1 = CARRIERNO // 2 - ZEROPAD1   # 有效子载波数（512）
CP = int(CP_RATIO * CARRIERNO)           # 循环前缀长度（33）
EFF_CARNO = (CARRIERNO - ZEROPAD1 * 2) // 2

# -----------------------------------------------------------------------------
# 调制/归一化参数
# -----------------------------------------------------------------------------
NORMALIZE_FLAG = 0                 # 1=平均功率归一化，0=峰值幅度归一化
CONSTELLATION_APSK = "APSK"
CONSTELLATION_QAM = "QAM"

# -----------------------------------------------------------------------------
# 导频图样选项
# -----------------------------------------------------------------------------
# "training_only"：仅使用最前面的少数训练符号进行信道估计（与原始 MATLAB 一致）
# "comb"：          梳状导频：固定子载波全部承载已知导频
# "mesh"：          网状导频：在二维时频网格上插入导频
PILOT_PATTERN = "training_only"
PILOT_VALUE = 1.0 + 0.0j             # 导频符号取值
PILOT_COMB_START = 4                 # 梳状导频起始子载波
PILOT_COMB_SPACING = 8               # 梳状导频子载波间隔
PILOT_MESH_START_FREQ = 4            # 网状导频起始子载波
PILOT_MESH_FREQ_SPACING = 8          # 网状导频频域间隔
PILOT_MESH_START_TIME = 2            # 网状导频起始符号
PILOT_MESH_TIME_SPACING = 5          # 网状导频时域间隔

# -----------------------------------------------------------------------------
# 预均衡选项（对应 MATLAB 的 pre_equ_flag）
# 0=无预均衡；1=符号级预均衡；2=波形级 NN 预均衡；3=波形级硬件预均衡；4=波形级 NN+硬件预均衡
# -----------------------------------------------------------------------------
PRE_EQU_FLAG = 3

# -----------------------------------------------------------------------------
# 预加重权重生成（移植自 MATLAB Pre.m，输出 th7.txt）
# PRE_METHOD: 0=常规拟合；1=逆+常规；2=截止；3=峰值点；
#             4=峰值点拟合；5=硬件预均衡（桥式 T 型均衡器响应）
# -----------------------------------------------------------------------------
PRE_METHOD = 4
PRE_EQUAL_DB = 20                  # 截止/分段阈值（dB）
PRE_EQUAL_DB2 = 20                 # 第二阈值（仅方法 4 诊断用）

# 硬件预均衡（method=5）桥式 T 型均衡器参数，与 Pre.m 的 case 5 一致
HW_PRE_FBEGIN = 1                  # 起始频率索引（MHz）
HW_PRE_ADB = 5                     # 最大衰减（dB）
HW_PRE_FCEN_MHZ = 600              # 中心频率（MHz），<1000
HW_PRE_FHALF_MHZ = 400             # 半衰减带宽（MHz），<1000
HW_PRE_FEND = 600                  # 结束频率（MHz，相对于 FBEGIN）
HW_PRE_R0 = 50                     # 参考阻抗（欧姆）

# -----------------------------------------------------------------------------
# 采样率 / 硬件参数
# -----------------------------------------------------------------------------
AWG_SAMPLE_RATE = 3.0e9              # AWG 采样率（Hz）
OSC_SAMPLE_RATE = 10e9             # 示波器采样率（Hz）
AWG_VPP = 0.4                      # AWG 输出幅度（Vpp）

# -----------------------------------------------------------------------------
# 运行模式
# -----------------------------------------------------------------------------
OFFLINE_FLAG = 1                   # 1=离线处理已有示波器文件，0=在线连接示波器

# -----------------------------------------------------------------------------
# 虚拟信道（用于无仪器时的调试/维护）
# -----------------------------------------------------------------------------
USE_VIRTUAL_CHANNEL = 0          # 1=离线模式使用虚拟信道生成接收信号，0=读取已有文件
VIRTUAL_CHANNEL_FC = 0.8e9         # 一阶低通截止频率（Hz），模拟发射端高频滚降
VIRTUAL_CHANNEL_SNR_DB = 20        # 接收端信噪比（dB）；数值越大噪声越小
VIRTUAL_CHANNEL_NONLINEARITY = 0.02  # 接收端三阶非线性系数
VIRTUAL_CHANNEL_DELAY = 5          # 整数样点延迟（建议 <= CP）
VIRTUAL_CHANNEL_ATTENUATION = 0.9  # 线性幅度衰减

# -----------------------------------------------------------------------------
# 硬件 VISA 地址
# -----------------------------------------------------------------------------
# M8190A VISA 地址（根据实际连接任选其一；将 localhost 替换为 AWG 的 IP）：
#   TCPIP Socket（最常用，无需额外 VISA 后端）：
#     "TCPIP0::192.168.1.10::5025::SOCKET"
#   HiSLIP：
#     "TCPIP0::192.168.1.10::hislip0::INSTR"
#   VXI-11：
#     "TCPIP0::192.168.1.10::inst0::INSTR"
#   USB-PXI：
#     "USB-PXI0::5564::4708::6&26821990&0&1-1::INSTR"
M8190A_VISA_ADDR = "TCPIP0::localhost::5025::SOCKET"
#M8190A_VISA_ADDR = "TCPIP0::localhost::60005::SOCKET"
#M8190A_VISA_ADDR = "USB-PXI0::5564::4708::6&26821990&0&1-1::INSTR "
M8190A_PORT = 5025
    
# M8190A 输出路由选择：
#   "DC"  - 直流耦合放大输出（默认，常用于基带/DMT）
#   "AC"  - 交流耦合放大输出（隔直，常用于射频/中频）
#   "DAC" - DAC 直接输出（无放大，幅度最小）
AWG_OUTPUT_ROUTE = "DAC"

# 示波器连接（与 MATLAB 一致，默认 TCPIP 端口 5025）：
#   TCPIP Socket:  "TCPIP0::192.168.1.10::5025::SOCKET"
#   USB-B/USBTMC:  "USB0::0x0957::0x17A6::MY12345678::INSTR"
# 留空则自动检测第一台 USB 仪器
# SC_VISA_ADDR = "USB1::0x2A8D::0x9008::MY50400106::0::INSTR"
# 使用网络端口（VXI-11 协议）
OSC_VISA_ADDR = "TCPIP0::192.168.193.176::5025::SOCKET"
OSC_CHANNEL = "CHAN1"              # MATLAB oscrunQPSK.m / oscrunDMT.m 均使用 CHAN2
OSC_TIMEBASE_SCALE = 80e-6         # QPSK 探测的时基（s/div）；DMT 比特加载使用 60e-6

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
F_GRID_FILE = DATA_DIR / "f_grid.txt"

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

# NN 相关文件（与 ZY_BiGRU_GPU.py 默认命名一致）
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
PLOT_SHOW = True                   # 是否在 Spyder 中 plt.show() 显示图窗
PLOT_SAVE = True                   # 是否保存 PNG（主流程现使用 CodePlot v5 脚本）
PLOT_DPI = 300                     # 默认图像分辨率（Spyder 显示与保存共用）

POSTEQ_FLAG = 1                    # 0=无 NN，1=RNN/GRU，2=MLP，3=Volterra
USE_NN = 1 if POSTEQ_FLAG != 0 else 0  # main.py 默认是否调用 NN 后均衡器


# -----------------------------------------------------------------------------
# 随机种子（用于可复现）
# -----------------------------------------------------------------------------
RANDOM_SEED = 110


# -----------------------------------------------------------------------------
# Keithley 2400 源表（RS-232 / USB 转 RS-232）
# -----------------------------------------------------------------------------
K2400_PORT = "COM1"               # 串口名（留空则自动检测）
K2400_BAUDRATE = 9600             # 2400 的默认 RS-232 波特率
K2400_TIMEOUT = 5.0               # 串口读取超时（秒）
K2400_SOURCE_MODE = "voltage"     # "voltage"（电压）或 "current"（电流）
K2400_LEVEL = 0.0                 # 源输出电平（电压模式为 V，电流模式为 A）
K2400_COMPLIANCE = 0.1            # 合规限值（电压模式为 A，电流模式为 V）
K2400_NPLC = 1.0                  # 测量积分时间


# -----------------------------------------------------------------------------
# 网格扫描参数（偏置电压 vs Vpp）
# -----------------------------------------------------------------------------
GRID_SCAN_PARAM1_NAME = "bias_voltage"  # 显示名称 / CSV 表头
GRID_SCAN_PARAM1_MODE = "voltage"       # "voltage" 或 "current"（Keithley 源模式）
GRID_SCAN_PARAM1_START = 0.0
GRID_SCAN_PARAM1_STOP = 1.0
GRID_SCAN_PARAM1_STEP = 0.2
GRID_SCAN_VPP_START = 0.1
GRID_SCAN_VPP_STOP = 0.5
GRID_SCAN_VPP_STEP = 0.1
GRID_SCAN_RUN_MODE = "step1-4"          # "step1-4" 或 "step1-2"
GRID_SCAN_REPEATS = 1                   # 最终测量步骤的重复次数
