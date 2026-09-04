"""DMT communication system configuration parameters.

Parameters are kept consistent with the original MATLAB code (STEP1/STEP2/STEP3/STEP4).
"""
from pathlib import Path

# -----------------------------------------------------------------------------
# Project paths
# -----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
TXDATA_DIR = DATA_DIR / "txdata"
RXDATA_DIR = DATA_DIR / "rxdata"
NN_DIR = DATA_DIR / "nn"

# New: plotting and record directories
PLOT_DIR = DATA_DIR / "plots"
RECORD_DIR = DATA_DIR / "records"

for _d in (DATA_DIR, TXDATA_DIR, RXDATA_DIR, NN_DIR, PLOT_DIR, RECORD_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# DMT signal parameters
# -----------------------------------------------------------------------------
CARRIERNO = 1056    # Total number of subcarriers (1056)
ZEROPAD1 = 16    # Number of zero-padding subcarriers on each side (16)
UPSAMPLENO = 2                     # Upsampling factor
DATANO_QPSK = 100                  # Number of symbols for QPSK channel probing
DATANO_BPL = 200                   # Number of symbols for bitloading transmission
TRAININGNO = 20                    # Number of training symbols for channel estimation
RATIO = 130                         # Subtrahend for raise_num in bitloading, affecting spectral efficiency
CP_RATIO = 1 / 32                  # Cyclic prefix ratio

# Commonly derived constants (computed from above)
CARRIERNO1 = CARRIERNO // 2 - ZEROPAD1   # Number of valid subcarriers (512)
CP = int(CP_RATIO * CARRIERNO)           # CP length (33)
EFF_CARNO = (CARRIERNO - ZEROPAD1 * 2) // 2

# -----------------------------------------------------------------------------
# Modulation/normalization parameters
# -----------------------------------------------------------------------------
NORMALIZE_FLAG = 0                 # 1=average power normalization, 0=peak amplitude normalization
CONSTELLATION_APSK = "APSK"
CONSTELLATION_QAM = "QAM"

# -----------------------------------------------------------------------------
# Pilot pattern options
# -----------------------------------------------------------------------------
# "training_only": use only the first few training symbols for channel estimation (consistent with original MATLAB)
# "comb":          comb pilots: fixed subcarriers all carry known pilots
# "mesh":          mesh pilots: insert pilots on a 2D time-frequency grid
PILOT_PATTERN = "training_only"
PILOT_VALUE = 1.0 + 0.0j             # Pilot symbol value
PILOT_COMB_START = 4                 # Comb pilot starting subcarrier
PILOT_COMB_SPACING = 8               # Comb pilot subcarrier spacing
PILOT_MESH_START_FREQ = 4            # Mesh pilot starting subcarrier
PILOT_MESH_FREQ_SPACING = 8          # Mesh pilot frequency spacing
PILOT_MESH_START_TIME = 2            # Mesh pilot starting symbol
PILOT_MESH_TIME_SPACING = 5          # Mesh pilot time spacing

# -----------------------------------------------------------------------------
# Pre-equalization options (corresponds to MATLAB pre_equ_flag)
# 0=no Pre; 1=Symbol Pre; 2=Wave NN Pre; 3=Wave Hardware Pre; 4=Wave NN+Hardware Pre
# -----------------------------------------------------------------------------
PRE_EQU_FLAG = 3

# -----------------------------------------------------------------------------
# Pre-emphasis weight generation (port from MATLAB Pre.m, outputs th7.txt)
# PRE_METHOD: 0=normal fit; 1=inverse+normal; 2=cut off; 3=peak point;
#             4=peak point fit; 5=Hardware Pre (Bridge-T equalizer response)
# -----------------------------------------------------------------------------
PRE_METHOD = 4
PRE_EQUAL_DB = 20                  # Cutoff/segmentation threshold (dB)
PRE_EQUAL_DB2 = 20                 # Second threshold (diagnostic only for method=4)

# Hardware Pre (method=5) Bridge-T equalizer parameters, consistent with Pre.m case 5
HW_PRE_FBEGIN = 1                  # Starting frequency index (MHz)
HW_PRE_ADB = 5                     # Maximum attenuation (dB)
HW_PRE_FCEN_MHZ = 600              # Center frequency (MHz), <1000
HW_PRE_FHALF_MHZ = 400             # Half-attenuation bandwidth (MHz), <1000
HW_PRE_FEND = 600                  # End frequency (MHz, relative to FBEGIN)
HW_PRE_R0 = 50                     # Reference impedance (Ohm)

# -----------------------------------------------------------------------------
# Sample rates / hardware parameters
# -----------------------------------------------------------------------------
AWG_SAMPLE_RATE = 3.0e9              # AWG sample rate (Hz)
OSC_SAMPLE_RATE = 10e9             # Oscilloscope sample rate (Hz)
AWG_VPP = 0.4                      # AWG output amplitude (Vpp)

# -----------------------------------------------------------------------------
# Run mode
# -----------------------------------------------------------------------------
OFFLINE_FLAG = 1                   # 1=offline processing of existing scope files, 0=online connection to scope

# -----------------------------------------------------------------------------
# Virtual channel (for debugging/maintenance without instruments)
# -----------------------------------------------------------------------------
USE_VIRTUAL_CHANNEL = 0          # 1=offline mode uses virtual channel to generate RX, 0=read existing files
VIRTUAL_CHANNEL_FC = 0.8e9         # First-order low-pass cutoff frequency (Hz), simulating TX high-frequency roll-off
VIRTUAL_CHANNEL_SNR_DB = 20        # Receiver SNR (dB); higher value means less noise
VIRTUAL_CHANNEL_NONLINEARITY = 0.02  # Receiver third-order nonlinearity coefficient
VIRTUAL_CHANNEL_DELAY = 5          # Integer sample delay (recommended <= CP)
VIRTUAL_CHANNEL_ATTENUATION = 0.9  # Linear amplitude attenuation

# -----------------------------------------------------------------------------
# Hardware VISA addresses
# -----------------------------------------------------------------------------
# M8190A VISA address (choose one according to actual setup; replace localhost with AWG IP):
#   TCPIP Socket (most common, no extra VISA backend needed):
#     "TCPIP0::192.168.1.10::5025::SOCKET"
#   HiSLIP:
#     "TCPIP0::192.168.1.10::hislip0::INSTR"
#   VXI-11:
#     "TCPIP0::192.168.1.10::inst0::INSTR"
#   USB-PXI:
#     "USB-PXI0::5564::4708::6&26821990&0&1-1::INSTR"
M8190A_VISA_ADDR = "TCPIP0::localhost::5025::SOCKET"
#M8190A_VISA_ADDR = "TCPIP0::localhost::60005::SOCKET"
#M8190A_VISA_ADDR = "USB-PXI0::5564::4708::6&26821990&0&1-1::INSTR "
M8190A_PORT = 5025
    
# M8190A output route selection:
#   "DC"  - DC-coupled amplified output (default, common for baseband/DMT)
#   "AC"  - AC-coupled amplified output (DC-blocked, common for RF/IF)
#   "DAC" - Direct DAC output (unamplified, smallest amplitude)
AWG_OUTPUT_ROUTE = "DAC"

# Oscilloscope connection (consistent with MATLAB, default TCPIP port 5025):
#   TCPIP Socket:  "TCPIP0::192.168.1.10::5025::SOCKET"
#   USB-B/USBTMC:  "USB0::0x0957::0x17A6::MY12345678::INSTR"
# Leave empty to auto-detect the first USB instrument
# SC_VISA_ADDR = "USB1::0x2A8D::0x9008::MY50400106::0::INSTR"
# Use network port (VXI-11 protocol)
OSC_VISA_ADDR = "TCPIP0::192.168.193.176::5025::SOCKET"
OSC_CHANNEL = "CHAN1"              # MATLAB oscrunQPSK.m / oscrunDMT.m both use CHAN2
OSC_TIMEBASE_SCALE = 80e-6         # Timebase for QPSK probing (s/div); DMT bitloading uses 60e-6

# -----------------------------------------------------------------------------
# File paths
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

# NN-related files (consistent with ZY_BiGRU_GPU.py default naming)
NN_TX_FILE = NN_DIR / "Txdata_NN.txt"
NN_RX1_FILE = NN_DIR / "Rxdata_NN1.txt"
NN_RX2_FILE = NN_DIR / "Rxdata_NN2.txt"
NN_OUTPUT1_FILE = NN_DIR / "Rxdata_afterNN1.txt"
NN_OUTPUT2_FILE = NN_DIR / "Rxdata_afterNN2.txt"
NN_PRETRAINED = NN_DIR / "pretrained_model.pth"
NN_MODEL_TEMP = NN_DIR / "trained_model_temp.pth"

# -----------------------------------------------------------------------------
# Plotting and post-equalization options
# -----------------------------------------------------------------------------
PLOT_SHOW = True                   # Whether to plt.show() figures in Spyder
PLOT_SAVE = True                   # Whether to save PNG (main flow now uses CodePlot v5 scripts)
PLOT_DPI = 300                     # Default image resolution (shared by Spyder display and saving)

POSTEQ_FLAG = 1                    # 0=no NN, 1=RNN/GRU, 2=MLP, 3=Volterra
USE_NN = 1 if POSTEQ_FLAG != 0 else 0  # Whether main.py calls NN post-equalizer by default


# -----------------------------------------------------------------------------
# Random seed (for reproducibility)
# -----------------------------------------------------------------------------
RANDOM_SEED = 110


# -----------------------------------------------------------------------------
# Keithley 2400 SourceMeter (RS-232 / USB-to-RS232)
# -----------------------------------------------------------------------------
K2400_PORT = "COM1"               # Serial port name (auto-detect if left empty)
K2400_BAUDRATE = 9600             # Default RS-232 baud rate for the 2400
K2400_TIMEOUT = 5.0               # Serial read timeout in seconds
K2400_SOURCE_MODE = "voltage"     # "voltage" or "current"
K2400_LEVEL = 0.0                 # Source level (V in voltage mode, A in current mode)
K2400_COMPLIANCE = 0.1            # Compliance limit (A in voltage mode, V in current mode)
K2400_NPLC = 1.0                  # Measurement integration time


# -----------------------------------------------------------------------------
# Grid scan parameters (bias vs Vpp)
# -----------------------------------------------------------------------------
GRID_SCAN_PARAM1_NAME = "bias_voltage"  # displayed name / CSV header
GRID_SCAN_PARAM1_MODE = "voltage"       # "voltage" or "current" (Keithley source mode)
GRID_SCAN_PARAM1_START = 0.0
GRID_SCAN_PARAM1_STOP = 1.0
GRID_SCAN_PARAM1_STEP = 0.2
GRID_SCAN_VPP_START = 0.1
GRID_SCAN_VPP_STOP = 0.5
GRID_SCAN_VPP_STEP = 0.1
GRID_SCAN_RUN_MODE = "step1-4"          # "step1-4" or "step1-2"
GRID_SCAN_REPEATS = 1                   # repeats of the final measurement step
