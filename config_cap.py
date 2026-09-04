"""Configuration for UW_APSK CAP Python transceiver.

Mirrors the MATLAB parameter set from:
  - oldcapAPSKTxRx20220406.m   (single-band)
  - main_CAP_3band_totalB.m    (multi-band)
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# Project paths
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
# Single-band CAP parameters (oldcapAPSKTxRx20220406.m)
# ---------------------------------------------------------------------------
SB_NUMOFSYMBOLS = 51200
SB_QAMORDER = 64
SB_CONSTELLATION = "QAM"          # "APSK" or "QAM"
SB_UPSAMPLENO = 4
SB_ALPHA = 0.205
SB_SUBCAR = 0.5
SB_STARTFREQ = 1 / 100
SB_TAPS = 35
SB_AWG_SAMPLE_RATE = 2.0e9        # Hz

SB_SNR_DB = 27
SB_CHANNEL_FS = 100               # VLC channel Fs parameter
SB_CHANNEL_FACTOR = 40
SB_CHANNEL_NONLINEAR = False

SB_LMS_TAPS = 17
SB_LMS_MU = 0.005
SB_VOLD_TAPS = 11
SB_VOLD_MU = 0.0004
SB_NUMOF_TS = 8000

# ---------------------------------------------------------------------------
# Multi-band CAP parameters (main_CAP_3band_totalB.m)
# ---------------------------------------------------------------------------
MB_NUMOFSYMBOLS = 1024 * 32
MB_M = 16                         # Modulation order per band
MB_CONSTELLATION = "QAM"
MB_ROLLOFF = 0.2
MB_RS = 300e6                     # Aggregate baud rate
MB_FS = 1.2e9                     # Sampling rate
MB_CF = 0.11                      # Compression factor
MB_SPAN = 8                       # Filter span in symbols
MB_SHAPE = "srrc"
MB_NUM_BANDS = 3

MB_SNR_DB = 25
MB_CHANNEL_FS = 100
MB_CHANNEL_FACTOR = 15
MB_CHANNEL_NONLINEAR = False

# ---------------------------------------------------------------------------
# Run mode
# ---------------------------------------------------------------------------
OFFLINE_FLAG = 1                  # 1 = process saved files, 0 = online hardware
USE_VIRTUAL_CHANNEL = 1           # 1 = generate RX via channel model, 0 = read file

# ---------------------------------------------------------------------------
# Hardware VISA addresses (reused from DMT_PY_NN)
# ---------------------------------------------------------------------------
M8190A_VISA_ADDR = "TCPIP0::localhost::5025::SOCKET"
M8190A_PORT = 5025
AWG_OUTPUT_ROUTE = "DAC"

OSC_VISA_ADDR = "TCPIP0::192.168.193.176::5025::SOCKET"
OSC_CHANNEL = "CHAN1"
OSC_TIMEBASE_SCALE = 80e-6

# ---------------------------------------------------------------------------
# NN post-equalizer (reused from DMT_PY_NN)
# ---------------------------------------------------------------------------
USE_NN = 0                        # 0 = conventional equalizer, 1 = NN
POSTEQ_FLAG = 0                   # 0 = none, 1 = RNN/GRU, 2 = MLP, 3 = Volterra

NN_TX_FILE = NN_DIR / "Txdata_NN.txt"
NN_RX1_FILE = NN_DIR / "Rxdata_NN1.txt"
NN_OUTPUT1_FILE = NN_DIR / "Rxdata_afterNN1.txt"
NN_MODEL_TEMP = NN_DIR / "trained_model_temp.pth"

# ---------------------------------------------------------------------------
# Plotting / records
# ---------------------------------------------------------------------------
PLOT_SHOW = False
PLOT_SAVE = True
PLOT_DPI = 300

# ---------------------------------------------------------------------------
# Keithley 2400 SourceMeter (RS-232 / USB-to-RS232)
# ---------------------------------------------------------------------------
K2400_PORT = "COM1"               # Serial port name (auto-detect if left empty)
K2400_BAUDRATE = 9600             # Default RS-232 baud rate for the 2400
K2400_TIMEOUT = 5.0               # Serial read timeout in seconds
K2400_SOURCE_MODE = "voltage"     # "voltage" or "current"
K2400_LEVEL = 0.0                 # Source level (V in voltage mode, A in current mode)
K2400_COMPLIANCE = 0.1            # Compliance limit (A in voltage mode, V in current mode)
K2400_NPLC = 1.0                  # Measurement integration time


# ---------------------------------------------------------------------------
# Grid scan parameters (bias vs Vpp / SNR)
# ---------------------------------------------------------------------------
GRID_SCAN_PARAM1_NAME = "bias_voltage"  # displayed name / CSV header
GRID_SCAN_PARAM1_MODE = "voltage"       # "voltage" or "current" (Keithley source mode)
GRID_SCAN_PARAM1_START = 0.0
GRID_SCAN_PARAM1_STOP = 1.0
GRID_SCAN_PARAM1_STEP = 0.2
GRID_SCAN_VPP_START = 0.1
GRID_SCAN_VPP_STOP = 0.5
GRID_SCAN_VPP_STEP = 0.1
GRID_SCAN_RUN_MODE = "singleband"       # "singleband" or "multiband"
GRID_SCAN_REPEATS = 1                   # repeats per grid point


# ---------------------------------------------------------------------------
# Helper constants
# ---------------------------------------------------------------------------
SB_SYMBOL_RATE = SB_AWG_SAMPLE_RATE / SB_UPSAMPLENO
