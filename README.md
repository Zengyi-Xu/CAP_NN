# DMT Communication System — Python Rewrite

**Version: 0.2.0** (see [CHANGELOG.md](CHANGELOG.md) for release notes)

This project rewrites the original MATLAB DMT communication code in Python, preserving the full communication flow and supporting direct control of the M8190A AWG and Keysight oscilloscope (USB-B) via pyvisa.

New features:
- Multiple pilot patterns (training_only / comb / mesh)
- Built-in virtual channel (first-order low-pass, AWGN, third-order nonlinearity, delay)
- NN post-equalizer (disabled by default, can be enabled manually)
- Automatic plotting with display in Spyder
- Generates both CodePlot v5 editable scripts + NPZ data for later figure refinement
- Each test automatically generates a unique ID and saves a transmission record
- Keithley 2400 source-meter control over RS-232 / USB-to-RS232
- Automated 2-D parameter grid scan (Keithley bias vs AWG Vpp) with CSV summary and CodePlot contour plots

## Project Structure

```
dmt_python/
├── config.py                   # DMT / hardware / plotting / NN / Keithley / grid-scan parameters
├── dmt_core.py                 # DMT modulation, demodulation, bitloading, SNR/BER/SER estimation
├── awg_m8190a.py               # M8190A pyvisa control
├── oscilloscope.py             # Oscilloscope USB-B pyvisa readout
├── nn_equalizer.py             # ZY_BiGRU_GPU wrapper
├── record.py                   # Transmission records and unique ID
├── plot_adapter.py             # Generate CodePlot v5 scripts and NPZ data
├── codeplot_v5.py              # CodePlot v5 figure layout tool
├── main.py                     # Full flow entry point
├── utils.py                    # File I/O, synchronization, resampling, plotting
├── virtual_channel.py          # Virtual channel simulation
├── keithley2400_controller.py  # Keithley 2400 RS-232/USB driver
├── grid_scan.py                # Automated bias vs Vpp grid scan engine
├── requirements.txt
├── data/                  # Data files
│   ├── txdata/
│   ├── rxdata/
│   ├── codeplot_assets/   # CodePlot scripts and data per test (organized by run_id)
│   ├── records/           # JSON/txt record per test
│   └── nn/                # NN scripts and models
└── README.md
```

## Install Dependencies

A virtual environment is recommended:

```bash
python -m venv .venv
.venv\Scripts\activate.bat   # Windows cmd
# or .venv\Scripts\Activate.ps1  # Windows PowerShell
# or source .venv/Scripts/activate  # Git Bash

pip install numpy scipy matplotlib pyvisa PyVISA-py tqdm pyserial
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

If `torch` installation on Windows reports "File name extension too long", enable long path support (Group Policy `Computer Configuration > Administrative Templates > System > Filesystem > Enable Win32 long paths`), or use a shorter project path.

If using Keysight VISA, install Keysight IO Libraries Suite and confirm the USB-B driver is installed; otherwise `PyVISA-py` can drive USBTMC directly.

## Quick Start

### 1. Run the full flow offline (no hardware, NN not called by default)

```bash
python main.py --offline 1 --use-awg 0
```

NN is not called by default to reduce runtime. To enable NN post-equalization:

```bash
python main.py --offline 1 --use-awg 0 --use-nn 1
```

In offline mode the tool reads existing oscilloscope files under `data/rxdata/` by default; when `--use-virtual-channel 1` is set, the virtual channel generates RX directly.

#### Locate offline waveforms by suffix

Online-acquired waveforms are now saved as `rawOSC_QPSK_SNRest_<run_id>.txt` and `rawOSC_DMT_<run_id>.txt` (`run_id` looks like `20260806_013459_dee786`). For offline rerun, enter only the last 6 characters to automatically locate both stages of the same experiment:

```bash
python main.py --offline 1 --use-awg 0 --run-suffix dee786
```

This is equivalent to specifying both paths:

```bash
python main.py --offline 1 --use-awg 0 \
  --qpsk-rx data/rxdata/rawOSC_QPSK_SNRest_20260806_013459_dee786.txt \
  --bpl-rx  data/rxdata/rawOSC_DMT_20260806_013459_dee786.txt
```

To reprocess only one stage, combine with `--step`:

```bash
python main.py --offline 1 --step step2 --run-suffix dee786
python main.py --offline 1 --step step4 --run-suffix dee786
```

> Tip: When both `--run-suffix` and `--qpsk-rx` / `--bpl-rx` are present, explicit paths take precedence.

### 2. Online run (M8190A + oscilloscope connected)

Edit the VISA addresses in `config.py`:

```python
M8190A_VISA_ADDR = "TCPIP0::192.168.1.10::5025::SOCKET"  # change to M8190A actual IP
OSC_VISA_ADDR = ""   # leave empty to auto-detect the first USB instrument
```

Then run:

```bash
python main.py --offline 0 --use-awg 1
```

Add `--use-nn 1` to enable NN in online mode.

### 3. Single-step run

```bash
python main.py --step step1  # generate QPSK TX only
python main.py --step step2  # receive QPSK only and estimate SNR
python main.py --step step3  # generate Bitloading TX only
python main.py --step step4  # receive Bitloading only and demodulate
```

### 4. Use the virtual channel (instrument-free debugging)

Set in `config.py`:

```python
USE_VIRTUAL_CHANNEL = 1
VIRTUAL_CHANNEL_FC = 2.0e9          # first-order low-pass cutoff frequency
VIRTUAL_CHANNEL_SNR_DB = 30         # receiver SNR
VIRTUAL_CHANNEL_NONLINEARITY = 0.02 # third-order nonlinearity coefficient
VIRTUAL_CHANNEL_DELAY = 5           # sample delay
VIRTUAL_CHANNEL_ATTENUATION = 0.9   # linear attenuation
```

Then run:

```bash
python main.py --offline 1 --use-virtual-channel 1 --use-awg 0
```

The virtual channel replaces oscilloscope readout and generates RX directly from TX, which is convenient for debugging and maintenance without hardware.

### 5. Switch pilot patterns

Set in `config.py`:

```python
PILOT_PATTERN = "comb"   # "training_only" / "comb" / "mesh"
PILOT_COMB_SPACING = 8   # comb pilot subcarrier spacing
PILOT_MESH_FREQ_SPACING = 8
PILOT_MESH_TIME_SPACING = 5
```

- `training_only`: uses only the first few training-symbol columns (same as the original MATLAB); all subcarriers carry data.
- `comb`: fixes certain subcarriers to carry known pilots for the entire duration, suitable for fast time-varying channels; during bitloading the bit count of these pilot subcarriers is automatically set to 0.
- `mesh`: time-frequency 2D sparse pilots with lower overhead; interpolation accuracy depends on pilot density. Verify with the virtual channel before moving to hardware.

> Tip: comb SNR estimation skips pure pilot subcarriers and fills with the SNR of the nearest data subcarrier; `mean SNR` now uses `nanmean` to avoid pilot positions inflating the average.

## Auto Plotting and CodePlot v5 Scripts

By default, figures are displayed directly in the **Spyder Plots pane** (via `plt.show()`).

At the same time, every test saves under `data/codeplot_assets/<run_id>/`:

```
data/codeplot_assets/<run_id>/
├── data/                # *.npz data for each figure
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
└── scripts/             # editable Python script for each figure
    ├── SNRest_QPSK_time.py
    ├── SNRest_QPSK_spec.py
    ...
    └── constellation_by_order.py
```

To edit a figure later, open `codeplot_v5.py`, click **📂 Load Script**, and select the corresponding `scripts/<name>.py`. The script uses the `fig` variable, and CodePlot v5 automatically loads data from `../data/<name>.npz` via relative paths.

### Currently Generated Figures

| Name | Description |
|------|-------------|
| `SNRest_QPSK_time` / `SNRest_QPSK_spec` / `SNRest_QPSK_constellation` | QPSK TX time / frequency / constellation |
| `SNRest_QPSK_rx_spec` | QPSK RX spectrum |
| `SNRest_QPSK_nonlinearity` | TX-RX amplitude nonlinearity scatter/density (blue-green `GnBu`) |
| `SNR_QPSK` | Per-carrier SNR estimated in the QPSK stage (dB) |
| `bit_power_loading` | Per-subcarrier SNR, bit loading, power loading, with ratio annotated |
| `DMT_bitloading_Tx_time` / `DMT_bitloading_Tx_spec` | Bitloading TX time / spectrum |
| `DMT_bitloading_Rx_spec` / `DMT_bitloading_nonlinearity` | Bitloading RX spectrum / nonlinearity |
| `SNR_compare` | QPSK-estimated SNR vs final recovered SNR (dB) |
| `constellation_density` | Constellation point density heatmap by modulation order (blue-green `GnBu`) |
| `constellation_by_order` | RX constellation scatter by modulation order (one subplot per order) |
| `ser_ber_per_carrier` | Per-subcarrier SER and BER |

> Note: The spectrogram (time-frequency plot) has been removed because the rendering quality was poor.

> **Troubleshooting missing plots in Spyder:**
> 1. Confirm `matplotlib-inline` is installed in the Spyder environment (`pip install matplotlib-inline`).
> 2. In Spyder choose **Tools > Preferences > IPython console > Graphics** and set **Backend** to **Inline**.
> 3. Restart the IPython Console and run `main.py`.

For pure command-line / headless environments, set:

```bash
set MPLBACKEND=Agg   # Windows cmd
# or
$env:MPLBACKEND="Agg" # PowerShell
# or
export MPLBACKEND=Agg # Git Bash
```

Or disable display directly in `config.py`:

```python
PLOT_SHOW = False
```

## Experimental Data Visualization GUI

`dmt_gui.py` provides a desktop GUI for browsing saved experiment data and running tests directly (no extra dependencies: tkinter + matplotlib, with high-DPI support):

```bash
.venv\Scripts\python dmt_gui.py
```

It contains six tabs:

| Tab | Content |
|-----|---------|
| Waveform Time/Frequency | TX/RX time waveforms and spectra for QPSK probing and DMT Bitloading |
| DMT Symbol Modulation | Bit/Power Loading, QPSK constellation, RX constellation by modulation order, density heatmap |
| Transmission Results | Experiment record table (rate/BER/SER/SNR), SNR comparison, per-subcarrier SER/BER, TX-RX nonlinearity, experiment trends |
| Run Test | Select mode (online / offline / virtual channel) and step, then call `main.py` directly; log is shown live and view refreshes on completion |
| Keithley 2400 | Connect and control a Keithley 2400 source meter over RS-232 or USB-to-RS232 |
| Grid Scan | Automated 2-D sweep of Keithley bias vs AWG Vpp with CSV summary and contour plots |

### Browse Experiment Data

- Switch experiments via the `run_id` dropdown at the top; in the "Transmission Results" tab you can also click a row in the record table.
- Data comes from `data/codeplot_assets/<run_id>/data/*.npz` and `data/records/record_*.json`.
- After running a new experiment, click "Refresh Data" to update the dropdown and plots.

### Export Images

Each plot tab provides on the left:

- **Export Current Image**: save the currently displayed figure as PNG / PDF / SVG; default directory `data/plots/<run_id>/`.
- **Save All Images**: one-click save of all figures in the current tab to `data/plots/<run_id>/`.

When saving, a same-name `.json` file is generated recording the array names, shapes, dtypes used in the figure, and the original `.npz` data path. If the target directory already contains a same-name image, a confirmation dialog is shown.

### Export Plot Data

Each plot tab also provides:

- **Export Current Plot Data**: export the current figure's `.npz` data as `.xlsx`, with different arrays in different sheets.
- **Export All Plot Data**: one-click export of all figure data in the current tab to a single `.xlsx`.

Complex arrays are split into real/imag columns; 0-D scalars, 1D, and 2D arrays are supported.

### Run Tests

In the "Run Test" tab:

1. Choose run mode: online / offline / offline + virtual channel.
2. Choose flow step: full flow (all) or single step (step1 ~ step4).
3. (Optional) Enable NN post-equalization.
4. In offline / virtual-channel mode, you can fill in:
   - **Last six digits of STEP2 waveform ID**: locate the QPSK probing RX file
   - **Last six digits of STEP4 waveform ID**: locate the Bitloading RX file
   - **Full run-id / filename**: automatically recognize the stage and strip `.txt` / `.json` suffixes
5. Modify `config.py` parameters (e.g. `RATIO`, `PLOT_SAVE`) in the parameter panel below, then click "Save to config.py".
6. Click "Start Test" to call `main.py`; the log is shown live on the right.

> Tip: When running offline with a waveform suffix specified, the GUI automatically uses the `run_id` from the source RX file so that generated records and image directories match the original waveform ID.

### Keithley 2400 Control

The "⚡ Keithley 2400" tab controls a Keithley 2400 SourceMeter through a COM port. The port can be a physical RS-232 port or a USB-to-RS232 adapter (FTDI / Prolific / CH340 / CP210x).

1. Select the COM port and baud rate, then click **Connect**.
2. Choose **Source Mode**: `voltage` (V source, current measure) or `current` (current source, voltage measure).
3. Set **Level** (V or A), **Compliance** limit, and **NPLC** integration time.
4. Click **Apply Settings**, then **Output ON**.
5. Click **Measure** to read voltage, current, resistance, and time.
6. The output is automatically turned off when the GUI closes.

Defaults are defined in `config.py`:

```python
K2400_PORT = "COM1"
K2400_BAUDRATE = 9600
K2400_SOURCE_MODE = "voltage"
K2400_LEVEL = 0.0
K2400_COMPLIANCE = 0.1
K2400_NPLC = 1.0
```

### Grid Scan (Bias vs Vpp)

The "🔲 Grid Scan" tab automates 2-D parameter sweeps:

- **Parameter 1**: Keithley-controlled bias (voltage or current).
- **Parameter 2**: AWG output amplitude `Vpp`.

For each grid point the DMT pipeline is executed and the final measurement step is repeated N times and averaged. Results are saved under `data/grid_scans/<scan_id>/`:

```
data/grid_scans/<scan_id>/
├── config.json
├── summary.csv
├── contour_ber.png
├── contour_ser.png
├── contour_snr.png
└── codeplot_assets/
    ├── data/
    └── scripts/          # CodePlot v5 loadable contour scripts
```

Usage:

1. Set the Keithley bias range (start / stop / step) and source mode.
2. Set the AWG Vpp range.
3. Choose pipeline mode:
   - `step1-4`: full DMT-NN flow; Step 4 is repeated and averaged.
   - `step1-2`: QPSK probing only; Step 2 is repeated and averaged.
4. Set **Repeats** for the final measurement step.
5. Select **Offline / Virtual channel / Use NN** as needed.
6. Click **Start Grid Scan**.
7. After completion, select a scan ID in the right panel to view the CSV summary table and the contour plot (BER / SER / SNR).

> Note: In offline mode, repeated measurements use the same resolved RX file unless multiple matching files exist. For meaningful averaging, supply distinct RX files or run online where each repeat captures fresh scope data.

### FAQ

- **`tuple index out of range` when exporting bit/power loading**: fixed in the current version. If it still occurs, fully close the GUI, delete `__pycache__`, and restart; the export failure dialog shows the full traceback, which can be pasted for further diagnosis.
- **Taskbar still shows the feather icon**: Windows icon cache may delay updates. Try deleting `data/.gui_icon.ico` and `data/.gui_icon.png` and restarting the GUI, or log out / restart the system.

## Transmission Records

Every full test automatically generates a unique ID `run_id` (like `20260731_024911_164ad1`) and saves in `data/records/`:

- `record_<run_id>.json`: full parameters and results (pilots, virtual channel, rate, BER, SER, SNR, ratio, NN, etc.)
- `record_<run_id>.txt`: text summary

Record contents include:
- Test ID and timestamp
- Pilot pattern, whether virtual channel / NN was used
- Virtual channel parameters (fc, SNR, nonlinearity, delay, attenuation)
- Channel-probing estimated rate
- Final bitloading rate
- Final BER / SER
- Recovered SNR (dB)
- ratio

## Key Parameters

Adjust in `config.py`:

| Parameter | Meaning |
|-----------|---------|
| `CARRIERNO` | Total number of subcarriers (1056) |
| `ZEROPAD1` | Zero padding on each side (16) |
| `UPSAMPLENO` | Upsampling factor (2) |
| `DATANO_QPSK` | Number of QPSK symbols (100) |
| `DATANO_BPL` | Number of Bitloading symbols (200) |
| `AWG_SAMPLE_RATE` | AWG sample rate 8 GSa/s |
| `OSC_SAMPLE_RATE` | Oscilloscope sample rate 10 GSa/s |
| `M8190A_VISA_ADDR` | AWG VISA address |
| `OSC_VISA_ADDR` | Oscilloscope USB VISA address |
| `POSTEQ_FLAG` | Post-equalization type: 0=no NN, 1=RNN/GRU, 2=MLP, 3=Volterra; when non-zero main.py calls NN by default |
| `USE_NN` | Automatically determined by `POSTEQ_FLAG`; whether to call NN post-equalization by default |
| `PLOT_SHOW` / `PLOT_SAVE` | Whether to show / save figures |


## Notes

1. When connecting the oscilloscope via USB-B, ensure Keysight IO Libraries Suite has recognized the device and the resource string starts with `USB0::`.
2. When connecting the M8190A via TCPIP socket, confirm the firewall allows port 5025.
3. NN is disabled by default to speed up execution; add `--use-nn 1` when post-equalization is needed. The first run trains for 8 epochs.
4. If `PyVISA` cannot find resources, test in Python first:

```python
import pyvisa
rm = pyvisa.ResourceManager()
print(rm.list_resources())
```
