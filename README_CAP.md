# UW_APSK_CAP_PY — Python CAP Transceiver for Underwater Visible-Light Communication

This project ports the MATLAB UW_APSK CAP (Carrierless Amplitude Phase) transceiver
to Python, reusing the engineering framework from `DMT_PY_NN`.

## Scope

- **Single-band CAP** — ports `oldcapAPSKTxRx20220406.m`
- **Multi-band CAP** — ports `main_CAP_3band_totalB.m`
- **APSK/QAM constellations** — ports `GS_CCSDSmodulation_cons.m` / `GS_CCSDSdemodulation_cons.m`
- **LMS / LMS+Volterra equalizers** — ports `LMS_1DownS_Testnan.m` / `LMS_volterra_1DownS_Testnan.m`
- **VLC channel model** — ports `vlc_channel.m`
- **NN post-equalizer** — ports the SCAP_DNNpy2 workflow (multi-band CAP DNN)

## File mapping

| Python file | MATLAB source | Purpose |
|---|---|---|
| `constellation.py` | `GS_CCSDSmodulation_cons.m`, `GS_CCSDSdemodulation_cons.m` | APSK/QAM mapping/demapping |
| `cap_core.py` | `CAPmod.m`, `cap_gen.m`, `shaping_fildes.m`, `Pulse_shaping_ZY.m` | CAP modulation, multi-band synthesis |
| `cap_rx.py` | `CAPmatch_filter.m`, `mdb_match_filter.m` | CAP matched filter / demodulation |
| `equalizer.py` | `LMS_1DownS_Testnan.m`, `LMS_volterra_1DownS_Testnan.m` | LMS and LMS+Volterra equalizers |
| `channel.py` | `vlc_channel.m` | VLC channel + AWGN |
| `config_cap.py` | — | CAP-specific parameters |
| `main_cap.py` | — | CLI entry point |
| `data/nn/CAP_multiband_NN.py` | `SCAP_DNNpy2` | PyTorch BiGRU post-equalizer |
| `nn_cap_equalizer.py` | — | Subprocess wrapper for CAP NN |

## Requirements

```bash
pip install -r requirements.txt
```

`torch` is required only for the NN equalizer (`--use-nn`). Use the CPU/CUDA wheel
appropriate for your machine.

## Quick start

### Run tests

```bash
python test_cap.py
```

### Single-band CAP offline simulation

```bash
python main_cap.py --mode singleband --order 64 --constellation QAM --snr 27 --seed 100
```

### Multi-band CAP offline simulation

```bash
python main_cap.py --mode multiband --order 16 --constellation QAM --snr 25 --seed 1
```

### Multi-band with NN post-equalizer

```bash
python main_cap.py --mode multiband --order 16 --use-nn
```

### GUI

```bash
python cap_gui.py
```

提供参数配置、一键运行、星座图/波形/频谱显示和结果面板。

## Configuration

Edit `config_cap.py` for:

- Symbol rates, sample rates, roll-off factors
- AWG / oscilloscope VISA addresses
- Virtual channel SNR, nonlinearity, fading
- LMS/Volterra tap counts and step sizes
- NN training parameters (inside `data/nn/CAP_multiband_NN.py`)

## Golden-reference verification

`test_cap.py` compares the Python-generated single-band CAP waveform with the
MATLAB-generated `data32QAM.txt`. The normalised maximum difference is below
`1e-6`, confirming bit-exact equivalence of the pulse-shaping path.

## Notes

- The MATLAB project uses **CAP**, not DMT/OFDM. This Python port therefore
  implements CAP modulation/demodulation, not IFFT/FFT-based multicarrier.
- Multi-band CAP bands intentionally overlap; raw matched-filter BER is high.
  Use LMS (`--no-lms` to disable) or NN (`--use-nn`) for separation/equalization.
